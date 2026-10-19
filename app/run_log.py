"""
app.run_log
===========

Run logging for the app: one flat record per run (inputs, model outputs and,
optionally, what you measured on the bench), written to a Google Sheet by a
service-account "bot" and kept in a local buffer that can be downloaded as CSV.

Why a fixed schema
------------------
Every row has the same columns in the same order (``COLUMNS``, versioned by
``SCHEMA_VERSION``), so the sheet stays usable as a dataset.  The optional
``meas_*`` columns are for what you actually observed; they are what the
planned Bayesian calibration will use later, so fill them in when you can.

Google Sheets setup (one time)
------------------------------
1. In Google Cloud, create a service account and download its JSON key.
2. Enable the Google Sheets API (and Drive API) for that project.
3. Share your sheet with the service account's e-mail (its ``client_email``) as
   *Editor*.  That e-mail is the "bot": it can only see sheets shared with it.
4. Give the app the key either in ``.streamlit/secrets.toml`` under
   ``[gcp_service_account]`` or by uploading the JSON in the app (kept in memory
   for the session only).

Security notes
--------------
* The private key is never displayed or logged; only the bot's e-mail is shown.
* Cell text that starts with ``= + - @`` is prefixed with an apostrophe so a
  note cannot be interpreted as a spreadsheet formula.
* ``.streamlit/secrets.toml`` and any key file belong in ``.gitignore``.

Everything here is independent of Streamlit, so it is unit-testable with a fake
client (the Google API itself is only reachable with real credentials).
"""

from __future__ import annotations

import csv
import io
import math
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

SCHEMA_VERSION = 1

# --------------------------------------------------------------------------
# Schema: column order is the order of the sheet
# --------------------------------------------------------------------------

META_COLUMNS = ["timestamp_utc", "run_id", "schema_version", "mode", "label"]

INPUT_COLUMNS = [
    "m_feed_kg", "n_pseudo", "sg_feed", "tbp_json", "T_lo_c", "T_hi_c", "r_key", "gap_c",
    "p_top_mbar", "R", "R_mode", "q_heater_w", "efficiency", "T_condenser_c",
    "D_mm", "H_pack_m", "packing_kind", "packing_size_mm", "void_fraction",
    "k_hetp", "wetted_fraction", "k_static", "k_dynamic", "loading_coeff", "Ea_kj_mol", "T_limit_c",
]

OUTPUT_COLUMNS = [
    # live estimate for the current column
    "N_total", "N_packing", "R_min", "alpha", "hetp_mm", "N_available", "R_needed", "percent_flood",
    "dp_mbar", "p_pot_mbar", "holdup_g", "head_T_lo_c", "head_T_hi_c", "cut_available_g",
    # batch simulation
    "stop_reason", "t_batch_min", "t_collect_min", "T_pot_max_c", "tau_equiv_min", "target_mass_g",
    "target_purity", "target_recovery", "residue_g", "light_ends_min",
    # design
    "D_solved_mm", "D_final_mm", "H_pack_design_m", "limiting",
]

MEASURED_COLUMNS = [
    "meas_mass_g", "meas_purity", "meas_recovery", "meas_dp_mbar", "meas_flooded",
    "meas_head_T_c", "meas_batch_min", "notes",
]

COLUMNS = META_COLUMNS + INPUT_COLUMNS + OUTPUT_COLUMNS + MEASURED_COLUMNS
_KNOWN = set(COLUMNS)


class RunLogError(Exception):
    """Anything that stops a run from being logged, with a message fit for the user."""


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------

def build_record(mode: str, inputs: dict, outputs: dict | None = None, measured: dict | None = None,
                 label: str = "", run_id: str | None = None, now: datetime | None = None) -> dict:
    """Assemble one record.  Unknown keys are rejected so the schema cannot drift silently.

    ``mode`` is "rating", "design" or "estimate" (the live numbers without a simulation).
    """
    if mode not in ("rating", "design", "estimate"):
        raise ValueError("mode must be 'rating', 'design' or 'estimate'.")
    rec = {c: "" for c in COLUMNS}
    stamp = (now or datetime.now(timezone.utc)).replace(microsecond=0)
    rec.update(timestamp_utc=stamp.isoformat(), run_id=run_id or uuid.uuid4().hex[:10],
               schema_version=SCHEMA_VERSION, mode=mode, label=label)
    for part in (inputs, outputs or {}, measured or {}):
        for k, v in part.items():
            if k not in _KNOWN:
                raise ValueError(f"Unknown column '{k}': add it to the schema first.")
            rec[k] = v
    return rec


def _cell(v):
    """One value in a form a spreadsheet and a CSV can both hold."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if hasattr(v, "item"):                               # numpy scalar -> python
        v = v.item()
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return ""
        return float(f"{v:.6g}")
    if isinstance(v, str) and v[:1] in ("=", "+", "-", "@"):
        return "'" + v                                   # never let text act as a formula
    return v


def record_to_row(rec: dict) -> list:
    """The record as a list in ``COLUMNS`` order."""
    return [_cell(rec.get(c, "")) for c in COLUMNS]


def parse_sheet_ref(ref: str) -> str:
    """Spreadsheet ID from a full Google Sheets URL or a bare ID."""
    ref = (ref or "").strip()
    m = re.search(r"/d/([a-zA-Z0-9_-]+)", ref)
    if m:
        return m.group(1)
    if re.fullmatch(r"[a-zA-Z0-9_-]{20,}", ref):
        return ref
    raise RunLogError("Could not read a spreadsheet ID from that. Paste the sheet's URL or its ID.")


# --------------------------------------------------------------------------
# Local buffer (always available)
# --------------------------------------------------------------------------

@dataclass
class MemoryLog:
    """Records kept for the session; downloadable as CSV."""

    records: list = field(default_factory=list)

    def append(self, rec: dict) -> None:
        self.records.append(rec)

    def __len__(self) -> int:
        return len(self.records)

    def to_csv(self) -> str:
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(COLUMNS)
        for r in self.records:
            w.writerow(record_to_row(r))
        return buf.getvalue()


# --------------------------------------------------------------------------
# Google Sheets
# --------------------------------------------------------------------------

class GSheetLogger:
    """Appends records to one worksheet of a Google spreadsheet.

    ``client`` is a gspread-style client (anything with ``open_by_key``), which
    lets tests pass a fake.  Use ``from_service_account_info`` for the real thing.
    """

    def __init__(self, client, sheet_ref: str, worksheet: str = "runs", bot_email: str = ""):
        self._client = client
        self.sheet_id = parse_sheet_ref(sheet_ref)
        self.worksheet_name = worksheet.strip() or "runs"
        self.bot_email = bot_email
        self._ws = None

    @classmethod
    def from_service_account_info(cls, info: dict, sheet_ref: str, worksheet: str = "runs") -> "GSheetLogger":
        try:
            import gspread
        except ImportError as exc:
            raise RunLogError("The 'gspread' package is not installed (pip install gspread google-auth).") from exc
        for key in ("client_email", "private_key", "token_uri"):
            if not info.get(key):
                raise RunLogError(f"The service-account key is missing '{key}'. Use the JSON file Google gave you.")
        try:
            client = gspread.service_account_from_dict(dict(info))
        except Exception as exc:
            raise RunLogError(f"Could not use that service-account key ({type(exc).__name__}).") from exc
        return cls(client, sheet_ref, worksheet, bot_email=info["client_email"])

    # -- connection ---------------------------------------------------------
    def connect(self):
        """Open the spreadsheet, create the worksheet and header if needed."""
        if self._ws is not None:
            return self._ws
        try:
            import gspread.exceptions as gx
        except ImportError:                              # fake clients in tests
            gx = None
        try:
            sh = self._client.open_by_key(self.sheet_id)
            try:
                ws = sh.worksheet(self.worksheet_name)
            except Exception as exc:
                if gx is not None and isinstance(exc, gx.WorksheetNotFound):
                    ws = sh.add_worksheet(title=self.worksheet_name, rows=1000, cols=len(COLUMNS))
                else:
                    raise
            header = ws.row_values(1)
        except RunLogError:
            raise
        except Exception as exc:
            raise RunLogError(self._explain(exc)) from exc
        if not header:
            try:
                ws.append_row(COLUMNS, value_input_option="USER_ENTERED")
            except Exception as exc:
                raise RunLogError(self._explain(exc)) from exc
        elif header != COLUMNS:
            raise RunLogError(
                f"Worksheet '{self.worksheet_name}' already has a different header (not schema v{SCHEMA_VERSION}). "
                "Use a new worksheet name, or clear the sheet.")
        self._ws = ws
        return ws

    def append(self, rec: dict) -> str:
        """Append one record; returns the range Google reports (e.g. ``runs!A5:BH5``)."""
        ws = self.connect()
        try:
            resp = ws.append_row(record_to_row(rec), value_input_option="USER_ENTERED")
        except Exception as exc:
            raise RunLogError(self._explain(exc)) from exc
        try:
            return resp["updates"]["updatedRange"]
        except (KeyError, TypeError):
            return ""

    def _explain(self, exc: Exception) -> str:
        """A user-facing message with the likely fix."""
        name = type(exc).__name__
        text = str(exc)
        if name == "SpreadsheetNotFound" or "not found" in text.lower():
            return (f"Spreadsheet not found. Check the ID/URL, and share the sheet with {self.bot_email or 'the bot'} "
                    "as Editor.")
        if "403" in text or "PERMISSION" in text.upper():
            return (f"Permission denied. Share the sheet with {self.bot_email or 'the bot'} as Editor, and enable "
                    "the Google Sheets API for the project.")
        if "429" in text or "quota" in text.lower():
            return "Google's rate limit was hit. Wait a minute and log again."
        return f"Google Sheets error ({name}): {text[:200]}"
