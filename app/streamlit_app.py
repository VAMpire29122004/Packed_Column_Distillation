"""
Packed fractional distillation column for WPPO - Streamlit app
==============================================================

Run from the repository root:

    streamlit run app/streamlit_app.py

Layout
------
* Sidebar : every input (feed, cut, operating point, column and packing,
  calibration constants).  Live results update as you edit.
* Overview: key numbers, a verdict on the current column, the live schematic.
* Rating  : simulate the batch (cuts, purity, recovery, timing) and scrub the
            schematic through time.
* Design  : solve diameter and packed height for the target cut, compare packing
            sizes and pressures, and apply a design to the inputs.
* Plots   : feed curve, vacuum boiling points, stages vs reflux, hydraulics.
* Run log : record each run (inputs, outputs and optional bench measurements) to
            a local buffer and, through a service-account "bot", to a Google Sheet.

Heavy calculations (batch simulation about 1 s, design about 4 s) run only when
you press their button; everything else recomputes on each edit.  A banner tells
you when stored results no longer match the current inputs.
"""

from __future__ import annotations

import hashlib
import io
import math
import re
import sys
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # repo root: `fdc` and `app` import

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from app import plots
from app.run_log import GSheetLogger, MemoryLog, RunLogError, build_record
from app.schematic import SchematicState, render_schematic, state_from_batch, state_from_design
from fdc.batch_pot import BatchSpec, ThermalModel, simulate_batch
from fdc.design_mode import DesignTargets, compare_packings, design_column
from fdc.hetp import HETPModel, hetp as hetp_model, packed_height_required, stages_available
from fdc.holdup import HoldupModel, internal_flows, liquid_holdup
from fdc.hydraulics import HydraulicsModel, column_hydraulics
from fdc.packing import PackingSpec, column_geometry
from fdc.properties.feed import DEFAULT_WPPO_TBP, TBPCurve, characterise_feed
from fdc.properties.fluids import Components, mixture_properties
from fdc.properties.vle import analyse_cut
from fdc.stages import R_FACTOR_DEFAULT, SeparationSpec, cut_stage_requirement, required_reflux

st.set_page_config(page_title="Packed distillation column", layout="wide")


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def run_captured(fn, *args, **kwargs):
    """Call ``fn`` and return ``(result, [distinct warning messages])``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn(*args, **kwargs)
    msgs: list[str] = []
    for w in caught:
        text = str(w.message)
        if text not in msgs:
            msgs.append(text)
    return out, msgs


def fnum(x, nd: int = 1, unit: str = "") -> str:
    """Format a number, showing infinity and NaN readably."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "\u2014"
    if isinstance(x, float) and math.isinf(x):
        return "\u221e"
    return f"{x:.{nd}f}{(' ' + unit) if unit else ''}"


def parse_numbers(text: str) -> list[float]:
    """Comma- or space-separated numbers from a text box."""
    return [float(t) for t in re.split(r"[,\s;]+", text.strip()) if t]


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class AppInputs:
    tbp: tuple
    n_pseudo: int
    sg_feed: float | None
    m_feed_kg: float
    T_lo_c: float
    T_hi_c: float
    r_key: float
    gap_c: float
    p_top_mbar: float
    R_auto: bool
    R_manual: float
    q_heater_w: float
    efficiency: float
    T_condenser_c: float
    D_mm: float
    H_pack_m: float
    kind: str
    size_mm: float
    void_fraction: float | None
    k_hetp: float
    wetted_fraction: float | None
    k_static: float
    k_dynamic: float
    loading_coeff: float
    Ea_kj_mol: float
    T_limit_c: float


def fingerprint(inp: AppInputs) -> str:
    return hashlib.md5(repr(inp).encode()).hexdigest()[:10]


TBP_DEFAULT = pd.DataFrame(DEFAULT_WPPO_TBP, columns=["Cumulative mass %", "Temperature \u00b0C (atm-equiv.)"])


def apply_design_to_inputs() -> None:
    """Callback: copy the designed diameter and height into the column inputs."""
    d = st.session_state["design"]["d"]
    st.session_state["D_mm"] = float(d.D_mm)
    st.session_state["H_pack_m"] = float(round(max(d.H_pack_m, 0.05), 2))


def read_sidebar() -> AppInputs:
    st.session_state.setdefault("D_mm", 50.0)          # set here, not as widget defaults, because the
    st.session_state.setdefault("H_pack_m", 0.60)      # design can overwrite them through a callback
    with st.sidebar:
        st.header("Inputs")
        with st.expander("Feed (WPPO)", expanded=True):
            m_feed = st.number_input("Charge to the pot (kg)", 0.05, 50.0, 1.0, 0.05, key="m_feed_kg")
            use_sg = st.checkbox("I have a measured density", value=True, key="use_sg")
            sg = st.number_input("Measured specific gravity (15.6 \u00b0C)", 0.60, 1.00, 0.80, 0.005, key="sg_feed",
                                 disabled=not use_sg, format="%.3f")
            n_pseudo = st.slider("Pseudo-components", 20, 40, 30, key="n_pseudo")
            st.caption("Boiling-point curve (replace the placeholder with your SimDis/TBP data):")
            tbp_df = st.data_editor(TBP_DEFAULT, num_rows="dynamic", hide_index=True, key="tbp_editor")
        with st.expander("Cut and separation", expanded=True):
            lo = st.number_input("Cut lower limit (\u00b0C, atmospheric-equivalent)", 0.0, 800.0, 150.0, 5.0, key="T_lo")
            hi = st.number_input("Cut upper limit (\u00b0C, atmospheric-equivalent)", 0.0, 800.0, 250.0, 5.0, key="T_hi")
            r_key = st.slider("Key recovery into each product", 0.80, 0.995, 0.95, 0.005, key="r_key")
            gap = st.slider("Grey-zone half-width (\u00b0C)", 2.0, 40.0, 10.0, 1.0, key="gap_c")
        with st.expander("Operating point", expanded=True):
            p_top = st.number_input("Top pressure P_top (mbar)", 1.0, 1013.25, 100.0, 5.0, key="p_top")
            r_auto = st.checkbox(f"Reflux ratio = {R_FACTOR_DEFAULT} \u00d7 minimum", value=True, key="R_auto")
            r_man = st.number_input("Reflux ratio R", 0.1, 100.0, 5.0, 0.5, key="R_manual", disabled=r_auto)
            q = st.number_input("Heater power (W)", 5.0, 3000.0, 100.0, 5.0, key="q_heater")
            eff = st.slider("Heater efficiency", 0.20, 1.00, 0.70, 0.05, key="eff")
            t_cond = st.number_input("Condenser coolant (\u00b0C)", -20.0, 60.0, 15.0, 1.0, key="T_cond")
        with st.expander("Column and packing", expanded=True):
            kind = st.radio("Packing", ["glass_beads", "steel_wool"], horizontal=True, key="kind",
                            format_func=lambda k: "Glass beads" if k == "glass_beads" else "Steel wool")
            size = st.number_input("Bead diameter (mm)" if kind == "glass_beads" else "Wire diameter (mm)",
                                   0.01, 25.0, 5.0 if kind == "glass_beads" else 0.05, 0.01 if kind == "steel_wool" else 0.5,
                                   key=f"size_{kind}", format="%.2f")
            use_eps = st.checkbox("I measured the void fraction", value=False, key="use_eps")
            eps = st.number_input("Void fraction", 0.20, 0.99, 0.40 if kind == "glass_beads" else 0.96, 0.01,
                                  key=f"eps_{kind}", disabled=not use_eps)
            d_mm = st.number_input("Column inner diameter (mm)", min_value=10.0, max_value=300.0, step=1.0, key="D_mm")
            h_pack = st.number_input("Packed height (m)", min_value=0.05, max_value=5.0, step=0.05, key="H_pack_m")
        with st.expander("Calibration and model constants"):
            k_hetp = st.number_input("HETP multiplier", 0.2, 5.0, 1.0, 0.1, key="k_hetp")
            use_wet = st.checkbox("Override wetted-area fraction (HETP and hold-up)", value=False, key="use_wet")
            wet = st.number_input("Wetted fraction", 0.05, 1.0, 0.5, 0.05, key="wet", disabled=not use_wet)
            k_stat = st.number_input("Static hold-up multiplier", 0.0, 3.0, 1.0, 0.1, key="k_static")
            k_dyn = st.number_input("Dynamic hold-up multiplier", 0.0, 3.0, 1.0, 0.1, key="k_dynamic")
            load_c = st.number_input("Flooding loading coefficient", 5.0, 60.0, 20.0, 1.0, key="loading_coeff")
            ea = st.number_input("Thermal degradation Ea (kJ/mol)", 50.0, 400.0, 200.0, 10.0, key="Ea")
            t_lim = st.number_input("Pot temperature limit (\u00b0C)", 100.0, 450.0, 300.0, 10.0, key="T_limit")

    # validate the boiling-point table before anything uses it
    d = tbp_df.dropna()
    try:
        pts = tuple(sorted((float(a), float(b)) for a, b in zip(d.iloc[:, 0], d.iloc[:, 1])))
        TBPCurve.from_points(pts)
    except (ValueError, IndexError) as exc:
        st.error(f"Boiling-point table problem: {exc}. It must start at 0 %, end at 100 %, and both columns must increase.")
        st.stop()
    if hi <= lo:
        st.error("The cut's upper limit must be above its lower limit.")
        st.stop()
    return AppInputs(
        tbp=pts, n_pseudo=n_pseudo, sg_feed=sg if use_sg else None, m_feed_kg=m_feed, T_lo_c=lo, T_hi_c=hi,
        r_key=r_key, gap_c=gap, p_top_mbar=p_top, R_auto=r_auto, R_manual=r_man, q_heater_w=q, efficiency=eff,
        T_condenser_c=t_cond, D_mm=d_mm, H_pack_m=h_pack, kind=kind, size_mm=size,
        void_fraction=eps if use_eps else None, k_hetp=k_hetp, wetted_fraction=wet if use_wet else None,
        k_static=k_stat, k_dynamic=k_dyn, loading_coeff=load_c, Ea_kj_mol=ea, T_limit_c=t_lim)


def models(inp: AppInputs):
    """Calibration objects built from the inputs."""
    hold = HoldupModel(wetted_fraction=inp.wetted_fraction or 1.0, k_static=inp.k_static, k_dynamic=inp.k_dynamic)
    hyd = HydraulicsModel(loading_coeff=inp.loading_coeff)
    het = HETPModel(k_hetp=inp.k_hetp, wetted_fraction=inp.wetted_fraction)
    return hold, hyd, het


# --------------------------------------------------------------------------
# Live calculation (rating of the current column)
# --------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def get_feed(tbp: tuple, n_pseudo: int, sg_feed: float | None):
    return characterise_feed(TBPCurve.from_points(tbp), n_cuts=n_pseudo, sg_feed=sg_feed)


@dataclass(frozen=True)
class Live:
    feed: object
    cut: object
    req: object
    R_use: float
    R_min: float
    gov: object
    fluid: object
    flows: object
    geom: object
    hetp: object
    N_avail: float
    R_needed: float
    H_required: float
    hyd: object
    hold: object
    avail_g: float
    notices: tuple


@st.cache_data(show_spinner=False)
def compute_live(key: str, _inp: AppInputs) -> Live:
    """Everything that can be computed instantly for the current column."""
    inp = _inp
    hold_m, hyd_m, het_m = models(inp)
    notices: list[str] = []
    feed = get_feed(inp.tbp, inp.n_pseudo, inp.sg_feed)
    sep = SeparationSpec(inp.r_key, inp.gap_c)

    probe, _ = run_captured(cut_stage_requirement, feed, inp.T_lo_c, inp.T_hi_c, inp.p_top_mbar, 1.0e6, sep)
    if not probe.boundaries:
        raise ValueError("This cut takes the whole feed: there is no separation to analyse.")
    R_use = R_FACTOR_DEFAULT * probe.R_min if inp.R_auto else inp.R_manual
    req, m1 = run_captured(cut_stage_requirement, feed, inp.T_lo_c, inp.T_hi_c, inp.p_top_mbar, R_use, sep)
    notices += m1
    gov = max(req.boundaries, key=lambda b: b.N_total if math.isfinite(b.N_total) else 1e9 + b.N_min)

    cut = analyse_cut(feed, inp.T_lo_c, inp.T_hi_c, inp.p_top_mbar, gap_c=inp.gap_c)
    _, x_cut = feed.cut_composition(inp.T_lo_c, inp.T_hi_c)
    fluid = mixture_properties(Components.from_feed(feed), x_cut, cut.T_hi_c, inp.p_top_mbar)
    flows = internal_flows(inp.q_heater_w, fluid.latent_kj_kg, R_use, inp.efficiency)

    spec = PackingSpec(inp.kind, inp.size_mm, void_fraction=inp.void_fraction)
    geom, m2 = run_captured(column_geometry, spec, inp.D_mm, inp.H_pack_m)
    notices += m2
    het, m3 = run_captured(hetp_model, geom, fluid, flows, gov.alpha, het_m)
    notices += m3
    N_avail = stages_available(inp.H_pack_m, het.hetp_m)
    R_needed = max(required_reflux(N_avail, b.N_min, b.R_min) for b in req.boundaries)
    hyd, m4 = run_captured(column_hydraulics, geom, fluid, flows, hold_m, hyd_m)
    notices += m4
    hold = liquid_holdup(geom, fluid, flows, hold_m)
    return Live(feed=feed, cut=cut, req=req, R_use=R_use, R_min=probe.R_min, gov=gov, fluid=fluid, flows=flows,
                geom=geom, hetp=het, N_avail=N_avail, R_needed=R_needed,
                H_required=packed_height_required(req.N_total, het.hetp_m) if math.isfinite(req.N_total) else math.inf,
                hyd=hyd, hold=hold, avail_g=inp.m_feed_kg * 1000.0 * feed.cut_mass_fraction(inp.T_lo_c, inp.T_hi_c),
                notices=tuple(dict.fromkeys(notices)))


# --------------------------------------------------------------------------
# Records for the run log
# --------------------------------------------------------------------------

def input_columns(inp: AppInputs, R_used: float) -> dict:
    return dict(
        m_feed_kg=inp.m_feed_kg, n_pseudo=inp.n_pseudo, sg_feed=inp.sg_feed, tbp_json=repr(inp.tbp).replace(" ", ""),
        T_lo_c=inp.T_lo_c, T_hi_c=inp.T_hi_c, r_key=inp.r_key, gap_c=inp.gap_c, p_top_mbar=inp.p_top_mbar,
        R=R_used, R_mode="auto" if inp.R_auto else "manual", q_heater_w=inp.q_heater_w, efficiency=inp.efficiency,
        T_condenser_c=inp.T_condenser_c, D_mm=inp.D_mm, H_pack_m=inp.H_pack_m, packing_kind=inp.kind,
        packing_size_mm=inp.size_mm, void_fraction=inp.void_fraction, k_hetp=inp.k_hetp,
        wetted_fraction=inp.wetted_fraction, k_static=inp.k_static, k_dynamic=inp.k_dynamic,
        loading_coeff=inp.loading_coeff, Ea_kj_mol=inp.Ea_kj_mol, T_limit_c=inp.T_limit_c)


def live_output_columns(live: Live) -> dict:
    return dict(
        N_total=live.req.N_total, N_packing=live.req.N_packing, R_min=live.R_min, alpha=live.gov.alpha,
        hetp_mm=live.hetp.hetp_m * 1000.0, N_available=live.N_avail, R_needed=live.R_needed,
        percent_flood=live.hyd.percent_flood, dp_mbar=live.hyd.dp_total_mbar, p_pot_mbar=live.hyd.p_pot_mbar,
        holdup_g=live.hold.mass_kg * 1000.0, head_T_lo_c=live.cut.T_lo_c, head_T_hi_c=live.cut.T_hi_c,
        cut_available_g=live.avail_g)


def batch_output_columns(res) -> dict:
    return dict(stop_reason=res.stop_reason, t_batch_min=res.t_batch_min, t_collect_min=res.t_collect_min,
                T_pot_max_c=res.T_pot_max_c, tau_equiv_min=res.tau_equiv_min,
                target_mass_g=res.target.mass_kg * 1000.0, target_purity=res.target.purity,
                target_recovery=res.target.recovery, residue_g=res.residue_kg * 1000.0,
                light_ends_min=res.light_ends_min)


def make_record(source: str, inp: AppInputs, live: Live, batch: dict | None, design: dict | None,
                label: str, measured: dict) -> dict:
    """One log record from the chosen source ('rating', 'design' or 'estimate')."""
    outputs = live_output_columns(live)
    run_id = None
    if source == "rating" and batch:
        outputs.update(batch_output_columns(batch["res"]))
        run_id = batch["run_id"]
    elif source == "design" and design:
        d = design["d"]
        outputs.update(N_total=d.N_total, N_packing=d.N_packing, R_min=d.R_min, alpha=d.alpha,
                       hetp_mm=d.hetp_m * 1000.0, percent_flood=d.percent_flood, dp_mbar=d.dp_mbar,
                       p_pot_mbar=d.p_pot_mbar, holdup_g=d.holdup_kg * 1000.0, D_solved_mm=d.D_solved_mm,
                       D_final_mm=d.D_mm, H_pack_design_m=d.H_pack_m, limiting=d.limiting)
        if d.verification is not None:
            outputs.update(batch_output_columns(d.verification))
        run_id = design["run_id"]
    R_used = design["d"].R if source == "design" and design else live.R_use
    return build_record(source, input_columns(inp, R_used), outputs, measured, label=label, run_id=run_id)


def log_record(rec: dict) -> str:
    """Add to the local buffer and, if connected, to the Google Sheet.  Returns a status line."""
    st.session_state["memlog"].append(rec)
    status = f"Logged run {rec['run_id']} locally ({len(st.session_state['memlog'])} in this session)."
    lg = st.session_state.get("gsheet")
    if lg is not None:
        try:
            where = lg.append(rec)
            status += f" Also written to Google Sheet {where}."
        except RunLogError as exc:
            st.session_state["gsheet_error"] = str(exc)
            status += f" Google Sheet NOT updated: {exc}"
    return status


def autolog(source: str, inp: AppInputs, live: Live) -> None:
    if st.session_state.get("autolog"):
        rec = make_record(source, inp, live, st.session_state.get("batch"), st.session_state.get("design"), "", {})
        st.session_state["last_log_status"] = log_record(rec)
        st.session_state.setdefault("logged", set()).add((rec["run_id"], source))


# --------------------------------------------------------------------------
# Operating sheet (download)
# --------------------------------------------------------------------------

def operating_sheet(inp: AppInputs, live: Live, batch: dict | None) -> str:
    L = [f"# Operating sheet", "", f"Cut {inp.T_lo_c:.0f}\u2013{inp.T_hi_c:.0f} \u00b0C (atmospheric-equivalent), "
         f"charge {inp.m_feed_kg:.2f} kg", "", "## Set-up", "",
         f"- P_top {inp.p_top_mbar:.0f} mbar, R = {live.R_use:.2f} ({'1.3 x minimum' if inp.R_auto else 'manual'}), "
         f"heater {inp.q_heater_w:.0f} W (efficiency {inp.efficiency:.2f})",
         f"- Column {inp.D_mm:.0f} mm x {inp.H_pack_m:.2f} m packed, "
         f"{'glass beads' if inp.kind == 'glass_beads' else 'steel wool'} {inp.size_mm:g} mm", "",
         "## What to watch", "",
         f"- Collect the cut while the head thermometer (TT-102) reads {live.cut.T_lo_c:.0f}\u2013{live.cut.T_hi_c:.0f} \u00b0C",
         f"- Expected: {fnum(live.hyd.percent_flood, 0)} % of flooding, bed pressure drop {fnum(live.hyd.dp_total_mbar, 2)} mbar "
         f"(pot about {fnum(live.hyd.p_pot_mbar, 1)} mbar), liquid held on the packing {live.hold.mass_kg * 1000:.0f} g",
         f"- Cut available in the charge: {live.avail_g:.0f} g", "", "## Column adequacy", "",
         f"- Stages needed {fnum(live.req.N_total)} (R_min {live.R_min:.2f}); this column gives {live.N_avail:.1f} "
         f"(HETP {live.hetp.hetp_m * 1000:.0f} mm); reflux needed with it: {fnum(live.R_needed, 2)}"]
    if batch:
        r = batch["res"]
        L += ["", "## Simulated run", "",
              f"- {r.t_batch_min:.0f} min in total, pot peaks at {r.T_pot_max_c:.0f} \u00b0C, stopped: {r.stop_reason}",
              f"- Target cut: {r.target.mass_kg * 1000:.0f} g, purity {r.target.purity:.2f}, recovery {r.target.recovery:.2f}"]
    L += ["", "_Model estimates: calibrate HETP, hold-up and flooding against your own runs before relying on them._"]
    return "\n".join(L)


# --------------------------------------------------------------------------
# Cached figures (every tab re-runs on each interaction, so plots are cached)
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner=False, max_entries=40)
def cached_fig(name: str, key: str, _make):
    """Build a figure once per (name, key); ``key`` is an inputs fingerprint or run id."""
    return _make()


# --------------------------------------------------------------------------
# Schematic embedding
# --------------------------------------------------------------------------

def show_schematic(state: SchematicState, key: str = "") -> None:
    svg = render_schematic(state)
    m = re.search(r'viewBox="0 0 1100 ([\d.]+)"', svg)
    height = int(float(m.group(1)) * 0.86) + 12 if m else 700
    if hasattr(st, "iframe"):                                  # newer Streamlit
        st.iframe(svg, height=height)
    else:                                                       # older Streamlit
        components.html(svg, height=height, scrolling=True)


def live_state(inp: AppInputs, live: Live) -> SchematicState:
    ibp = live.feed.curve.ibp_c
    labels = (f"<{inp.T_lo_c:.0f} \u00b0C", f"{inp.T_lo_c:.0f}\u2013{inp.T_hi_c:.0f} \u00b0C") if inp.T_lo_c > ibp else \
        (f"{inp.T_lo_c:.0f}\u2013{inp.T_hi_c:.0f} \u00b0C",)
    return SchematicState(
        D_mm=inp.D_mm, H_pack_m=inp.H_pack_m, H_col_m=live.geom.H_col_m, packing_kind=inp.kind,
        packing_size_mm=inp.size_mm, R=live.R_use, q_heater_w=inp.q_heater_w, p_top_mbar=inp.p_top_mbar,
        percent_flood=live.hyd.percent_flood, flooded=live.hyd.flooded, saturation=live.hold.saturation,
        cut_labels=labels, cut_masses_g=tuple(0.0 for _ in labels))


# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------

def tab_overview(inp: AppInputs, live: Live) -> None:
    c = st.columns(4)
    c[0].metric("Head window to collect", f"{live.cut.T_lo_c:.0f}\u2013{live.cut.T_hi_c:.0f} \u00b0C")
    c[1].metric("Cut available", f"{live.avail_g:.0f} g", f"{100 * live.avail_g / (inp.m_feed_kg * 1000):.0f} % of feed", delta_color="off")
    c[2].metric("Stages needed", fnum(live.req.N_total), f"R_min {live.R_min:.2f}, R = {live.R_use:.2f}", delta_color="off")
    c[3].metric("Stages this column gives", fnum(live.N_avail), f"HETP {live.hetp.hetp_m * 1000:.0f} mm", delta_color="off")
    c = st.columns(4)
    c[0].metric("Reflux needed with this column", fnum(live.R_needed, 2))
    c[1].metric("% of flooding", fnum(live.hyd.percent_flood, 0))
    c[2].metric("Bed pressure drop", fnum(live.hyd.dp_total_mbar, 2, "mbar"), f"pot {fnum(live.hyd.p_pot_mbar, 1)} mbar", delta_color="off")
    c[3].metric("Liquid held on packing", f"{live.hold.mass_kg * 1000:.0f} g", f"{live.hold.saturation:.0%} of voids", delta_color="off")

    if live.hyd.flooded:
        st.error("The column floods at this heater power: lower the power, use larger packing or a wider column.")
    elif live.N_avail >= live.req.N_total:
        st.success(f"This column provides enough stages at R = {live.R_use:.2f}.")
    elif math.isfinite(live.R_needed):
        st.warning(f"Short of stages at R = {live.R_use:.2f}: it would need R \u2265 {live.R_needed:.2f} "
                   f"(longer run), or about {fnum(live.H_required, 2)} m of packing.")
    else:
        st.error(f"This column cannot make the cut at any reflux ratio; it needs about {fnum(live.H_required, 2)} m of packing "
                 "at the chosen R. Use the Design tab.")
    if live.notices:
        with st.expander(f"Model notices ({len(live.notices)})"):
            for n in live.notices:
                st.write("\u2022 " + n)
    show_schematic(live_state(inp, live))


def tab_rating(inp: AppInputs, live: Live, fp: str) -> None:
    st.write("Simulate the whole batch with the column as set in the sidebar.")
    if st.button("Simulate batch", type="primary", key="btn_sim"):
        edges = (inp.T_lo_c, inp.T_hi_c) if inp.T_lo_c > live.feed.curve.ibp_c + 1e-9 else (inp.T_hi_c,)
        hold_m, hyd_m, _ = models(inp)
        spec = BatchSpec(m_feed_kg=inp.m_feed_kg, p_top_mbar=inp.p_top_mbar, R=live.R_use, q_heater_w=inp.q_heater_w,
                         cut_edges_c=edges, efficiency=inp.efficiency, T_condenser_c=inp.T_condenser_c)
        thermal = ThermalModel(Ea_kj_mol=inp.Ea_kj_mol, T_limit_c=inp.T_limit_c)
        try:
            with st.spinner("Simulating..."):
                res, msgs = run_captured(simulate_batch, live.feed, live.geom, spec, hetp_m=live.hetp.hetp_m,
                                         thermal=thermal, hold_model=hold_m, hyd_model=hyd_m)
        except ValueError as exc:
            st.error(str(exc))
            return
        st.session_state["batch"] = dict(res=res, spec=spec, geom=live.geom, msgs=msgs, fp=fp, run_id=uuid.uuid4().hex[:10])
        autolog("rating", inp, live)

    b = st.session_state.get("batch")
    if not b:
        return
    if b["fp"] != fp:
        st.warning("The inputs have changed since this simulation: press **Simulate batch** to update it.")
    res, spec = b["res"], b["spec"]
    if len(res.t_min) == 0:
        st.error(f"The run stopped before collecting anything (stopped: {res.stop_reason}). "
                 "Lower the heater power, use larger packing or a wider column.")
        return
    c = st.columns(4)
    c[0].metric("Target cut", f"{res.target.mass_kg * 1000:.0f} g")
    c[1].metric("Purity", fnum(res.target.purity, 2))
    c[2].metric("Recovery", fnum(res.target.recovery, 2))
    c[3].metric("Batch time", f"{res.t_batch_min:.0f} min", f"stopped: {res.stop_reason}", delta_color="off")
    c = st.columns(4)
    c[0].metric("Pot peak", f"{res.T_pot_max_c:.0f} \u00b0C")
    c[1].metric("Residue (pot + packing)", f"{res.residue_kg * 1000:.0f} g")
    c[2].metric("Thermal exposure", f"{res.tau_equiv_min:.3f} min-eq.")
    c[3].metric("Head colder than coolant", f"{res.light_ends_min:.0f} min")
    if b["msgs"]:
        with st.expander(f"Model notices ({len(b['msgs'])})"):
            for m in b["msgs"]:
                st.write("\u2022 " + m)
    st.dataframe(pd.DataFrame([dict(cut=f"{c_.T_lo_atm_c:.0f}\u2013{c_.T_hi_atm_c:.0f} \u00b0C", head_window=f"{c_.T_head_lo_c:.0f}\u2013{c_.T_head_hi_c:.0f} \u00b0C",
                                    mass_g=round(c_.mass_kg * 1000, 1), purity=round(c_.purity, 3), recovery=round(c_.recovery, 3),
                                    start_min=round(c_.t_start_min, 1), end_min=round(c_.t_end_min, 1)) for c_ in res.cuts]),
                 hide_index=True)
    step = st.slider("Time through the run", 0, len(res.t_min) - 1, len(res.t_min) // 2,
                     format="step %d", key="time_step")
    show_schematic(state_from_batch(res, b["geom"], spec, step, percent_flood=live.hyd.percent_flood,
                                    saturation=live.hold.saturation))
    st.pyplot(cached_fig("batch_history", b["run_id"], lambda: plots.batch_history_plot(res, spec)))
    st.pyplot(cached_fig("cut_summary", b["run_id"], lambda: plots.cut_summary_plot(res)))
    st.download_button("Download operating sheet (Markdown)", operating_sheet(inp, live, b), "operating_sheet.md", key="dl_sheet")


def tab_design(inp: AppInputs, live: Live, fp: str) -> None:
    st.write("Solve diameter and packed height for the target cut, using the heater power, pressure and packing in the sidebar.")
    c = st.columns(3)
    flood = c[0].slider("Target % of flooding", 40, 90, 70, key="d_flood")
    dpf = c[1].slider("Max pressure drop (fraction of P_top)", 0.02, 0.30, 0.10, 0.01, key="d_dpf")
    std_txt = c[2].text_input("Standard diameters (mm)", "20, 25, 30, 40, 50, 60, 80, 100, 150", key="d_std")
    try:
        std = tuple(parse_numbers(std_txt)) or None
        targets = DesignTargets(flood_pct=float(flood), max_dp_fraction=float(dpf), standard_diameters_mm=std)
    except ValueError as exc:
        st.error(f"Design targets: {exc}")
        return
    hold_m, hyd_m, het_m = models(inp)
    common = dict(feed=live.feed, T_lo_c=inp.T_lo_c, T_hi_c=inp.T_hi_c, q_heater_w=inp.q_heater_w,
                  R=None if inp.R_auto else inp.R_manual, sep=SeparationSpec(inp.r_key, inp.gap_c), targets=targets,
                  efficiency=inp.efficiency, hold_model=hold_m, hyd_model=hyd_m, hetp_cfg=het_m)
    spec_pk = PackingSpec(inp.kind, inp.size_mm, void_fraction=inp.void_fraction)

    if st.button("Design column", type="primary", key="btn_design"):
        try:
            with st.spinner("Designing (about 5 s)..."):
                d, msgs = run_captured(design_column, packing=spec_pk, p_top_mbar=inp.p_top_mbar,
                                       m_feed_kg=inp.m_feed_kg, verify=True, **common)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.session_state["design"] = dict(d=d, msgs=msgs, fp=fp, run_id=uuid.uuid4().hex[:10])
            autolog("design", inp, live)

    dz = st.session_state.get("design")
    if dz:
        d = dz["d"]
        if dz["fp"] != fp:
            st.warning("The inputs have changed since this design: press **Design column** to update it.")
        c = st.columns(4)
        c[0].metric("Diameter", f"{d.D_mm:.0f} mm", f"solved {d.D_solved_mm:.1f} mm, set by {d.limiting}", delta_color="off")
        c[1].metric("Packed height", f"{d.H_pack_m:.2f} m", f"column {d.H_col_m:.2f} m", delta_color="off")
        c[2].metric("Stages", fnum(d.N_total), f"R = {d.R:.2f} (min {d.R_min:.2f})", delta_color="off")
        c[3].metric("HETP", f"{d.hetp_m * 1000:.0f} mm")
        c = st.columns(4)
        c[0].metric("% of flooding", f"{d.percent_flood:.0f}")
        c[1].metric("Pressure drop", f"{d.dp_mbar:.2f} mbar", f"pot {d.p_pot_mbar:.1f} mbar", delta_color="off")
        c[2].metric("Liquid held on packing", f"{d.holdup_kg * 1000:.0f} g")
        c[3].metric("Packing mass", f"{d.m_pack_kg * 1000:.0f} g")
        for n in d.notes:
            st.info(n)
        if d.verification is not None:
            t = d.verification.target
            st.success(f"Simulated on the designed column: target {t.mass_kg * 1000:.0f} g, purity {t.purity:.2f}, "
                       f"recovery {t.recovery:.2f}, {d.verification.t_batch_min:.0f} min.")
        elif d.verification_error:
            st.warning("Not verified by simulation: " + d.verification_error)
        st.button("Apply this design to the column inputs", on_click=apply_design_to_inputs, key="btn_apply")
        show_schematic(state_from_design(d))

    st.divider()
    st.subheader("Compare packing sizes")
    sizes_txt = st.text_input("Sizes (mm)", "3, 4, 5, 6, 8", key="cmp_sizes")
    if st.button("Compare sizes", key="btn_cmp"):
        try:
            sizes = parse_numbers(sizes_txt)
            with st.spinner("Designing each size..."):
                out = compare_packings(sizes, kind=inp.kind, p_top_mbar=inp.p_top_mbar, verify=False, **common)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.session_state["cmp"] = out
    if st.session_state.get("cmp"):
        out = st.session_state["cmp"]
        st.dataframe(pd.DataFrame([dict(size_mm=s, diameter_mm=getattr(r, "D_mm", None), packed_height_m=getattr(r, "H_pack_m", None),
                                        stages=getattr(r, "N_total", None), hetp_mm=(r.hetp_m * 1000 if r else None),
                                        dp_mbar=getattr(r, "dp_mbar", None), problem=err) for s, r, err in out]),
                     hide_index=True)
        st.pyplot(plots.design_sweep_plot([s for s, _, _ in out], [r for _, r, _ in out], "Packing size (mm)"))

    st.subheader("Effect of operating pressure")
    p_txt = st.text_input("Pressures (mbar)", "20, 50, 100, 300, 1013", key="p_sweep")
    if st.button("Sweep pressure", key="btn_psweep"):
        try:
            ps = parse_numbers(p_txt)
        except ValueError as exc:
            st.error(str(exc))
            ps = []
        rows = []
        with st.spinner("Designing at each pressure..."):
            for p in ps:
                try:
                    rows.append((p, design_column(packing=spec_pk, p_top_mbar=p, verify=False, **common), None))
                except ValueError as exc:
                    rows.append((p, None, str(exc)))
        st.session_state["psweep"] = rows
    if st.session_state.get("psweep"):
        rows = st.session_state["psweep"]
        st.dataframe(pd.DataFrame([dict(P_top_mbar=p, diameter_mm=getattr(r, "D_mm", None), packed_height_m=getattr(r, "H_pack_m", None),
                                        stages=getattr(r, "N_total", None), dp_mbar=getattr(r, "dp_mbar", None), problem=err)
                                   for p, r, err in rows]), hide_index=True)
        st.pyplot(plots.design_sweep_plot([p for p, _, _ in rows], [r for _, r, _ in rows], "Top pressure (mbar)", log_x=True))


def tab_plots(inp: AppInputs, live: Live, fp: str) -> None:
    feed = live.feed
    ibp = feed.curve.ibp_c
    cuts = [(ibp, inp.T_lo_c), (inp.T_lo_c, inp.T_hi_c)] if inp.T_lo_c > ibp else [(inp.T_lo_c, inp.T_hi_c)]
    hold_m, hyd_m, _ = models(inp)
    figs = [
        ("Feed curve", lambda: plots.feed_curve_plot(feed, cuts=cuts)),
        ("Boiling points", lambda: plots.boiling_point_plot(feed, [1013.25, 100.0, 20.0], inp.p_top_mbar,
                                                            (inp.T_lo_c, inp.T_hi_c), inp.T_condenser_c)),
        ("Stages vs reflux", lambda: plots.stages_vs_reflux_plot(
            feed, inp.T_lo_c, inp.T_hi_c, inp.p_top_mbar, SeparationSpec(inp.r_key, inp.gap_c),
            R_current=live.R_use, N_available=live.N_avail)),
        ("Hydraulics", lambda: plots.hydraulics_vs_power_plot(live.geom, live.fluid, live.R_use, inp.efficiency,
                                                             q_current=inp.q_heater_w, hold_model=hold_m, hyd_model=hyd_m)),
    ]
    cols = st.columns(2)
    for i, (name, make) in enumerate(figs):
        with cols[i % 2]:
            try:
                st.pyplot(cached_fig(name, fp, make))
            except Exception as exc:                        # a failed plot must not take the app down
                st.error(f"{name}: {exc}")


# --------------------------------------------------------------------------
# Run log tab (Google Sheets bot)
# --------------------------------------------------------------------------

def _service_account_info():
    """Credentials from Streamlit secrets or the uploaded key (None if neither)."""
    try:
        if "gcp_service_account" in st.secrets:
            return dict(st.secrets["gcp_service_account"]), "Streamlit secrets"
    except Exception:
        pass
    up = st.session_state.get("uploaded_key")
    return (up, "uploaded key") if up else (None, "")


def tab_log(inp: AppInputs, live: Live) -> None:
    mem: MemoryLog = st.session_state["memlog"]
    st.subheader("Google Sheet bot")
    st.caption("A service account (the \u201cbot\u201d) appends one row per run to a sheet you share with it. "
               "Runs are always kept locally too, so nothing is lost if Google is unreachable.")
    with st.expander("How to set it up"):
        st.markdown(
            "1. In Google Cloud, create a **service account**, enable the **Google Sheets API** and **Drive API**, and download its JSON key.\n"
            "2. **Share your Google Sheet** with the service account's e-mail (shown below once the key is loaded) as *Editor*.\n"
            "3. Give the key to the app: put it in `.streamlit/secrets.toml` under `[gcp_service_account]` "
            "(see `.streamlit/secrets.toml.example`), or upload the JSON here (kept in memory for this session only).\n"
            "4. Paste the sheet's URL, press **Connect / test**. The first connection writes the header row.\n\n"
            "Never commit the key to GitHub.")
    info, source = _service_account_info()
    if info is None:
        up = st.file_uploader("Service-account key (JSON)", type="json", key="key_upload")
        if up is not None:
            try:
                import json
                st.session_state["uploaded_key"] = json.load(up)
                st.rerun()
            except ValueError:
                st.error("That file is not valid JSON.")
    else:
        st.success(f"Key loaded from {source}. Bot e-mail: **{info.get('client_email', '?')}**")
        if source == "uploaded key" and st.button("Forget the uploaded key", key="btn_forget"):
            st.session_state.pop("uploaded_key", None)
            st.session_state.pop("gsheet", None)
            st.rerun()
    c = st.columns([3, 1])
    sheet = c[0].text_input("Google Sheet URL or ID", key="sheet_ref")
    ws_name = c[1].text_input("Worksheet", "runs", key="ws_name")
    if st.button("Connect / test", key="btn_connect", disabled=info is None):
        try:
            lg = GSheetLogger.from_service_account_info(info, sheet, ws_name)
            lg.connect()
        except RunLogError as exc:
            st.session_state.pop("gsheet", None)
            st.error(str(exc))
        else:
            st.session_state["gsheet"] = lg
            st.session_state.pop("gsheet_error", None)
    if st.session_state.get("gsheet") is not None:
        lg = st.session_state["gsheet"]
        st.success(f"Connected: worksheet \u201c{lg.worksheet_name}\u201d. New runs are written there.")
    else:
        st.info("Not connected to a Google Sheet: runs are kept locally and can be downloaded as CSV.")

    st.checkbox("Log every simulation and design run automatically", key="autolog")
    if st.session_state.get("last_log_status"):
        st.caption(st.session_state["last_log_status"])

    st.subheader("Log a run")
    options = {"Current estimate (no simulation)": "estimate"}
    if st.session_state.get("batch"):
        options["Latest simulation (rating)"] = "rating"
    if st.session_state.get("design"):
        options["Latest design"] = "design"
    choice = st.selectbox("What to log", list(options), index=len(options) - 1, key="log_choice")
    source = options[choice]
    label = st.text_input("Label (optional)", key="log_label")
    measured: dict = {}
    if st.checkbox("I have bench measurements for this run", key="has_meas"):
        m = st.columns(4)
        fields = [("meas_mass_g", "Cut mass (g)"), ("meas_purity", "Purity"), ("meas_recovery", "Recovery"),
                  ("meas_dp_mbar", "Pressure drop (mbar)"), ("meas_head_T_c", "Head T (\u00b0C)"), ("meas_batch_min", "Batch time (min)")]
        for i, (k, lab) in enumerate(fields):
            txt = m[i % 4].text_input(lab, key=f"m_{k}")
            if txt.strip():
                try:
                    measured[k] = float(txt)
                except ValueError:
                    st.error(f"{lab}: not a number.")
        if st.checkbox("The column flooded", key="m_flooded"):
            measured["meas_flooded"] = True
        note = st.text_area("Notes", key="m_notes")
        if note.strip():
            measured["notes"] = note.strip()
    run_id = (st.session_state.get("batch") or {}).get("run_id") if source == "rating" else \
             (st.session_state.get("design") or {}).get("run_id") if source == "design" else None
    already = run_id is not None and (run_id, source) in st.session_state.get("logged", set())
    again = st.checkbox("Log again even though this run is already logged", key="log_again", disabled=not already)
    if st.button("Log this run", type="primary", key="btn_log", disabled=already and not again):
        rec = make_record(source, inp, live, st.session_state.get("batch"), st.session_state.get("design"), label, measured)
        st.session_state["last_log_status"] = log_record(rec)
        st.session_state.setdefault("logged", set()).add((rec["run_id"], source))
        st.rerun()

    st.subheader(f"This session ({len(mem)} runs)")
    if len(mem):
        df = pd.read_csv(io.StringIO(mem.to_csv()))
        st.dataframe(df.iloc[::-1].head(20), hide_index=True)
        st.download_button("Download all as CSV", mem.to_csv(), "fdc_runs.csv", "text/csv", key="dl_csv")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    st.session_state.setdefault("memlog", MemoryLog())
    st.title("Packed fractional distillation column for WPPO")
    st.caption("Model estimates for a lab-scale batch column. The placeholder feed curve and several constants are "
               "assumptions: replace them with your data and calibrate against your own runs.")
    inp = read_sidebar()
    fp = fingerprint(inp)
    try:
        live = compute_live(fp, inp)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    tabs = st.tabs(["Overview", "Rating (simulate)", "Design", "Plots", "Run log"])
    with tabs[0]:
        tab_overview(inp, live)
    with tabs[1]:
        tab_rating(inp, live, fp)
    with tabs[2]:
        tab_design(inp, live, fp)
    with tabs[3]:
        tab_plots(inp, live, fp)
    with tabs[4]:
        tab_log(inp, live)


main()
