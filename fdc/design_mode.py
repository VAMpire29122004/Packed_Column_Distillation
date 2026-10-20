"""
fdc.design_mode
===============

Design mode: given the separation you want, solve for the column.  This is the
inverse of the rating calculations in the other modules, which take a column
and predict the cut.

Corr-map position
-----------------
Drivers : V03 (cut and separation spec), V04 (P_top), V05 (R), V06 (heater
          power), V08 (packing type and size), V01 (charge, for verification)
Solves  : V07 (column diameter) and V09 (packed height), the links marked 'D'
          (design mode only) in the corr map, then V21 (column height).

How the design is solved
------------------------
1. Stages.  ``stages.cut_stage_requirement`` gives the theoretical stages
   needed for the cut at the chosen reflux ratio (default 1.3 x R_min).
2. Diameter.  The vapour rate is fixed by the heater power, so the diameter sets
   the vapour velocity and hence the margin to flooding.  D is solved so the
   column runs at ``flood_pct`` of flooding (default 70 %); if the resulting
   pressure drop exceeds ``max_dp_fraction`` of P_top (a larger share than that
   moves the pot temperature too much), D is increased until it does not.
   Beads also need D / d_p >= 8, which can set a minimum diameter.
3. Height.  HETP is evaluated at that diameter and operating point, and
   H_pack = (N_total - 1) x HETP.  H_col adds the unpacked allowances.
4. Standard glassware.  The continuous diameter is rounded up to the next size
   in ``standard_diameters_mm``; rounding up keeps every constraint satisfied.
5. Verification.  If a charge is given, the designed column is run through
   ``batch_pot.simulate_batch`` and the achieved purity and recovery are returned.

The hydraulics are sized at the *upper* edge of the cut (hottest head, largest
vapour volume), which is the most demanding point of that cut's collection.

Conventions
-----------
Diameters in mm, heights in m, pressures in mbar, temperatures in degC.  Cut limits
are atmospheric-equivalent, like everywhere else.

Known limitations
-----------------
* Sized for one target cut at one reflux ratio, not for the whole fraction
  train; check forerun and later cuts with the batch simulation.
* Inherits every uncertainty of the underlying models (HETP and flooding most
  of all): calibrate those first, then trust the design.
* ASSUMPTIONS: the 70 % flooding target, the 10 % pressure-drop limit, the list
  of standard diameters and the 1.5 m height warning are defaults you should
  set to suit your glassware and bench.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

from scipy.optimize import brentq

from .batch_pot import BatchResult, BatchSpec, SimSettings, ThermalModel, simulate_batch
from .hetp import HETPModel, HETPResult, hetp as hetp_model, packed_height_required
from .holdup import HoldupModel, internal_flows, liquid_holdup
from .hydraulics import HydraulicsModel, column_hydraulics, flooding_vapour_rate
from .packing import (H_BOTTOM_DEFAULT, H_TOP_DEFAULT, WALL_RATIO_MIN, ColumnGeometry, PackingSpec,
                      column_geometry)
from .properties.feed import FeedCharacterisation
from .properties.fluids import Components, mixture_properties
from .properties.vle import analyse_cut
from .stages import R_FACTOR_DEFAULT, SeparationSpec, cut_stage_requirement

# ASSUMPTION: common inner diameters of laboratory glass columns [mm].
STANDARD_DIAMETERS_MM = (20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 150.0)


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DesignTargets:
    """Constraints and preferences of the design (ASSUMPTIONS, see module docstring).

    flood_pct              : target % of flooding at the design heater power.
    max_dp_fraction        : allowed bed pressure drop as a fraction of P_top.
    D_min_mm, D_max_mm     : diameter search range.
    standard_diameters_mm  : round up to these sizes; ``None`` keeps the continuous value.
    H_warn_m               : warn if the packed height exceeds this.
    """

    flood_pct: float = 70.0
    max_dp_fraction: float = 0.10
    D_min_mm: float = 20.0
    D_max_mm: float = 200.0
    standard_diameters_mm: tuple[float, ...] | None = STANDARD_DIAMETERS_MM
    H_warn_m: float = 1.5

    def __post_init__(self):
        if not 10.0 <= self.flood_pct <= 95.0:
            raise ValueError("flood_pct should lie between 10 and 95.")
        if not 0.0 < self.max_dp_fraction < 1.0:
            raise ValueError("max_dp_fraction must lie between 0 and 1.")
        if not 0.0 < self.D_min_mm < self.D_max_mm:
            raise ValueError("Need 0 < D_min_mm < D_max_mm.")


@dataclass(frozen=True)
class DesignResult:
    """A designed column and how it performs."""

    # what was asked
    p_top_mbar: float
    T_lo_c: float
    T_hi_c: float
    q_heater_w: float
    # the design
    D_mm: float                    # final diameter (standard size if rounding is on)
    D_solved_mm: float             # continuous diameter that meets the constraints
    limiting: str                  # "flooding", "pressure drop" or "minimum diameter"
    H_pack_m: float
    H_col_m: float
    N_total: float                 # theoretical stages including the pot
    N_packing: float
    R: float
    R_min: float
    alpha: float                   # governing key relative volatility
    hetp_m: float
    # how it performs at the design point
    percent_flood: float
    F_factor: float
    dp_mbar: float
    p_pot_mbar: float
    holdup_kg: float               # liquid held on the packing
    m_pack_kg: float
    wall_ratio: float
    geom: ColumnGeometry
    hetp_detail: HETPResult
    notes: tuple[str, ...]
    verification: BatchResult | None = None
    verification_error: str | None = None


# --------------------------------------------------------------------------
# Design
# --------------------------------------------------------------------------

def design_column(
    feed: FeedCharacterisation,
    packing: PackingSpec,
    p_top_mbar: float,
    T_lo_c: float,
    T_hi_c: float,
    q_heater_w: float,
    m_feed_kg: float | None = None,
    R: float | None = None,
    sep: SeparationSpec = SeparationSpec(),
    targets: DesignTargets = DesignTargets(),
    efficiency: float = 0.70,
    h_top_m: float = H_TOP_DEFAULT,
    h_bottom_m: float = H_BOTTOM_DEFAULT,
    hold_model: HoldupModel = HoldupModel(),
    hyd_model: HydraulicsModel = HydraulicsModel(),
    hetp_cfg: HETPModel = HETPModel(),
    verify: bool = True,
) -> DesignResult:
    """Solve column diameter and packed height for a target cut.

    Parameters
    ----------
    feed, packing        : characterised feed and the packing to use.
    p_top_mbar           : top pressure.
    T_lo_c, T_hi_c       : cut limits, atmospheric-equivalent degC.
    q_heater_w           : heater power, which fixes the vapour rate.
    m_feed_kg            : charge; needed only to verify the design by simulation.
    R                    : reflux ratio; default 1.3 x the minimum.
    sep                  : cut-sharpness spec (key recovery, grey-zone width).
    targets              : flooding margin, pressure-drop limit, diameter range.
    h_top_m, h_bottom_m  : unpacked column height above and below the bed [m]; they set the
                           column body height (H_pack + both) but not the separation.
    verify               : run the batch simulation on the designed column
                           (needs ``m_feed_kg``).
    """
    comps = Components.from_feed(feed)
    notes: list[str] = []

    # ---- 1. stages ------------------------------------------------------------
    probe = cut_stage_requirement(feed, T_lo_c, T_hi_c, p_top_mbar, R=1.0e6, spec=sep)
    if not probe.boundaries:
        raise ValueError("This cut takes the whole feed: there is no separation to design.")
    R_min = probe.R_min
    R_use = R_FACTOR_DEFAULT * R_min if R is None else R
    if R_use <= R_min:
        raise ValueError(f"R = {R_use:.2f} is at or below the minimum reflux ratio {R_min:.2f} "
                         "for this cut and separation spec.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        req = cut_stage_requirement(feed, T_lo_c, T_hi_c, p_top_mbar, R_use, sep)
    gov = max(req.boundaries, key=lambda b: b.N_total)

    # ---- 2. operating point for sizing: upper edge of the cut ----------------------
    cut = analyse_cut(feed, T_lo_c, T_hi_c, p_top_mbar, gap_c=sep.gap_c)
    _, x_cut = feed.cut_composition(T_lo_c, T_hi_c)
    fluid = mixture_properties(comps, x_cut, cut.T_hi_c, p_top_mbar)
    flows = internal_flows(q_heater_w, fluid.latent_kj_kg, R_use, efficiency)
    phi = R_use / (R_use + 1.0)

    d_size = packing.size_mm
    D_lo = targets.D_min_mm
    if packing.kind == "glass_beads":
        D_lo = max(D_lo, WALL_RATIO_MIN * d_size)         # wall-effect minimum

    def geometry(D_mm: float, H_m: float) -> ColumnGeometry:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return column_geometry(packing, D_mm, H_m, h_top_m, h_bottom_m)

    def percent_flood(D_mm: float) -> float:
        g = geometry(D_mm, 0.30)
        return 100.0 * flows.m_vapour / flooding_vapour_rate(phi, g, fluid, hold_model, hyd_model)

    def evaluate(D_mm: float):
        """Full design at a diameter: HETP, packed height, hydraulics."""
        g0 = geometry(D_mm, 0.30)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            h = hetp_model(g0, fluid, flows, gov.alpha, hetp_cfg)
        H = max(packed_height_required(req.N_total, h.hetp_m), 0.05)
        g = geometry(D_mm, H)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hyd = column_hydraulics(g, fluid, flows, hold_model, hyd_model)
        return g, h, H, hyd

    # ---- 3. diameter from the flooding margin --------------------------------------
    target = targets.flood_pct
    if percent_flood(targets.D_max_mm) > target:
        raise ValueError(f"Even a {targets.D_max_mm:.0f} mm column would run above {target:.0f} % of "
                         "flooding at this heater power; lower the power or use larger packing.")
    if percent_flood(D_lo) <= target:
        D_flood, limiting = D_lo, "minimum diameter"
    else:
        # flooding vapour rate scales roughly with area, so D ~ sqrt(pct): use that as a start
        D_ref = max(D_lo, 40.0)
        guess = D_ref * math.sqrt(percent_flood(D_ref) / target)
        lo_b = max(D_lo, 0.7 * guess)
        hi_b = min(targets.D_max_mm, 1.4 * guess)
        if not (percent_flood(lo_b) > target >= percent_flood(hi_b)):
            lo_b, hi_b = D_lo, targets.D_max_mm
        D_flood = brentq(lambda D: percent_flood(D) - target, lo_b, hi_b, xtol=1e-3)
        limiting = "flooding"

    # ---- 4. diameter from the pressure-drop limit -----------------------------------
    def dp_excess(D_mm: float) -> float:
        _, _, _, hyd = evaluate(D_mm)
        return hyd.dp_total_mbar / p_top_mbar - targets.max_dp_fraction

    D_cont = D_flood
    if dp_excess(D_flood) > 0.0:
        if dp_excess(targets.D_max_mm) > 0.0:
            raise ValueError(f"The pressure-drop limit ({targets.max_dp_fraction:.0%} of P_top) cannot "
                             f"be met even at {targets.D_max_mm:.0f} mm; the bed is too tall or the "
                             "packing too small for this pressure.")
        D_cont = brentq(dp_excess, D_flood, targets.D_max_mm, xtol=1e-3)
        limiting = "pressure drop"

    # ---- 5. standard glassware ------------------------------------------------------
    D_final = D_cont
    if targets.standard_diameters_mm:
        bigger = [d for d in sorted(targets.standard_diameters_mm) if d >= D_cont - 1e-9]
        if bigger and bigger[0] <= targets.D_max_mm:
            D_final = bigger[0]
        else:
            notes.append("No standard diameter is large enough; the continuous diameter is reported.")

    geom, h_res, H_pack, hyd = evaluate(D_final)
    pct_final = 100.0 * flows.m_vapour / hyd.m_flood_kg_s

    # ---- notes ---------------------------------------------------------------------
    if limiting == "minimum diameter":
        notes.append("Wall-effect minimum (D/d_p >= 8) sets the diameter, not flooding.")
    if H_pack > targets.H_warn_m:
        notes.append(f"Packed height {H_pack:.2f} m is above {targets.H_warn_m:.1f} m: consider a higher "
                     "pressure, larger packing, a looser separation spec or a different reflux ratio.")
    if fluid.T_c < 15.0:
        notes.append("The head is colder than typical condenser coolant at this pressure: light ends will "
                     "not condense. Check the forerun with the batch simulation.")
    if packing.kind == "steel_wool":
        notes.append("Steel wool: hold-up, pressure drop and HETP are extrapolations; calibrate before trusting.")

    # ---- 6. verification by simulation ----------------------------------------------
    batch, batch_err = None, None
    if verify and m_feed_kg is not None:
        edges = (T_lo_c, T_hi_c) if T_lo_c > feed.curve.ibp_c + 1e-9 else (T_hi_c,)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                batch = simulate_batch(
                    feed, geom,
                    BatchSpec(m_feed_kg=m_feed_kg, p_top_mbar=p_top_mbar, R=R_use, q_heater_w=q_heater_w,
                              cut_edges_c=edges, efficiency=efficiency),
                    hetp_m=h_res.hetp_m, hold_model=hold_model, hyd_model=hyd_model)
        except ValueError as exc:
            batch_err = str(exc)

    return DesignResult(
        p_top_mbar=p_top_mbar, T_lo_c=T_lo_c, T_hi_c=T_hi_c, q_heater_w=q_heater_w,
        D_mm=D_final, D_solved_mm=D_cont, limiting=limiting, H_pack_m=H_pack, H_col_m=geom.H_col_m,
        N_total=req.N_total, N_packing=req.N_packing, R=R_use, R_min=R_min, alpha=gov.alpha,
        hetp_m=h_res.hetp_m, percent_flood=pct_final, F_factor=hyd.F_factor, dp_mbar=hyd.dp_total_mbar,
        p_pot_mbar=hyd.p_pot_mbar, holdup_kg=liquid_holdup(geom, fluid, flows, hold_model).mass_kg,
        m_pack_kg=geom.m_pack_kg, wall_ratio=geom.wall_ratio, geom=geom, hetp_detail=h_res,
        notes=tuple(notes), verification=batch, verification_error=batch_err)


def compare_packings(sizes_mm, kind: str = "glass_beads", **kwargs) -> list:
    """Design the column for several packing sizes.

    Returns a list of ``(size_mm, DesignResult | None, error_message | None)``;
    a size that cannot meet the constraints gives ``None`` and the reason.
    Keyword arguments are passed to ``design_column`` (without ``packing``).
    """
    out = []
    for s in sizes_mm:
        try:
            out.append((s, design_column(packing=PackingSpec(kind, s), **kwargs), None))
        except ValueError as exc:
            out.append((s, None, str(exc)))
    return out


# --------------------------------------------------------------------------
# Demo / smoke test:  python -m fdc.design_mode
# --------------------------------------------------------------------------

if __name__ == "__main__":
    from .properties.feed import DEFAULT_WPPO_TBP, TBPCurve, characterise_feed

    warnings.simplefilter("ignore")
    feed = characterise_feed(TBPCurve.from_points(DEFAULT_WPPO_TBP), n_cuts=30, sg_feed=0.80)
    common = dict(feed=feed, T_lo_c=150.0, T_hi_c=250.0, m_feed_kg=1.0)

    d = design_column(packing=PackingSpec("glass_beads", 5.0), p_top_mbar=100.0, q_heater_w=100.0, **common)
    print("Design for the 150-250 C cut, 100 mbar, 100 W, 5 mm glass beads, 1.0 kg charge\n")
    print(f"  stages {d.N_total:.1f} (packing {d.N_packing:.1f}) at R = {d.R:.2f} (R_min {d.R_min:.2f}), "
          f"key alpha {d.alpha:.2f}")
    print(f"  HETP {d.hetp_m * 1000:.0f} mm  ->  packed height {d.H_pack_m:.2f} m, column height {d.H_col_m:.2f} m")
    print(f"  diameter {d.D_mm:.0f} mm (solved {d.D_solved_mm:.1f} mm, limited by {d.limiting}), "
          f"D/d_p = {d.wall_ratio:.1f}")
    print(f"  {d.percent_flood:.0f} % of flooding, F = {d.F_factor:.2f} Pa^0.5, dP {d.dp_mbar:.2f} mbar "
          f"(pot at {d.p_pot_mbar:.1f} mbar), hold-up {d.holdup_kg * 1000:.0f} g, packing {d.m_pack_kg * 1000:.0f} g")
    if d.verification:
        t = d.verification.target
        print(f"  simulated run: target {t.mass_kg * 1000:.0f} g, purity {t.purity:.2f}, recovery "
              f"{t.recovery:.2f}, {d.verification.t_batch_min:.0f} min")
    for n in d.notes:
        print("  note:", n)

    print("\nPacking size comparison (same spec):")
    print(f"  {'bead/mm':>7} {'D/mm':>5} {'H_pack/m':>8} {'stages':>7} {'HETP/mm':>8} {'dP/mbar':>8}")
    for s, r, err in compare_packings((3, 4, 5, 6, 8), p_top_mbar=100.0, q_heater_w=100.0, verify=False, **common):
        if r:
            print(f"  {s:7.0f} {r.D_mm:5.0f} {r.H_pack_m:8.2f} {r.N_total:7.1f} {r.hetp_m * 1000:8.0f} {r.dp_mbar:8.2f}")
        else:
            print(f"  {s:7.0f}  --  {err[:60]}")

    print("\nEffect of operating pressure (5 mm beads, 100 W):")
    print(f"  {'P_top/mbar':>10} {'D/mm':>5} {'H_pack/m':>8} {'stages':>7} {'dP/mbar':>8}")
    for p in (20.0, 50.0, 100.0, 300.0, 1013.25):
        try:
            r = design_column(packing=PackingSpec("glass_beads", 5.0), p_top_mbar=p, q_heater_w=100.0, verify=False, **common)
            print(f"  {p:10.0f} {r.D_mm:5.0f} {r.H_pack_m:8.2f} {r.N_total:7.1f} {r.dp_mbar:8.2f}")
        except ValueError as exc:
            print(f"  {p:10.0f}  --  {str(exc)[:60]}")
