"""
fdc.stages
==========

Number of theoretical stages needed to make a boiling-range cut at a given
reflux ratio: Fenske minimum stages, Underwood-type minimum reflux and the
Gilliland (Molokanov) correlation, applied to each boundary of the cut.

Corr-map position
-----------------
Drivers : V03 (cut range and separation spec), V13 (relative volatility from
          ``properties.vle``), V05 (reflux ratio R)
Solves  : V19 (theoretical stages required, N_req).  The packed height then
          follows in ``hetp.py`` / ``design_mode.py`` as H_pack = N_pack x HETP.

How a continuous feed is turned into a two-key problem
------------------------------------------------------
WPPO has no sharp components, so "purity" needs a practical definition.  Each
cut boundary is treated as a split between two *groups*: material boiling
below the boundary (light group) and above it (heavy group).  The separation
spec has two numbers:

* ``gap_c``  : light and heavy keys sit this far either side of the boundary
               (atmospheric-equivalent degC); material inside the gap may go
               either way.  Larger gap = easier separation.
* ``r_key``  : recovery of each key into its own product (0.95 = 95 % of the
               light key goes overhead and 95 % of the heavy key stays in the
               pot).

The still pot counts as one theoretical stage, so the packed section has
N_total - 1 stages.  A cut with a lower boundary needs a forerun removed first;
the upper boundary is then split from the material that remains.  The harder
of the two boundaries governs the column.

Method
------
* Fenske:  N_min = ln[ r_LK/(1-r_LK) * r_HK/(1-r_HK) ] / ln(alpha).
* Minimum reflux for a saturated-liquid pot treated as a pseudo-binary of the
  two groups (Colburn / Underwood form),
      R_min = [ x_D/z - alpha (1 - x_D)/(1 - z) ] / (alpha - 1),
  z = mole fraction of the light group in the pot, x_D = its purity overhead
  implied by the key recoveries.
* Gilliland, in the Molokanov (1972) form:
      Y = 1 - exp[ (1 + 54.4 X)/(11 + 117.2 X) * (X - 1)/sqrt(X) ],
      X = (R - R_min)/(R + 1),  Y = (N - N_min)/(N + 1).

Conventions
-----------
Temperatures in degC (atmospheric-equivalent for cut limits), pressure in mbar,
compositions in mole fractions.

Main references
---------------
* Fenske (1932), Ind. Eng. Chem. 24, 482.   * Underwood (1948), Chem. Eng. Prog. 44, 603.
* Gilliland (1940), Ind. Eng. Chem. 32, 1220.
* Molokanov, Korablina, Mazurina & Nikiforov (1972), Int. Chem. Eng. 12, 209.

Known limitations
-----------------
* Constant relative volatility; the key alpha is taken at the boundary's
  boiling temperature at P_top, so the pot (hotter, higher pressure) is
  slightly harder than assumed.
* Batch operation: z falls as the light group is removed, so the true R_min
  rises during the cut.  The values here are for the start of collection of
  each boundary's split; ``batch_pot.py`` handles the time dependence.
* The two-group picture is a design estimate, not a rigorous
  multicomponent calculation.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .properties.feed import FeedCharacterisation
from .properties.vle import analyse_cut

# --------------------------------------------------------------------------
# Constants and defaults
# --------------------------------------------------------------------------

# Common practice is to run at 1.2-1.5 times the minimum reflux ratio.
R_FACTOR_DEFAULT = 1.3


@dataclass(frozen=True)
class SeparationSpec:
    """Practical definition of cut sharpness (ASSUMPTION: defaults are typical lab targets).

    r_key  : recovery of each key into its own product, between 0.5 and 1.
    gap_c  : half-width of the grey zone either side of a boundary [degC].
    """

    r_key: float = 0.95
    gap_c: float = 10.0

    def __post_init__(self):
        if not 0.5 < self.r_key < 1.0:
            raise ValueError("r_key must lie between 0.5 and 1 (exclusive).")
        if self.gap_c <= 0:
            raise ValueError("gap_c must be positive.")


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------

def minimum_stages(alpha: float, r_lk: float, r_hk: float | None = None) -> float:
    """Fenske minimum number of theoretical stages (total reflux).

    r_lk, r_hk are the recoveries of the light key overhead and of the heavy
    key in the pot; if ``r_hk`` is omitted it is taken equal to ``r_lk``.
    """
    if alpha <= 1.0:
        raise ValueError("Relative volatility must exceed 1.")
    r_hk = r_lk if r_hk is None else r_hk
    for r in (r_lk, r_hk):
        if not 0.0 < r < 1.0:
            raise ValueError("Recoveries must lie strictly between 0 and 1.")
    return math.log((r_lk / (1.0 - r_lk)) * (r_hk / (1.0 - r_hk))) / math.log(alpha)


def minimum_reflux(alpha: float, z: float, x_d: float) -> float:
    """Minimum reflux ratio, pseudo-binary with a saturated-liquid pot.

    ``z``: light-group mole fraction in the pot; ``x_d``: its mole fraction in
    the distillate.  Returned value is floored at zero.
    """
    if alpha <= 1.0:
        raise ValueError("Relative volatility must exceed 1.")
    if not (0.0 < z < 1.0 and 0.0 < x_d < 1.0):
        raise ValueError("z and x_d must lie strictly between 0 and 1.")
    return max(0.0, (x_d / z - alpha * (1.0 - x_d) / (1.0 - z)) / (alpha - 1.0))


def gilliland_y(X: float) -> float:
    """Molokanov form of the Gilliland correlation, Y = (N - N_min)/(N + 1)."""
    if X <= 0.0:
        return 1.0
    if X >= 1.0:
        return 0.0
    return 1.0 - math.exp((1.0 + 54.4 * X) / (11.0 + 117.2 * X) * (X - 1.0) / math.sqrt(X))


def stages_at_reflux(N_min: float, R_min: float, R: float) -> float:
    """Total theoretical stages (including the pot) at reflux ratio ``R``.

    Returns infinity if ``R <= R_min`` (the separation cannot be made).
    """
    if R < 0:
        raise ValueError("Reflux ratio cannot be negative.")
    if math.isinf(R):
        return N_min                                   # total reflux
    if R <= R_min:
        return math.inf
    Y = gilliland_y((R - R_min) / (R + 1.0))
    return (Y + N_min) / (1.0 - Y)


def required_reflux(N_available: float, N_min: float, R_min: float) -> float:
    """Reflux ratio needed to make the split with ``N_available`` total stages.

    The inverse of ``stages_at_reflux`` (used in rating mode, where the stage
    count is fixed by the column).  Returns infinity if ``N_available`` does
    not exceed ``N_min``.
    """
    if N_available <= N_min:
        return math.inf
    Y_target = (N_available - N_min) / (N_available + 1.0)
    X = brentq(lambda X: gilliland_y(X) - Y_target, 1e-12, 1.0 - 1e-12, xtol=1e-12)
    return (X + R_min) / (1.0 - X)


# --------------------------------------------------------------------------
# Cut analysis
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class BoundaryStages:
    """Stage requirement of one cut boundary."""

    name: str              # "lower" or "upper"
    T_boundary_c: float    # atmospheric-equivalent boundary [degC]
    alpha: float           # key relative volatility at the operating pressure
    z: float               # light-group mole fraction in the pot
    x_d: float             # light-group mole fraction overhead
    N_min: float           # Fenske, total reflux
    R_min: float           # minimum reflux ratio
    N_total: float         # at the requested R, including the pot (inf if R <= R_min)


@dataclass(frozen=True)
class StageRequirement:
    """Stages needed for a cut at a given reflux ratio."""

    R: float
    p_mbar: float
    spec: SeparationSpec
    boundaries: tuple[BoundaryStages, ...]   # empty if the whole feed is taken
    N_total: float          # governing boundary, including the pot
    N_packing: float        # stages the packed section must provide (N_total - 1)
    R_min: float            # governing minimum reflux ratio
    R_suggested: float      # R_FACTOR_DEFAULT x R_min
    feasible: bool          # False if R is at or below R_min


def _group_moles(feed: FeedCharacterisation, T_a_c: float, T_b_c: float) -> float:
    """Relative moles of feed boiling between two atmospheric-equivalent
    temperatures: (mass fraction in range) / (mean molecular weight of range)."""
    w_range = feed.cut_mass_fraction(T_a_c, T_b_c)
    w_cut, _ = feed.cut_composition(T_a_c, T_b_c)
    M_avg = 1.0 / float(np.sum(w_cut / feed.mw))
    return w_range / M_avg


def _boundary(name: str, T_b: float, alpha: float, n_light: float, n_heavy: float,
              spec: SeparationSpec, R: float) -> BoundaryStages:
    """Fenske / minimum reflux / Gilliland for one boundary."""
    z = n_light / (n_light + n_heavy)
    r = spec.r_key
    d_light = r * z                          # light group going overhead
    d_heavy = (1.0 - r) * (1.0 - z)          # heavy group carried overhead
    x_d = d_light / (d_light + d_heavy)

    N_min = minimum_stages(alpha, r)
    R_min = minimum_reflux(alpha, z, x_d)
    return BoundaryStages(name=name, T_boundary_c=T_b, alpha=alpha, z=z, x_d=x_d,
                          N_min=N_min, R_min=R_min, N_total=stages_at_reflux(N_min, R_min, R))


def cut_stage_requirement(
    feed: FeedCharacterisation,
    T_lo_c: float,
    T_hi_c: float,
    p_mbar: float,
    R: float,
    spec: SeparationSpec = SeparationSpec(),
) -> StageRequirement:
    """Theoretical stages needed to collect the cut ``T_lo_c``-``T_hi_c``.

    Parameters
    ----------
    feed             : characterised feed.
    T_lo_c, T_hi_c   : cut limits, atmospheric-equivalent degC.
    p_mbar           : top pressure (sets the key relative volatility).
    R                : operating reflux ratio (``math.inf`` for total reflux).
    spec             : separation spec (key recovery and grey-zone width).
    """
    cut = analyse_cut(feed, T_lo_c, T_hi_c, p_mbar, gap_c=spec.gap_c)
    ibp, fbp = feed.curve.ibp_c, feed.curve.fbp_c
    lo = max(T_lo_c, ibp)
    boundaries = []

    # Lower boundary: forerun (below T_lo) against everything else
    if T_lo_c > ibp:
        boundaries.append(_boundary("lower", T_lo_c, cut.alpha_lo,
                                    n_light=_group_moles(feed, ibp, T_lo_c),
                                    n_heavy=_group_moles(feed, T_lo_c, fbp),
                                    spec=spec, R=R))
    # Upper boundary: the cut against the material above it (forerun already gone)
    if T_hi_c < fbp:
        boundaries.append(_boundary("upper", T_hi_c, cut.alpha_hi,
                                    n_light=_group_moles(feed, lo, T_hi_c),
                                    n_heavy=_group_moles(feed, T_hi_c, fbp),
                                    spec=spec, R=R))

    if not boundaries:                       # whole feed requested: only the pot is needed
        return StageRequirement(R=R, p_mbar=p_mbar, spec=spec, boundaries=(), N_total=1.0,
                                N_packing=0.0, R_min=0.0, R_suggested=0.0, feasible=True)

    governing = max(boundaries, key=lambda b: b.N_total)
    R_min = max(b.R_min for b in boundaries)
    feasible = R > R_min
    if not feasible:
        warnings.warn(f"R = {R:.2f} is at or below the minimum reflux ratio {R_min:.2f}: "
                      "this cut cannot be made at the requested sharpness.", stacklevel=2)
    return StageRequirement(R=R, p_mbar=p_mbar, spec=spec, boundaries=tuple(boundaries),
                            N_total=governing.N_total,
                            N_packing=governing.N_total - 1.0,
                            R_min=R_min, R_suggested=R_FACTOR_DEFAULT * R_min, feasible=feasible)


# --------------------------------------------------------------------------
# Demo / smoke test:  python -m fdc.stages
# --------------------------------------------------------------------------

if __name__ == "__main__":
    from .properties.feed import DEFAULT_WPPO_TBP, TBPCurve, characterise_feed

    feed = characterise_feed(TBPCurve.from_points(DEFAULT_WPPO_TBP), n_cuts=20, sg_feed=0.80)
    spec = SeparationSpec(r_key=0.95, gap_c=10.0)

    print("Cut 150-250 C (atm-equivalent), key recovery 95 %, 10 C grey zone\n")
    for p in (1013.25, 20.0):
        res = cut_stage_requirement(feed, 150.0, 250.0, p, R=5.0, spec=spec)
        print(f"P_top = {p:7.2f} mbar, R = 5:  N_total {res.N_total:5.1f} (packing {res.N_packing:5.1f}), "
              f"R_min {res.R_min:.2f}, suggested R {res.R_suggested:.2f}")
        for b in res.boundaries:
            print(f"   {b.name:5s} boundary {b.T_boundary_c:5.0f} C: alpha {b.alpha:.2f}, z {b.z:.2f}, "
                  f"x_D {b.x_d:.2f}, N_min {b.N_min:5.1f}, R_min {b.R_min:.2f}, N {b.N_total:5.1f}")

    print("\nStages vs reflux ratio at 20 mbar:")
    for R in (3.0, 4.0, 6.0, 10.0, 20.0):
        res = cut_stage_requirement(feed, 150.0, 250.0, 20.0, R=R, spec=spec)
        print(f"  R = {R:5.1f}: N_total {res.N_total:6.1f}  packing {res.N_packing:6.1f}")
