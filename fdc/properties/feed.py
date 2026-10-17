"""
fdc.properties.feed
===================

Feed characterisation for waste plastic pyrolysis oil (WPPO).

WPPO is a wide-boiling mixture of hundreds of hydrocarbons, so it cannot be
treated as one component with one density.  The standard refinery approach is
to describe it by a True Boiling Point (TBP) curve and split that curve into
*pseudo-components*, each with its own boiling point, specific gravity and
molecular weight.  Every later module (vapour pressure, relative volatility,
stage calculations, batch cut simulation) works on these pseudo-components.

Corr-map position
-----------------
Solves / supplies  : V02 (TBP, feed curve) -> pseudo-component set
Used downstream by : V03 cut mass, V12 fluid properties, V13 relative
                     volatility, V19 stages, V24 yield  (see FDC_corr_map.xlsx)

Conventions
-----------
* Public temperature inputs are in degC and are *atmospheric-equivalent* TBP
  temperatures.  Internal pseudo-component arrays are in kelvin.
* Compositions are mass fractions unless the name says otherwise (``x_mol``).
* SG is specific gravity at 60/60 degF (about 15.6 degC).  Temperature
  dependence of density is handled later in ``fluids.py``.

Main references
---------------
* Watson / UOP characterisation factor, K_w = (T_b[degR])^(1/3) / SG.
* Riazi & Daubert (1980), "Simplify property predictions", Hydrocarbon
  Processing 59(3), 115-116 (molecular weight from T_b and SG).
* Whitson & Brule, "Phase Behavior" (SPE Monograph 20), for the
  pseudo-component splitting idea.

Typical use
-----------
>>> curve = TBPCurve.from_points(DEFAULT_WPPO_TBP)
>>> feed = characterise_feed(curve, n_cuts=20, sg_feed=0.80)
>>> feed.cut_mass_fraction(150.0, 250.0)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import PchipInterpolator

# --------------------------------------------------------------------------
# Constants and placeholder data
# --------------------------------------------------------------------------

KELVIN_OFFSET = 273.15          # degC -> K
RANKINE_PER_KELVIN = 1.8        # K -> degR

# Plausible Watson K range for hydrocarbon mixtures: aromatic-rich (~10) to
# highly paraffinic (~13).  Values outside this range trigger a warning.
K_WATSON_MIN, K_WATSON_MAX = 9.5, 13.5

# ASSUMPTION / PLACEHOLDER: illustrative polyolefin-pyrolysis-oil TBP curve as
# (cumulative mass % distilled, atmospheric-equivalent temperature degC).
# Replace this with your own SimDis (ASTM D2887) or TBP data before trusting
# any numerical result.
DEFAULT_WPPO_TBP: tuple[tuple[float, float], ...] = (
    (0.0, 40.0),     # IBP
    (10.0, 95.0),
    (30.0, 180.0),
    (50.0, 260.0),
    (70.0, 340.0),
    (90.0, 430.0),
    (100.0, 500.0),  # FBP
)


# --------------------------------------------------------------------------
# TBP curve
# --------------------------------------------------------------------------

class TBPCurve:
    """Monotone interpolation of a TBP curve in both directions.

    Stores cumulative mass fraction ``w`` (0..1) against atmospheric-equivalent
    temperature ``T`` (degC).  PCHIP (shape-preserving cubic) interpolation is
    used so the curve never overshoots or turns back on itself between the
    measured points, which a plain cubic spline can do on sparse data.
    """

    def __init__(self, w: np.ndarray, T_c: np.ndarray):
        w = np.asarray(w, dtype=float)
        T_c = np.asarray(T_c, dtype=float)

        # --- validation: the curve must run from 0 to 1 and strictly increase
        if w.shape != T_c.shape or w.ndim != 1 or w.size < 3:
            raise ValueError("TBP curve needs at least 3 (mass %, T) points.")
        if not (np.isclose(w[0], 0.0) and np.isclose(w[-1], 1.0)):
            raise ValueError("TBP curve must start at 0 %% (IBP) and end at 100 %% (FBP).")
        if np.any(np.diff(w) <= 0) or np.any(np.diff(T_c) <= 0):
            raise ValueError("Mass % and temperature must both be strictly increasing.")

        self.w = w
        self.T_c = T_c
        self._T_of_w = PchipInterpolator(w, T_c)   # w -> T
        self._w_of_T = PchipInterpolator(T_c, w)   # T -> w (inverse, also monotone)

    @classmethod
    def from_points(cls, points: Sequence[tuple[float, float]]) -> "TBPCurve":
        """Build from (cumulative mass %, T in degC) pairs, e.g. DEFAULT_WPPO_TBP."""
        pts = np.asarray(points, dtype=float)
        order = np.argsort(pts[:, 0])
        return cls(pts[order, 0] / 100.0, pts[order, 1])

    # -- lookups ------------------------------------------------------------
    @property
    def ibp_c(self) -> float:
        return float(self.T_c[0])

    @property
    def fbp_c(self) -> float:
        return float(self.T_c[-1])

    def temp_at(self, w: np.ndarray | float) -> np.ndarray | float:
        """Temperature (degC) at which cumulative mass fraction ``w`` has distilled."""
        return self._T_of_w(np.clip(w, 0.0, 1.0))

    def frac_below(self, T_c: np.ndarray | float) -> np.ndarray | float:
        """Cumulative mass fraction distilled by temperature ``T_c`` (degC).

        Clamped to 0 below the IBP and 1 above the FBP.
        """
        T = np.clip(T_c, self.ibp_c, self.fbp_c)
        return self._w_of_T(T)


# --------------------------------------------------------------------------
# Correlations for pseudo-component properties
# --------------------------------------------------------------------------

def _watson_s(Tb_K: np.ndarray) -> np.ndarray:
    """Cube root of boiling point in degR: the numerator of the Watson factor."""
    return (RANKINE_PER_KELVIN * Tb_K) ** (1.0 / 3.0)


def sg_from_watson(Tb_K: np.ndarray, k_watson: float) -> np.ndarray:
    """Specific gravity (60/60 degF) from boiling point and Watson K.

    Rearranged from K_w = (T_b[degR])^(1/3) / SG.  A single K_w for the whole
    feed is the usual first approximation; it assumes the chemical *type*
    (paraffinic / olefinic / aromatic) does not change across the boiling range.
    """
    return _watson_s(Tb_K) / k_watson


def mw_riazi_daubert(Tb_K: np.ndarray, sg: np.ndarray) -> np.ndarray:
    """Molecular weight [g/mol] from Riazi-Daubert (1980).

    M = 1.6607e-4 * Tb^2.1962 * SG^-1.0164   (Tb in K).

    Checked against n-alkanes it is typically within about 3-8 % (high),
    which is acceptable for pseudo-components but not for pure species.
    """
    return 1.6607e-4 * Tb_K ** 2.1962 * sg ** -1.0164


def calibrate_k_watson(Tb_K: np.ndarray, w: np.ndarray, sg_feed: float) -> float:
    """Closed-form Watson K that reproduces a measured bulk feed SG.

    Assuming ideal volume mixing, SG_mix = 1 / sum(w_i / SG_i).  With
    SG_i = s_i / K (s_i = (1.8 Tb_i)^(1/3)) this inverts exactly to
    K = 1 / (SG_mix * sum(w_i / s_i)), so no root-finding is required.
    """
    return 1.0 / (sg_feed * float(np.sum(w / _watson_s(Tb_K))))


# --------------------------------------------------------------------------
# Characterised feed
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FeedCharacterisation:
    """Pseudo-component description of the feed (all arrays share one length)."""

    curve: TBPCurve
    w: np.ndarray          # mass fraction of each pseudo-component (sums to 1)
    Tb_K: np.ndarray       # normal (atmospheric) boiling point [K]
    sg: np.ndarray         # specific gravity, 60/60 degF [-]
    mw: np.ndarray         # molecular weight [g/mol]
    x_mol: np.ndarray      # mole fraction (sums to 1)
    k_watson: float        # Watson K actually used [-]

    # -- bulk properties ------------------------------------------------------
    @property
    def n_cuts(self) -> int:
        return int(self.w.size)

    @property
    def sg_feed(self) -> float:
        """Bulk feed SG assuming ideal volume mixing."""
        return float(1.0 / np.sum(self.w / self.sg))

    @property
    def mw_avg(self) -> float:
        """Mole-averaged molecular weight of the feed [g/mol]."""
        return float(1.0 / np.sum(self.w / self.mw))

    # -- cuts ----------------------------------------------------------------
    def cut_mass_fraction(self, T_lo_c: float, T_hi_c: float) -> float:
        """Mass fraction of the feed boiling between two atmospheric-equivalent
        temperatures, read straight from the TBP curve (not from the discrete
        pseudo-components, so it has no discretisation error)."""
        if T_hi_c <= T_lo_c:
            raise ValueError("Cut upper limit must be above the lower limit.")
        return float(self.curve.frac_below(T_hi_c) - self.curve.frac_below(T_lo_c))

    def cut_mass(self, m_feed_kg: float, T_lo_c: float, T_hi_c: float) -> float:
        """Mass [kg] of the desired cut available in a charge of ``m_feed_kg``."""
        return m_feed_kg * self.cut_mass_fraction(T_lo_c, T_hi_c)

    def cut_composition(self, T_lo_c: float, T_hi_c: float) -> tuple[np.ndarray, np.ndarray]:
        """Composition of the feed material boiling inside a cut.

        Each pseudo-component covers one equal-mass slice of the TBP curve.  A
        slice that straddles a cut boundary is included in proportion to the
        part of it that lies inside the cut, so the result changes smoothly as
        the limits move (no jumps when a boundary crosses a slice edge).

        Returns
        -------
        (w_cut, x_cut) : mass fractions and mole fractions over *all*
        pseudo-components (zero outside the cut), each summing to 1.
        """
        if T_hi_c <= T_lo_c:
            raise ValueError("Cut upper limit must be above the lower limit.")
        w_lo = float(self.curve.frac_below(T_lo_c))
        w_hi = float(self.curve.frac_below(T_hi_c))
        if w_hi - w_lo <= 0.0:
            raise ValueError("No feed boils inside this cut.")

        # Slice edges in cumulative mass fraction (equal-mass slices, as built
        # in characterise_feed) and the overlap of each slice with the cut.
        edges = np.linspace(0.0, 1.0, self.n_cuts + 1)
        overlap = np.clip(np.minimum(edges[1:], w_hi) - np.maximum(edges[:-1], w_lo), 0.0, None)

        w_cut = self.w * overlap * self.n_cuts      # slice mass x fraction of slice inside
        w_cut = w_cut / w_cut.sum()
        n = w_cut / self.mw                          # moles are proportional to w / M
        return w_cut, n / n.sum()

    def as_table(self) -> dict[str, np.ndarray]:
        """Column-wise dict (easy to wrap in a DataFrame or st.dataframe)."""
        return {
            "w": self.w,
            "Tb_C": self.Tb_K - KELVIN_OFFSET,
            "SG": self.sg,
            "MW": self.mw,
            "x_mol": self.x_mol,
        }


def characterise_feed(
    curve: TBPCurve,
    n_cuts: int = 20,
    k_watson: float = 12.0,
    sg_feed: float | None = None,
) -> FeedCharacterisation:
    """Split a TBP curve into equal-mass pseudo-components.

    Parameters
    ----------
    curve     : TBPCurve of the feed.
    n_cuts    : number of pseudo-components.  More cuts resolve sharper
                separations but cost run time in later modules (20-30 is a
                sensible range for a lab column).
    k_watson  : Watson K used when ``sg_feed`` is not given.  ASSUMPTION: 12.0
                is a typical value for a mostly paraffinic/olefinic
                polyolefin pyrolysis oil; measure the oil density to do better.
    sg_feed   : measured bulk SG (60/60 degF).  If given, K_w is calibrated so
                the model reproduces it, and ``k_watson`` is ignored.
    """
    if n_cuts < 3:
        raise ValueError("Use at least 3 pseudo-components.")

    # Equal-mass slices: each pseudo-component carries 1/n_cuts of the feed.
    w_edges = np.linspace(0.0, 1.0, n_cuts + 1)
    w = np.full(n_cuts, 1.0 / n_cuts)

    # Representative boiling point of a slice = mass-average of the TBP
    # temperature across the slice (trapezoidal integral over 41 sub-points).
    Tb_K = np.empty(n_cuts)
    for i in range(n_cuts):
        sub = np.linspace(w_edges[i], w_edges[i + 1], 41)
        Tb_K[i] = trapezoid(curve.temp_at(sub), sub) / (w_edges[i + 1] - w_edges[i]) + KELVIN_OFFSET

    if sg_feed is not None:
        if not 0.5 < sg_feed < 1.1:
            raise ValueError("sg_feed should be a specific gravity near 0.7-0.9.")
        k_watson = calibrate_k_watson(Tb_K, w, sg_feed)
    if not K_WATSON_MIN <= k_watson <= K_WATSON_MAX:
        warnings.warn(
            f"Watson K = {k_watson:.2f} is outside the usual {K_WATSON_MIN}-{K_WATSON_MAX} "
            "range for hydrocarbon mixtures; check the density or TBP data.",
            stacklevel=2,
        )

    sg = sg_from_watson(Tb_K, k_watson)
    mw = mw_riazi_daubert(Tb_K, sg)

    # Mole fractions from mass fractions: n_i is proportional to w_i / M_i.
    n = w / mw
    x_mol = n / n.sum()

    return FeedCharacterisation(curve=curve, w=w, Tb_K=Tb_K, sg=sg, mw=mw,
                                x_mol=x_mol, k_watson=float(k_watson))


# --------------------------------------------------------------------------
# Demo / smoke test:  python -m fdc.properties.feed
# --------------------------------------------------------------------------

if __name__ == "__main__":
    curve = TBPCurve.from_points(DEFAULT_WPPO_TBP)
    feed = characterise_feed(curve, n_cuts=10, sg_feed=0.80)

    print(f"Watson K (calibrated to SG 0.80): {feed.k_watson:.3f}")
    print(f"Bulk SG {feed.sg_feed:.3f} | average MW {feed.mw_avg:.1f} g/mol\n")
    print(f"{'cut':>3} {'w':>6} {'Tb/C':>7} {'SG':>6} {'MW':>6} {'x_mol':>7}")
    for i in range(feed.n_cuts):
        print(f"{i + 1:>3} {feed.w[i]:6.3f} {feed.Tb_K[i] - KELVIN_OFFSET:7.1f} "
              f"{feed.sg[i]:6.3f} {feed.mw[i]:6.1f} {feed.x_mol[i]:7.4f}")

    lo, hi = 150.0, 250.0
    print(f"\nMass fraction boiling {lo:.0f}-{hi:.0f} C: {feed.cut_mass_fraction(lo, hi):.3f}")
    print(f"Cut mass from a 1.0 kg charge: {feed.cut_mass(1.0, lo, hi):.3f} kg")
