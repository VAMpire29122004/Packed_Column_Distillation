"""
fdc.properties.vle
==================

Vapour-liquid equilibrium (VLE) for the WPPO pseudo-components built in
``feed.py``: vapour pressure, boiling point at vacuum, bubble/dew points and
relative volatility.

Corr-map position
-----------------
Drivers  : V02 (feed curve, via ``FeedCharacterisation``), V03 (cut range),
           V04 (P_top)
Solves   : V13 (relative volatility) and the *expected head-temperature
           window* for V11 (T_head).  The full T_head history during a batch
           is refined later in ``batch_pot.py``.

Method (all hydrocarbon-oriented, ideal-solution assumptions)
-------------------------------------------------------------
1. Critical temperature and pressure of each pseudo-component from the
   Kesler-Lee (1976) correlations (inputs: T_b and SG).
2. Vapour pressure from the Lee-Kesler (1975) Pitzer-type equation
       ln Pr = f0(Tr) + omega * f1(Tr).
3. The acentric factor is *anchored*: it is solved so that the vapour
   pressure at the normal boiling point is exactly 1 atm.  This guarantees
   that the atmospheric-equivalent boiling points in the TBP curve are
   reproduced exactly, whatever error the critical-property correlations
   carry.  The correlation then only has to extrapolate to vacuum.
4. Raoult's law (ideal liquid, ideal vapour) for bubble/dew points and
   relative volatility.  This is reasonable for a mixture of similar
   hydrocarbons at vacuum; it is not meant for strongly polar mixtures.

Conventions
-----------
* Public pressures are in mbar (absolute), public temperatures in degC.
  Internally pressures are in bar and temperatures in kelvin.
* The boundaries of a cut are given as *atmospheric-equivalent* temperatures
  (the same basis as the TBP curve); this module converts them to the
  temperatures they correspond to at the operating pressure.

Main references
---------------
* Kesler & Lee (1976), Hydrocarbon Processing 55(3), 153-158.
* Lee & Kesler (1975), AIChE J. 21(3), 510-527.
* Riazi, "Characterization and Properties of Petroleum Fractions" (ASTM, 2005).

Known limitations
-----------------
* Kesler-Lee critical properties degrade for very heavy fractions (T_b above
  roughly 700 K); a warning is raised.
* Raoult's law ignores liquid non-ideality (olefin/paraffin/aromatic mixing).
* Vapour pressures extrapolated far above a component's critical temperature
  (light ends in a hot pot) are only qualitative.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import brentq

from .feed import (
    FeedCharacterisation,
    KELVIN_OFFSET,
    RANKINE_PER_KELVIN,
    sg_from_watson,
)

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

P_ATM_BAR = 1.01325            # standard atmosphere [bar]
MBAR_PER_BAR = 1000.0
PSI_TO_BAR = 0.0689476
TB_WARN_K = 700.0              # above this Kesler-Lee gets unreliable
TR_MIN = 0.30                  # lower end of the Lee-Kesler reduced-temperature range (0.3-1.0)


# --------------------------------------------------------------------------
# Correlations
# --------------------------------------------------------------------------

def critical_properties(Tb_K, sg):
    """Critical temperature [K] and pressure [bar] from T_b and SG.

    Kesler-Lee (1976).  The published equations use degR and psia, so the
    inputs are converted on the way in and the outputs on the way out.
    """
    Tb_K = np.asarray(Tb_K, dtype=float)
    sg = np.asarray(sg, dtype=float)
    Tb_R = RANKINE_PER_KELVIN * Tb_K

    Tc_R = (341.7 + 811.0 * sg
            + (0.4244 + 0.1174 * sg) * Tb_R
            + (0.4669 - 3.2623 * sg) * 1.0e5 / Tb_R)

    ln_pc_psia = (8.3634 - 0.0566 / sg
                  - (0.24244 + 2.2898 / sg + 0.11857 / sg ** 2) * 1.0e-3 * Tb_R
                  + (1.4685 + 3.648 / sg + 0.47227 / sg ** 2) * 1.0e-7 * Tb_R ** 2
                  - (0.42019 + 1.6977 / sg ** 2) * 1.0e-10 * Tb_R ** 3)

    return Tc_R / RANKINE_PER_KELVIN, np.exp(ln_pc_psia) * PSI_TO_BAR


def _f0(Tr):
    """Simple-fluid term of the Lee-Kesler vapour-pressure equation."""
    return 5.92714 - 6.09648 / Tr - 1.28862 * np.log(Tr) + 0.169347 * Tr ** 6


def _f1(Tr):
    """Deviation term of the Lee-Kesler vapour-pressure equation."""
    return 15.2518 - 15.6875 / Tr - 13.4721 * np.log(Tr) + 0.43577 * Tr ** 6


def anchored_acentric_factor(Tb_K, Tc_K, Pc_bar):
    """Acentric factor chosen so that Pv(T_b) = 1 atm exactly.

    Solving ln(P_atm/Pc) = f0(Tbr) + omega * f1(Tbr) for omega pins the
    vapour-pressure curve to the measured normal boiling point.
    """
    Tbr = np.asarray(Tb_K) / np.asarray(Tc_K)
    if np.any(Tbr >= 1.0):
        raise ValueError("Boiling point at or above the estimated critical temperature.")
    return (np.log(P_ATM_BAR / Pc_bar) - _f0(Tbr)) / _f1(Tbr)


# --------------------------------------------------------------------------
# Property container with the VLE operations
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class VLEProps:
    """Vapour-pressure parameters for a set of (pseudo-)components."""

    Tb_K: np.ndarray
    Tc_K: np.ndarray
    Pc_bar: np.ndarray
    omega: np.ndarray

    def psat_bar(self, T_K: float) -> np.ndarray:
        """Vapour pressure [bar] of every component at temperature ``T_K``."""
        Tr = T_K / self.Tc_K
        return self.Pc_bar * np.exp(_f0(Tr) + self.omega * _f1(Tr))

    def boiling_T_K(self, p_bar: float) -> np.ndarray:
        """Temperature [K] at which each pure component boils at ``p_bar``.

        Pv(T) rises monotonically with T, so a bracketed root find is safe.
        """
        out = np.empty_like(self.Tb_K)
        for i in range(self.Tb_K.size):
            Tc = self.Tc_K[i]

            def residual(T, i=i):
                return np.log(self.psat_bar(T)[i] / p_bar)

            # Bracket within the range the Lee-Kesler equation is valid for
            # (Tr from 0.3 up to just below the critical point).
            lo, hi = TR_MIN * Tc, 0.999 * Tc
            if residual(lo) > 0 or residual(hi) < 0:
                raise ValueError(f"Pressure {p_bar * MBAR_PER_BAR:.4g} mbar is outside the "
                                 f"range the vapour-pressure correlation covers for "
                                 f"component {i} (Tr {TR_MIN}-1.0).")
            out[i] = brentq(residual, lo, hi, xtol=1e-6)
        return out

    def bubble_point(self, x: np.ndarray, p_bar: float):
        """Bubble-point temperature [K] and vapour composition y (Raoult's law).

        Solves sum(x_i * Pv_i(T)) = P.  The root lies between the lightest and
        heaviest component's pure boiling points, which gives the bracket.
        """
        x = np.asarray(x, dtype=float)
        x = x / x.sum()
        Ti = self.boiling_T_K(p_bar)[x > 0]
        lo, hi = Ti.min(), Ti.max()
        if np.isclose(lo, hi):
            T = lo
        else:
            T = brentq(lambda T: np.sum(x * self.psat_bar(T)) / p_bar - 1.0, lo, hi, xtol=1e-6)
        y = x * self.psat_bar(T) / p_bar
        return float(T), y / y.sum()

    def dew_point(self, y: np.ndarray, p_bar: float):
        """Dew-point temperature [K] and liquid composition x (Raoult's law).

        Solves sum(y_i * P / Pv_i(T)) = 1.
        """
        y = np.asarray(y, dtype=float)
        y = y / y.sum()
        Ti = self.boiling_T_K(p_bar)[y > 0]
        lo, hi = Ti.min(), Ti.max()
        if np.isclose(lo, hi):
            T = lo
        else:
            T = brentq(lambda T: p_bar * np.sum(y / self.psat_bar(T)) - 1.0, lo, hi, xtol=1e-6)
        x = y * p_bar / self.psat_bar(T)
        return float(T), x / x.sum()


def build_vle_props(Tb_K, sg) -> VLEProps:
    """Assemble VLE parameters from boiling points and specific gravities."""
    Tb_K = np.atleast_1d(np.asarray(Tb_K, dtype=float))
    sg = np.atleast_1d(np.asarray(sg, dtype=float))
    if np.any(Tb_K > TB_WARN_K):
        warnings.warn(
            f"Boiling point above {TB_WARN_K:.0f} K: Kesler-Lee critical properties "
            "are unreliable for such heavy fractions.", stacklevel=2)
    Tc, Pc = critical_properties(Tb_K, sg)
    return VLEProps(Tb_K=Tb_K, Tc_K=Tc, Pc_bar=Pc, omega=anchored_acentric_factor(Tb_K, Tc, Pc))


def feed_vle_props(feed: FeedCharacterisation) -> VLEProps:
    """VLE parameters for the pseudo-components of a characterised feed."""
    return build_vle_props(feed.Tb_K, feed.sg)


# --------------------------------------------------------------------------
# Hypothetical single components (used for cut boundaries)
# --------------------------------------------------------------------------

def _hypothetical(Tb_atm_c, k_watson: float) -> VLEProps:
    """Component(s) with given atmospheric boiling point(s) and the feed's Watson K."""
    Tb_K = np.atleast_1d(np.asarray(Tb_atm_c, dtype=float)) + KELVIN_OFFSET
    return build_vle_props(Tb_K, sg_from_watson(Tb_K, k_watson))


def boiling_point_at_pressure(Tb_atm_c, p_mbar: float, k_watson: float):
    """Boiling temperature [degC] at ``p_mbar`` of component(s) whose
    atmospheric-equivalent boiling point is ``Tb_atm_c`` [degC].

    This is the vacuum conversion used for cut boundaries: a 300 degC (atm)
    fraction boils at a much lower temperature at 20 mbar.
    """
    props = _hypothetical(Tb_atm_c, k_watson)
    T_c = props.boiling_T_K(p_mbar / MBAR_PER_BAR) - KELVIN_OFFSET
    return T_c if np.ndim(Tb_atm_c) else float(T_c[0])


def relative_volatility(Tb_lk_c: float, Tb_hk_c: float, T_c: float, k_watson: float) -> float:
    """alpha = Pv(light key) / Pv(heavy key) at temperature ``T_c`` [degC].

    The keys are given by their atmospheric-equivalent boiling points; the
    light key is the lower-boiling one.
    """
    if Tb_hk_c <= Tb_lk_c:
        raise ValueError("Heavy key must boil higher than the light key.")
    props = _hypothetical([Tb_lk_c, Tb_hk_c], k_watson)
    pv = props.psat_bar(T_c + KELVIN_OFFSET)
    return float(pv[0] / pv[1])


# --------------------------------------------------------------------------
# Cut analysis: the pieces V11 and V13 of the corr map need
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CutVLE:
    """VLE summary of a boiling-range cut at the operating pressure."""

    p_mbar: float
    T_lo_c: float        # cut lower boundary at p_mbar [degC]
    T_mid_c: float       # boiling point at p_mbar of the cut's mean fraction [degC]
    T_hi_c: float        # cut upper boundary at p_mbar [degC]
    alpha_lo: float      # relative volatility at the lower boundary (nan for first cut)
    alpha_hi: float      # relative volatility at the upper boundary (nan for last cut)
    alpha_min: float     # the harder of the two separations (governs stages)
    gap_c: float         # key-component half-gap used [degC]


def _cut_mean_tb_c(feed: FeedCharacterisation, T_lo_c: float, T_hi_c: float) -> float:
    """Mass-average atmospheric boiling point of the fraction inside the cut."""
    w_lo = float(feed.curve.frac_below(T_lo_c))
    w_hi = float(feed.curve.frac_below(T_hi_c))
    if w_hi - w_lo < 1e-9:
        return 0.5 * (T_lo_c + T_hi_c)
    w = np.linspace(w_lo, w_hi, 41)
    return float(trapezoid(feed.curve.temp_at(w), w) / (w_hi - w_lo))


def analyse_cut(
    feed: FeedCharacterisation,
    T_lo_c: float,
    T_hi_c: float,
    p_mbar: float,
    gap_c: float = 10.0,
) -> CutVLE:
    """Head-temperature window and relative volatility for a cut.

    Parameters
    ----------
    T_lo_c, T_hi_c : cut limits, atmospheric-equivalent degC (TBP basis).
    p_mbar         : top pressure P_top, absolute.
    gap_c          : ASSUMPTION - the light and heavy keys sit ``gap_c`` either
                     side of each cut boundary (atmospheric-equivalent).  It
                     stands in for the purity spec: a smaller gap means a
                     harder separation (alpha closer to 1).  ``stages.py`` will
                     later tie this to the required purity.

    The head vapour temperature during collection of the cut is expected to
    rise from ``T_lo_c`` to ``T_hi_c`` (the converted boundaries), which is the
    window an operator would watch on TT-102.
    """
    if T_hi_c <= T_lo_c:
        raise ValueError("Cut upper limit must be above the lower limit.")
    k = feed.k_watson

    T_lo_p = boiling_point_at_pressure(T_lo_c, p_mbar, k)
    T_hi_p = boiling_point_at_pressure(T_hi_c, p_mbar, k)
    T_mid_p = boiling_point_at_pressure(_cut_mean_tb_c(feed, T_lo_c, T_hi_c), p_mbar, k)

    def alpha_at(boundary_atm_c: float, boundary_p_c: float) -> float:
        # keys straddle the boundary; evaluate where the boundary boils at p
        return relative_volatility(boundary_atm_c - gap_c, boundary_atm_c + gap_c, boundary_p_c, k)

    # A boundary only exists if the cut does not start at the IBP / end at the FBP
    alpha_lo = alpha_at(T_lo_c, T_lo_p) if T_lo_c > feed.curve.ibp_c else float("nan")
    alpha_hi = alpha_at(T_hi_c, T_hi_p) if T_hi_c < feed.curve.fbp_c else float("nan")
    if np.isnan(alpha_lo) and np.isnan(alpha_hi):
        alpha_min = float("nan")      # whole feed requested: no separation needed
    else:
        alpha_min = float(np.nanmin([alpha_lo, alpha_hi]))

    return CutVLE(p_mbar=p_mbar, T_lo_c=T_lo_p, T_mid_c=T_mid_p, T_hi_c=T_hi_p,
                  alpha_lo=alpha_lo, alpha_hi=alpha_hi, alpha_min=alpha_min, gap_c=gap_c)


# --------------------------------------------------------------------------
# Demo / smoke test:  python -m fdc.properties.vle
# --------------------------------------------------------------------------

if __name__ == "__main__":
    from .feed import DEFAULT_WPPO_TBP, TBPCurve, characterise_feed

    feed = characterise_feed(TBPCurve.from_points(DEFAULT_WPPO_TBP), n_cuts=10, sg_feed=0.80)
    props = feed_vle_props(feed)

    p_vac = 20.0  # mbar
    print(f"{'cut':>3} {'Tb atm/C':>9} {'Tc/C':>7} {'Pc/bar':>7} {'omega':>6} {'Tb@20mbar/C':>12}")
    Tb_vac = props.boiling_T_K(p_vac / MBAR_PER_BAR) - KELVIN_OFFSET
    for i in range(feed.n_cuts):
        print(f"{i + 1:>3} {props.Tb_K[i] - KELVIN_OFFSET:9.1f} {props.Tc_K[i] - KELVIN_OFFSET:7.1f} "
              f"{props.Pc_bar[i]:7.1f} {props.omega[i]:6.3f} {Tb_vac[i]:12.1f}")

    print("\nCut 150-250 C (atmospheric equivalent), 10 C key gap:")
    for p in (1013.25, 100.0, 20.0):
        c = analyse_cut(feed, 150.0, 250.0, p)
        print(f"  P_top {p:7.2f} mbar: head window {c.T_lo_c:6.1f} - {c.T_hi_c:6.1f} C "
              f"(mid {c.T_mid_c:6.1f}), alpha_lo {c.alpha_lo:.3f}, alpha_hi {c.alpha_hi:.3f}")
