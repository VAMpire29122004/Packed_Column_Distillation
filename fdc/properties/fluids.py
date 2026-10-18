"""
fdc.properties.fluids
=====================

Transport and thermodynamic fluid properties of the WPPO pseudo-components and
of mixtures of them: liquid density, vapour density, liquid viscosity, surface
tension and latent heat of vaporisation.

Corr-map position
-----------------
Drivers : V02 (feed, via ``FeedCharacterisation``), V04 (pressure),
          V10 / V11 (pot and head temperature), plus the liquid / vapour
          composition at the point of interest.
Solves  : V12 (fluid properties).  Used downstream by ``hydraulics.py``
          (rho_V, mu_V, rho_L, mu, sigma), ``holdup.py``, ``hetp.py`` and the
          energy balance in ``batch_pot.py`` (latent heat).

Methods
-------
Per component (T in K, built on the critical constants from ``vle.py``):
* Liquid density : Rackett-type temperature scaling (Spencer-Danner Z_RA)
                   anchored to the SG at 60 degF, so the model reproduces the
                   measured density at the reference temperature exactly.
* Viscosity      : Orrick-Erbar group-contribution form with the n-alkane
                   carbon-number terms, carbon number taken from molecular weight.
* Surface tension: Brock-Bird corresponding-states correlation.
* Latent heat    : Chen's equation at the normal boiling point, moved to T with
                   the Watson exponent (0.38).
* Vapour viscosity: Chung et al. (1988) low-pressure method (nonpolar form),
                   with the critical volume from Zc = 0.291 - 0.08 omega.
Mixing rules:
* Liquid density : ideal volume additivity (mass-fraction weighted 1/rho).
* Viscosity      : Kendall-Monroe, mu^(1/3) = sum(x_i * mu_i^(1/3)).
* Surface tension: mole-fraction average.
* Vapour density : ideal gas, rho = P M / (R T).
* Vapour viscosity: Herning-Zipperer, mu = sum(y_i mu_i sqrt(M_i)) / sum(y_i sqrt(M_i)).
* Latent heat    : mole-fraction average, converted to a mass basis.

Conventions
-----------
* Public temperatures in degC, pressures in mbar (absolute).
* Outputs: rho in kg/m3, viscosities in mPa.s, surface tension in mN/m,
  latent heat in kJ/kg, molecular weight in g/mol.

Main references
---------------
* Spencer & Danner (1972), J. Chem. Eng. Data 17, 236 (modified Rackett).
* Orrick & Erbar, in Poling, Prausnitz & O'Connell, "The Properties of Gases
  and Liquids", 5th ed. (liquid viscosity, group contributions).
* Brock & Bird (1955), AIChE J. 1, 174 (surface tension).
* Chen (1965) and Watson (1943) (latent heat), same reference text.
* Chung, Ajlan, Lee & Starling (1988), Ind. Eng. Chem. Res. 27, 671 (vapour
  viscosity); Herning & Zipperer (1936) (vapour mixing rule).
* Kendall & Monroe (1917), J. Am. Chem. Soc. 39, 1787 (mixture viscosity).

Known limitations
-----------------
* Treats every pseudo-component as a saturated paraffin of equal molecular
  weight for viscosity; olefin/aromatic content shifts real values.
* Orrick-Erbar is a low-to-moderate temperature correlation (reduced
  temperature up to roughly 0.75); values for a hot pot are indicative.
* Vapour treated as an ideal gas; fine at vacuum, a few percent off for heavy
  vapours near atmospheric pressure.
* Best practice: calibrate density and viscosity against measurements of your
  own oil (see ``feed.characterise_feed(sg_feed=...)`` for density).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .feed import FeedCharacterisation, KELVIN_OFFSET
from .vle import MBAR_PER_BAR, P_ATM_BAR, VLEProps, feed_vle_props

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

R_GAS = 8.314462618            # J/(mol K)
RHO_WATER_60F = 999.0          # kg/m3, water at 60 degF (SG reference)
T_REF_K = 288.706              # 60 degF in K, reference of the SG
MW_H2 = 2.016                  # g/mol, end-group offset for CnH(2n+2)
MW_CH2 = 14.027                # g/mol per CH2 unit


# --------------------------------------------------------------------------
# Component set: everything the property equations need, per pseudo-component
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Components:
    """Pseudo-component data: VLE constants plus SG and molecular weight."""

    vle: VLEProps
    sg: np.ndarray       # specific gravity 60/60 degF
    mw: np.ndarray       # molecular weight [g/mol]

    @classmethod
    def from_feed(cls, feed: FeedCharacterisation) -> "Components":
        return cls(vle=feed_vle_props(feed), sg=feed.sg, mw=feed.mw)


# --------------------------------------------------------------------------
# Pure-component correlations (vectorised over components)
# --------------------------------------------------------------------------

def liquid_density(T_K: float, sg, Tc_K, omega):
    """Saturated-liquid density [kg/m3] at ``T_K``.

    Rackett volume ratio V(T)/V(T_ref) = Z_RA ^ [(1-Tr)^(2/7) - (1-Tr_ref)^(2/7)]
    with Z_RA = 0.29056 - 0.08775*omega, applied to the density measured at
    60 degF (rho_ref = SG x 999 kg/m3).  Only the *temperature trend* comes from
    the equation of state-like correlation; the level is your measured SG.
    """
    Z = 0.29056 - 0.08775 * np.asarray(omega)
    e_T = np.clip(1.0 - T_K / Tc_K, 1e-6, None) ** (2.0 / 7.0)
    e_ref = np.clip(1.0 - T_REF_K / Tc_K, 1e-6, None) ** (2.0 / 7.0)
    return np.asarray(sg) * RHO_WATER_60F / Z ** (e_T - e_ref)


def liquid_viscosity_mpas(T_K: float, mw, rho_kg_m3):
    """Pure-liquid viscosity [mPa.s] from the Orrick-Erbar form.

        ln( mu / (rho * M) ) = A + B / T
        A = -(6.95 + 0.21 N),  B = 275 + 99 N     (N = carbon number)

    with mu in cP, rho in g/cm3, M in g/mol.  The carbon number is taken from
    the molecular weight assuming CnH(2n+2): N = (M - 2.016) / 14.027.
    """
    mw = np.asarray(mw)
    N = (mw - MW_H2) / MW_CH2
    A = -(6.95 + 0.21 * N)
    B = 275.0 + 99.0 * N
    return (np.asarray(rho_kg_m3) / 1000.0) * mw * np.exp(A + B / T_K)


def surface_tension_mnm(T_K: float, Tb_K, Tc_K, Pc_bar):
    """Surface tension [mN/m] from Brock-Bird.

        sigma = Pc^(2/3) Tc^(1/3) Q (1 - Tr)^(11/9)
        Q = 0.1196 [1 + Tbr ln(Pc/1.01325) / (1 - Tbr)] - 0.279

    (Pc in bar, T in K, sigma in dyn/cm = mN/m).  Goes to zero at the critical
    temperature.
    """
    Tbr = np.asarray(Tb_K) / np.asarray(Tc_K)
    Q = 0.1196 * (1.0 + Tbr * np.log(Pc_bar / P_ATM_BAR) / (1.0 - Tbr)) - 0.279
    one_minus_Tr = np.clip(1.0 - T_K / np.asarray(Tc_K), 0.0, None)
    return Pc_bar ** (2.0 / 3.0) * Tc_K ** (1.0 / 3.0) * Q * one_minus_Tr ** (11.0 / 9.0)


def latent_heat_jmol(T_K: float, Tb_K, Tc_K, Pc_bar):
    """Molar heat of vaporisation [J/mol] at ``T_K``.

    Chen's equation at the normal boiling point,
        dHvb = R Tb (3.978 Tbr - 3.958 + 1.555 ln Pc) / (1.07 - Tbr)   (Pc in bar),
    then the Watson relation dH(T) = dHvb ((1-Tr)/(1-Tbr))^0.38.  Zero at and
    above the critical temperature.
    """
    Tb_K = np.asarray(Tb_K)
    Tbr = Tb_K / Tc_K
    dH_b = R_GAS * Tb_K * (3.978 * Tbr - 3.958 + 1.555 * np.log(Pc_bar)) / (1.07 - Tbr)
    one_minus_Tr = np.clip(1.0 - T_K / Tc_K, 0.0, None)
    return dH_b * (one_minus_Tr / (1.0 - Tbr)) ** 0.38


def vapour_viscosity_mpas(T_K: float, mw, Tc_K, Pc_bar, omega):
    """Low-pressure vapour viscosity [mPa.s] of each component (Chung et al., 1988).

        mu [uP] = 40.785 Fc sqrt(M T) / (Vc^(2/3) Omega_v)

    with Fc = 1 - 0.2756 omega (nonpolar, no association), the Neufeld
    collision integral Omega_v(T*), T* = 1.2593 T / Tc, and the critical volume
    Vc [cm3/mol] = Zc R Tc / Pc with Zc = 0.291 - 0.08 omega.  Pressure
    dependence is ignored, which is appropriate at vacuum.
    """
    mw = np.asarray(mw)
    omega = np.asarray(omega)
    Vc = (0.291 - 0.08 * omega) * 83.14 * np.asarray(Tc_K) / np.asarray(Pc_bar)   # cm3/mol
    T_star = 1.2593 * T_K / np.asarray(Tc_K)
    omega_v = (1.16145 * T_star ** -0.14874 + 0.52487 * np.exp(-0.77320 * T_star)
               + 2.16178 * np.exp(-2.43787 * T_star))
    mu_micropoise = 40.785 * (1.0 - 0.2756 * omega) * np.sqrt(mw * T_K) / (Vc ** (2.0 / 3.0) * omega_v)
    return mu_micropoise * 1.0e-4          # 1 uP = 1e-4 mPa.s


# --------------------------------------------------------------------------
# Mixture properties at a state point
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FluidProps:
    """Fluid properties at one temperature, pressure and composition."""

    T_c: float
    p_mbar: float
    rho_L: float          # liquid density [kg/m3]
    rho_V: float          # vapour density [kg/m3]
    mu_V: float           # vapour viscosity [mPa.s]
    mu_L: float           # liquid viscosity [mPa.s]
    sigma: float          # surface tension [mN/m]
    latent_kj_kg: float   # latent heat of the vapour mixture [kJ/kg]
    mw_vapour: float      # mean molecular weight of the vapour [g/mol]


def _normalise(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    if np.any(v < 0) or v.sum() <= 0:
        raise ValueError("Composition must be non-negative and not all zero.")
    return v / v.sum()


def mixture_properties(
    comps: Components,
    x_liq,
    T_c: float,
    p_mbar: float,
    y_vap=None,
) -> FluidProps:
    """All fluid properties at a state point.

    Parameters
    ----------
    comps  : Components (``Components.from_feed(feed)``).
    x_liq  : liquid mole fractions over *all* pseudo-components (zeros allowed);
             sets liquid density, viscosity and surface tension.
    T_c    : temperature [degC].
    p_mbar : absolute pressure [mbar]; sets the vapour density.
    y_vap  : vapour mole fractions; sets vapour density and latent heat.
             If omitted the vapour is taken to have the liquid's composition,
             which is a reasonable approximation for a narrow cut.  For a
             pot liquid pass the equilibrium vapour from ``vle.bubble_point``.
    """
    if p_mbar <= 0:
        raise ValueError("Pressure must be positive.")
    T = T_c + KELVIN_OFFSET
    x = _normalise(x_liq)
    y = x if y_vap is None else _normalise(y_vap)
    v = comps.vle

    # --- liquid side ------------------------------------------------------
    rho_i = liquid_density(T, comps.sg, v.Tc_K, v.omega)
    w = x * comps.mw / np.sum(x * comps.mw)           # mass fractions
    rho_L = 1.0 / np.sum(w / rho_i)                   # ideal volume additivity

    mu_i = liquid_viscosity_mpas(T, comps.mw, rho_i)
    mu_L = np.sum(x * mu_i ** (1.0 / 3.0)) ** 3       # Kendall-Monroe

    sigma = float(np.sum(x * surface_tension_mnm(T, v.Tb_K, v.Tc_K, v.Pc_bar)))

    # --- vapour side ------------------------------------------------------
    M_v = float(np.sum(y * comps.mw))                 # g/mol
    rho_V = (p_mbar * 100.0) * (M_v / 1000.0) / (R_GAS * T)   # ideal gas

    # Herning-Zipperer mixing of the pure-vapour viscosities
    mu_Vi = vapour_viscosity_mpas(T, comps.mw, v.Tc_K, v.Pc_bar, v.omega)
    sqrt_m = np.sqrt(comps.mw)
    mu_V = float(np.sum(y * mu_Vi * sqrt_m) / np.sum(y * sqrt_m))

    dH_mol = float(np.sum(y * latent_heat_jmol(T, v.Tb_K, v.Tc_K, v.Pc_bar)))   # J/mol
    latent = dH_mol / M_v                             # J/g = kJ/kg

    return FluidProps(T_c=T_c, p_mbar=p_mbar, rho_L=float(rho_L), rho_V=float(rho_V), mu_V=mu_V,
                      mu_L=float(mu_L), sigma=sigma, latent_kj_kg=float(latent),
                      mw_vapour=M_v)


# --------------------------------------------------------------------------
# Demo / smoke test:  python -m fdc.properties.fluids
# --------------------------------------------------------------------------

if __name__ == "__main__":
    from .feed import DEFAULT_WPPO_TBP, TBPCurve, characterise_feed
    from .vle import analyse_cut

    feed = characterise_feed(TBPCurve.from_points(DEFAULT_WPPO_TBP), n_cuts=20, sg_feed=0.80)
    comps = Components.from_feed(feed)

    # State 1: the 150-250 C cut as reflux liquid / vapour at the head (20 mbar)
    p = 20.0
    cut = analyse_cut(feed, 150.0, 250.0, p)
    _, x_cut = feed.cut_composition(150.0, 250.0)
    print(f"Cut 150-250 C at {p:.0f} mbar, head T = {cut.T_mid_c:.1f} C")
    fp = mixture_properties(comps, x_cut, cut.T_mid_c, p)
    print(f"  rho_L {fp.rho_L:7.1f} kg/m3 | rho_V {fp.rho_V:.4f} kg/m3 | mu_V {fp.mu_V * 1000:.2f} uPa.s | mu_L {fp.mu_L:.3f} mPa.s | "
          f"sigma {fp.sigma:.1f} mN/m | latent {fp.latent_kj_kg:.0f} kJ/kg | M_v {fp.mw_vapour:.0f} g/mol")

    # State 2: whole feed as pot liquid at 20 C (a viscosity / density sanity check)
    fp2 = mixture_properties(comps, feed.x_mol, 20.0, 1013.25)
    print(f"\nWhole feed at 20 C: rho_L {fp2.rho_L:.1f} kg/m3, mu_L {fp2.mu_L:.2f} mPa.s, "
          f"sigma {fp2.sigma:.1f} mN/m")
