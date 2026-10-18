"""
fdc.hetp
========

Height equivalent to a theoretical plate (HETP) of the packed bed, from a
mass-transfer (film theory) model, plus the helpers that connect HETP to the
stage count: H_pack = N_pack x HETP.

Corr-map position
-----------------
Drivers : V05 (reflux, through the liquid and vapour loads), V07 (D),
          V08 / V14 (packing and its geometry), V12 (fluid properties),
          V16 (vapour load), V13 (relative volatility, for the equilibrium slope)
Solves  : V20 (HETP).  This is the *least certain* link in the whole model
          ('M' in the corr map): treat the absolute value as an estimate and
          calibrate ``HETPModel.k_hetp`` against one measured separation on
          your own column.  The trends (load, packing size, reflux) are more
          trustworthy than the absolute level.

Model
-----
Film theory with Onda's (1968) correlations for random packings:
* wetted area fraction  a_w / a,
* liquid-side coefficient k_L and gas-side coefficient k_G (both in m/s),
* transfer-unit heights  H_G = u_V / (k_G a_w),  H_L = u_L / (k_L a_w),
* overall height  H_OG = H_G + lambda H_L,  lambda = m V / L (m = slope of the
  equilibrium line, V/L from the internal flows),
* HETP = H_OG ln(lambda) / (lambda - 1)   (-> H_OG as lambda -> 1).
Diffusivities: Fuller-Schettler-Giddings (vapour, treated as diffusion between
like molecules) and Wilke-Chang (liquid).  The equilibrium slope uses the
relative volatility of the key pair, m = alpha / (1 + (alpha - 1) x)^2, with x
a representative light-component mole fraction.

Conventions
-----------
SI throughout (m, kg, s, W); diffusivities in m2/s.  The wetted-area fraction
returned here can be passed to ``holdup.HoldupModel(wetted_fraction=...)`` to
make the hold-up estimate consistent with the mass-transfer model.

Main references
---------------
* Onda, Takeuchi & Okumoto (1968), J. Chem. Eng. Japan 1, 56-62.
* Treybal, "Mass-Transfer Operations", 3rd ed. (HETP from transfer units).
* Poling, Prausnitz & O'Connell, "The Properties of Gases and Liquids", 5th ed.
  (Fuller and Wilke-Chang methods).

Known limitations
-----------------
* Onda's correlations were fitted to packings of about 6-50 mm with
  water-like liquids and gases at normal pressure.  The 3-5 mm beads of a lab
  column and any steel wool are extrapolations; a warning says so.
* No wall-channelling or flooding effect is included: HETP is likely to rise
  sharply near flooding and in columns with D/d_p below about 8.
* Liquid viscosity and diffusivity are for the average cut, not each
  component.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

from .holdup import G, InternalFlows
from .packing import ColumnGeometry
from .properties.fluids import FluidProps

# --------------------------------------------------------------------------
# Constants and defaults
# --------------------------------------------------------------------------

# Critical surface tension of the packing material [N/m] (Onda).
SIGMA_C = {"glass_beads": 0.073, "steel_wool": 0.075}

ONDA_VALID_MIN_SIZE_M = 0.006           # Onda's fit covers roughly 6-50 mm packings
ONDA_C_GAS_SMALL = 2.00                 # gas-side constant for packings below 15 mm
ONDA_C_GAS_LARGE = 5.23                 # ... and for 15 mm and above
ONDA_SIZE_SWITCH_M = 0.015

# Plausible HETP / (sphere-equivalent diameter) for small random packings;
# outside this range the model is probably being pushed too far.
RATIO_WARN_LOW, RATIO_WARN_HIGH = 3.0, 80.0

MW_H2, MW_CH2 = 2.016, 14.027


@dataclass(frozen=True)
class HETPModel:
    """Calibration parameters.

    k_hetp          : multiplier on the final HETP; set it from one measured
                      separation (1.0 = Onda model as is).
    x_light         : representative light-component mole fraction used for the
                      equilibrium slope (ASSUMPTION: 0.5).
    wetted_fraction : if given, overrides Onda's a_w / a (for example a value
                      measured by a dye or drain test).
    """

    k_hetp: float = 1.0
    x_light: float = 0.5
    wetted_fraction: float | None = None

    def __post_init__(self):
        if self.k_hetp <= 0:
            raise ValueError("k_hetp must be positive.")
        if not 0.0 < self.x_light < 1.0:
            raise ValueError("x_light must lie strictly between 0 and 1.")
        if self.wetted_fraction is not None and not 0.0 < self.wetted_fraction <= 1.0:
            raise ValueError("wetted_fraction must be in (0, 1].")


# --------------------------------------------------------------------------
# Diffusivities
# --------------------------------------------------------------------------

def vapour_diffusivity_m2s(T_K: float, p_bar: float, mw: float) -> float:
    """Vapour-phase diffusivity [m2/s] (Fuller, Schettler & Giddings).

    D = 0.00143 T^1.75 / (P sqrt(M_AB) [v_A^(1/3) + v_B^(1/3)]^2)  [cm2/s],
    P in bar.  Diffusion is taken between like molecules (M_AB = M, v_A = v_B)
    of carbon number N = (M - 2.016)/14.027, with diffusion volume
    v = 15.9 N + 2.31 (2N + 2) from the atomic increments.
    """
    N = (mw - MW_H2) / MW_CH2
    sum_v = 15.9 * N + 2.31 * (2.0 * N + 2.0)
    D_cm2 = 0.00143 * T_K ** 1.75 / (p_bar * math.sqrt(mw) * (2.0 * sum_v ** (1.0 / 3.0)) ** 2)
    return D_cm2 * 1.0e-4


def liquid_diffusivity_m2s(T_K: float, mu_mpas: float, mw: float, rho_kg_m3: float) -> float:
    """Liquid-phase diffusivity [m2/s] (Wilke-Chang, unassociated solvent).

    D = 7.4e-8 sqrt(phi M_B) T / (mu V_A^0.6)  [cm2/s], mu in cP, phi = 1.
    The molar volume is taken as M / rho at the operating temperature (the
    method asks for the volume at the normal boiling point; this is the usual
    shortcut for a narrow cut).
    """
    V = mw / (rho_kg_m3 / 1000.0)                      # cm3/mol
    D_cm2 = 7.4e-8 * math.sqrt(mw) * T_K / (mu_mpas * V ** 0.6)
    return D_cm2 * 1.0e-4


# --------------------------------------------------------------------------
# Onda correlations
# --------------------------------------------------------------------------

def wetted_area_fraction(geom: ColumnGeometry, fluid: FluidProps, L_w: float,
                         sigma_c: float) -> float:
    """Onda wetted-area fraction a_w / a_t for liquid mass flux ``L_w`` [kg/m2 s]."""
    if L_w <= 0.0:
        return 0.0
    a = geom.a_m2_m3
    mu = fluid.mu_L * 1.0e-3                           # Pa.s
    sigma = fluid.sigma * 1.0e-3                       # N/m
    Re = L_w / (a * mu)
    Fr = a * L_w ** 2 / (fluid.rho_L ** 2 * G)
    We = L_w ** 2 / (fluid.rho_L * sigma * a)
    expo = 1.45 * (sigma_c / sigma) ** 0.75 * Re ** 0.1 * Fr ** -0.05 * We ** 0.2
    return min(1.0, 1.0 - math.exp(-expo))


def liquid_film_coefficient(geom: ColumnGeometry, fluid: FluidProps, L_w: float,
                            a_w: float, D_L: float) -> float:
    """Onda liquid-side coefficient k_L [m/s]; ``a_w`` is the wetted area [m2/m3]."""
    mu = fluid.mu_L * 1.0e-3
    Sc = mu / (fluid.rho_L * D_L)
    return (0.0051 * (L_w / (a_w * mu)) ** (2.0 / 3.0) * Sc ** -0.5
            * (geom.a_m2_m3 * geom.d_equiv_m) ** 0.4 * (mu * G / fluid.rho_L) ** (1.0 / 3.0))


def gas_film_coefficient(geom: ColumnGeometry, fluid: FluidProps, G_w: float, D_G: float) -> float:
    """Onda gas-side coefficient k_G R T, i.e. a velocity [m/s].

    k_G R T / (a D_G) = C (G'/(a mu_G))^0.7 Sc_G^(1/3) (a d)^-2,
    C = 2.0 for packings below 15 mm, 5.23 otherwise.
    """
    a, d = geom.a_m2_m3, geom.d_equiv_m
    mu = fluid.mu_V * 1.0e-3
    Sc = mu / (fluid.rho_V * D_G)
    C = ONDA_C_GAS_LARGE if d >= ONDA_SIZE_SWITCH_M else ONDA_C_GAS_SMALL
    return C * a * D_G * (G_w / (a * mu)) ** 0.7 * Sc ** (1.0 / 3.0) * (a * d) ** -2.0


# --------------------------------------------------------------------------
# Result and entry point
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class HETPResult:
    """HETP and the quantities behind it."""

    hetp_m: float            # height equivalent to a theoretical plate [m] (after k_hetp)
    H_G_m: float             # gas-phase transfer-unit height [m]
    H_L_m: float             # liquid-phase transfer-unit height [m]
    H_OG_m: float            # overall height of a gas-phase transfer unit [m]
    lam: float               # lambda = m V / L (stripping factor)
    liquid_resistance: float # lambda H_L / H_OG: share of resistance in the liquid film
    wetted_fraction: float   # a_w / a_t
    k_L_m_s: float
    k_G_m_s: float           # k_G R T, as a velocity
    D_G_m2_s: float
    D_L_m2_s: float
    hetp_over_d: float       # HETP / sphere-equivalent diameter [-] (meaningful for beads only)


def hetp(
    geom: ColumnGeometry,
    fluid: FluidProps,
    flows: InternalFlows,
    alpha: float,
    model: HETPModel = HETPModel(),
) -> HETPResult:
    """HETP of the packed bed at one operating point.

    Parameters
    ----------
    geom  : ``packing.column_geometry(...)`` result.
    fluid : ``properties.fluids.mixture_properties(...)`` at column conditions
            (needs rho_L, rho_V, mu_L, mu_V, sigma, mw_vapour, T_c, p_mbar).
    flows : ``holdup.internal_flows(...)`` (vapour and reflux).
    alpha : relative volatility of the key pair (for example
            ``stages.BoundaryStages.alpha`` of the governing boundary).
    model : calibration parameters.
    """
    if alpha <= 1.0:
        raise ValueError("Relative volatility must exceed 1.")
    if flows.m_vapour <= 0.0:
        raise ValueError("No vapour flow: heater power must be positive.")
    if flows.m_reflux <= 0.0:
        raise ValueError("No reflux (R = 0): the column does no rectification, HETP is undefined.")

    if geom.d_equiv_m < ONDA_VALID_MIN_SIZE_M:
        warnings.warn(f"Packing size {geom.d_equiv_m * 1000:.2f} mm is below the ~6 mm range "
                      "Onda's correlations were fitted to: HETP is an extrapolation. "
                      "Calibrate k_hetp against a measured separation.", stacklevel=2)

    T_K = fluid.T_c + 273.15
    p_bar = fluid.p_mbar / 1000.0
    A = geom.area_m2
    G_w = flows.m_vapour / A                           # vapour mass flux [kg/m2 s]
    L_w = flows.m_reflux / A                           # liquid mass flux [kg/m2 s]
    u_V = G_w / fluid.rho_V
    u_L = L_w / fluid.rho_L

    D_G = vapour_diffusivity_m2s(T_K, p_bar, fluid.mw_vapour)
    D_L = liquid_diffusivity_m2s(T_K, fluid.mu_L, fluid.mw_vapour, fluid.rho_L)

    # wetted area: Onda unless overridden by a measured value
    if model.wetted_fraction is not None:
        wet = model.wetted_fraction
    else:
        wet = wetted_area_fraction(geom, fluid, L_w, SIGMA_C[geom.kind])
    a_w = wet * geom.a_m2_m3

    k_L = liquid_film_coefficient(geom, fluid, L_w, a_w, D_L)
    k_G = gas_film_coefficient(geom, fluid, G_w, D_G)
    H_G = u_V / (k_G * a_w)
    H_L = u_L / (k_L * a_w)

    # lambda = m V / L, with m the local slope of the equilibrium line
    x = model.x_light
    m_eq = alpha / (1.0 + (alpha - 1.0) * x) ** 2
    lam = m_eq * flows.m_vapour / flows.m_reflux
    H_OG = H_G + lam * H_L
    factor = 1.0 if abs(lam - 1.0) < 1e-6 else math.log(lam) / (lam - 1.0)
    h = model.k_hetp * H_OG * factor

    ratio = h / geom.d_equiv_m
    # The HETP / particle-size rule of thumb only makes sense for particles, not
    # for the fine wire of steel wool (whose equivalent diameter is tiny).
    if geom.kind == "glass_beads" and not RATIO_WARN_LOW <= ratio <= RATIO_WARN_HIGH:
        warnings.warn(f"HETP is {ratio:.0f} sphere-equivalent diameters: outside the usual "
                      f"{RATIO_WARN_LOW:.0f}-{RATIO_WARN_HIGH:.0f} range, so treat it with suspicion.",
                      stacklevel=2)

    return HETPResult(hetp_m=h, H_G_m=H_G, H_L_m=H_L, H_OG_m=H_OG, lam=lam,
                      liquid_resistance=lam * H_L / H_OG, wetted_fraction=wet,
                      k_L_m_s=k_L, k_G_m_s=k_G, D_G_m2_s=D_G, D_L_m2_s=D_L, hetp_over_d=ratio)


# --------------------------------------------------------------------------
# HETP <-> height conversions (the still pot counts as one stage, see stages.py)
# --------------------------------------------------------------------------

def packed_height_required(N_total: float, hetp_m: float) -> float:
    """Packed height [m] for ``N_total`` theoretical stages including the pot:
    H_pack = (N_total - 1) x HETP.  Design mode."""
    if hetp_m <= 0:
        raise ValueError("HETP must be positive.")
    return max(0.0, N_total - 1.0) * hetp_m


def stages_available(H_pack_m: float, hetp_m: float) -> float:
    """Total theoretical stages a bed of height ``H_pack_m`` provides, including
    the pot: N_total = H_pack / HETP + 1.  Rating mode."""
    if hetp_m <= 0:
        raise ValueError("HETP must be positive.")
    return H_pack_m / hetp_m + 1.0


# --------------------------------------------------------------------------
# Demo / smoke test:  python -m fdc.hetp
# --------------------------------------------------------------------------

if __name__ == "__main__":
    from .holdup import internal_flows
    from .packing import PackingSpec, column_geometry
    from .properties.feed import DEFAULT_WPPO_TBP, TBPCurve, characterise_feed
    from .properties.fluids import Components, mixture_properties
    from .properties.vle import analyse_cut
    from .stages import SeparationSpec, cut_stage_requirement

    feed = characterise_feed(TBPCurve.from_points(DEFAULT_WPPO_TBP), n_cuts=20, sg_feed=0.80)
    comps = Components.from_feed(feed)
    p_mbar, T_lo, T_hi, R = 20.0, 150.0, 250.0, 5.0

    cut = analyse_cut(feed, T_lo, T_hi, p_mbar)
    _, x_cut = feed.cut_composition(T_lo, T_hi)
    fluid = mixture_properties(comps, x_cut, cut.T_mid_c, p_mbar)
    req = cut_stage_requirement(feed, T_lo, T_hi, p_mbar, R, SeparationSpec())
    gov = max(req.boundaries, key=lambda b: b.N_total)
    print(f"Cut {T_lo:.0f}-{T_hi:.0f} C at {p_mbar:.0f} mbar, R = {R:.0f}: {req.N_total:.1f} stages needed "
          f"({req.N_packing:.1f} in the packing), governing alpha {gov.alpha:.2f}\n")

    flows = internal_flows(10.0, fluid.latent_kj_kg, R)            # 10 W heater
    for label, spec in (("glass beads 3 mm", PackingSpec("glass_beads", 3.0)),
                        ("glass beads 5 mm", PackingSpec("glass_beads", 5.0))):
        g = column_geometry(spec, D_mm=40.0, H_pack_m=0.30)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = hetp(g, fluid, flows, gov.alpha)
        print(f"{label}: HETP {r.hetp_m * 1000:.0f} mm ({r.hetp_over_d:.0f} d) | a_w/a {r.wetted_fraction:.2f} | "
              f"H_G {r.H_G_m * 1000:.0f} mm, H_L {r.H_L_m * 1000:.0f} mm, lambda {r.lam:.2f}, "
              f"liquid resistance {r.liquid_resistance:.0%}")
        print(f"   packed height for {req.N_packing:.1f} packing stages: "
              f"{packed_height_required(req.N_total, r.hetp_m):.2f} m; "
              f"a 0.30 m bed gives {stages_available(0.30, r.hetp_m):.1f} stages\n")
