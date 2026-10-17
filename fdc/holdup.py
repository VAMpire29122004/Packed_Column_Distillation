"""
fdc.holdup
==========

Internal liquid and vapour flows of the column, the liquid hold-up held on the
packing, and the liquid residence time.

Corr-map position
-----------------
Drivers : V12 (fluid properties), V08 / V14 (packing and its geometry),
          V05 (reflux ratio R), V06 (heater power Q), V07 / V09 (D, H_pack)
Solves  : V18 (liquid hold-up) and V22 (liquid residence time).
          ``internal_flows`` is also used by ``hydraulics.py`` for the vapour
          load.  Hold-up is an *empirical* link ('M' in the corr map): treat
          the numbers as estimates until checked against a drain-down
          measurement on your own column (see ``HoldupModel`` scale factors).

Model
-----
Total hold-up (m3 of liquid per m3 of packed bed) is the sum of
* a static part, the liquid retained by capillarity after draining:
  Saez & Carbonell (1985),  h_s = 1 / (20 + 0.9 Eo),
  Eo = rho g d^2 / sigma * eps^2 / (1 - eps)^2   (d = sphere-equivalent
  diameter of the packing);
* a dynamic (operating) part carried by the flowing film, from the
  film-flow expression used by Billet & Schultes (1999) below the loading point,
  h_d = (12 nu_L u_L a^2 / g)^(1/3) (a_h/a)^(2/3),
  with u_L the superficial liquid velocity, a the packing surface area and
  a_h/a the fraction of that area that is wetted.

Internal flows follow from the heat balance with a total condenser: a
fraction ``efficiency`` of the heater power evaporates liquid, V = eta Q /
latent heat, and the reflux ratio splits it into reflux L = V R/(R+1) and
distillate D = V/(R+1).

Conventions
-----------
SI throughout (kg, m, s, W).  ``h_total`` is the "liquid hold-up ratio" of the
original parameter list, as m3 liquid per m3 bed; ``saturation`` is the same
liquid expressed as a fraction of the void volume.

Main references
---------------
* Saez & Carbonell (1985), AIChE J. 31, 52-62 (static hold-up).
* Billet & Schultes (1999), Chem. Eng. Technol. 22, 1-14 (film hold-up).

Known limitations
-----------------
* Both correlations were developed for packings of millimetre scale and
  water-like liquids; steel wool is an extrapolation.
* The dynamic part assumes the film wets the whole packing area by default;
  at the very low liquid loads of a lab column wetting is usually partial, so
  lower ``wetted_fraction`` if drain-down measurements say so.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

from .packing import ColumnGeometry
from .properties.fluids import FluidProps

# --------------------------------------------------------------------------
# Constants and defaults
# --------------------------------------------------------------------------

G = 9.80665                     # m/s2

# ASSUMPTION: fraction of the heater power that ends up evaporating liquid
# (the rest heats the flask and surroundings and is lost).  Measure or tune.
EFFICIENCY_DEFAULT = 0.70

# A bed holding more than this fraction of its void volume as liquid is close
# to flooding for practical purposes.
SATURATION_WARN = 0.60


# --------------------------------------------------------------------------
# Internal flows
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class InternalFlows:
    """Mass flows [kg/s] in the column for a given power and reflux ratio."""

    m_vapour: float        # vapour rising through the column, V
    m_reflux: float        # liquid returning down the column, L
    m_distillate: float    # product taken off, D


def internal_flows(
    q_heater_w: float,
    latent_kj_kg: float,
    R: float,
    efficiency: float = EFFICIENCY_DEFAULT,
) -> InternalFlows:
    """Vapour, reflux and distillate flows from heater power and reflux ratio.

    Constant mass overflow is assumed (liquid and vapour flows are equal on
    every tray-equivalent), and the head condenser is total.  ``R = math.inf``
    means total reflux: nothing is taken off.
    """
    if q_heater_w < 0 or latent_kj_kg <= 0:
        raise ValueError("Power must be non-negative and latent heat positive.")
    if not 0.0 < efficiency <= 1.0:
        raise ValueError("Efficiency must be in (0, 1].")
    if R < 0:
        raise ValueError("Reflux ratio cannot be negative.")

    V = efficiency * q_heater_w / (latent_kj_kg * 1000.0)       # kg/s
    reflux_fraction = 1.0 if math.isinf(R) else R / (R + 1.0)
    return InternalFlows(m_vapour=V, m_reflux=V * reflux_fraction,
                         m_distillate=V * (1.0 - reflux_fraction))


# --------------------------------------------------------------------------
# Hold-up correlations
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class HoldupModel:
    """Tunable parameters, so the model can be calibrated to lab data.

    wetted_fraction : a_h / a, fraction of the packing area wetted by the film
                      (1.0 = fully wetted; ASSUMPTION).
    k_static        : multiplier on the static hold-up (1.0 = correlation).
    k_dynamic       : multiplier on the dynamic hold-up (1.0 = correlation).
    """

    wetted_fraction: float = 1.0
    k_static: float = 1.0
    k_dynamic: float = 1.0

    def __post_init__(self):
        if not 0.0 < self.wetted_fraction <= 1.0:
            raise ValueError("wetted_fraction must be in (0, 1].")
        if self.k_static < 0 or self.k_dynamic < 0:
            raise ValueError("Scale factors cannot be negative.")


def static_holdup(geom: ColumnGeometry, fluid: FluidProps) -> float:
    """Static hold-up [m3 liquid / m3 bed] (Saez & Carbonell, 1985).

    Uses the Eotvos number built on the sphere-equivalent diameter, so larger,
    heavier, lower-surface-tension liquids are retained more.
    """
    eps = geom.eps
    Eo = (fluid.rho_L * G * geom.d_equiv_m ** 2 / (fluid.sigma * 1.0e-3)) * eps ** 2 / (1.0 - eps) ** 2
    return 1.0 / (20.0 + 0.9 * Eo)


def dynamic_holdup(geom: ColumnGeometry, fluid: FluidProps, u_L_m_s: float,
                   wetted_fraction: float = 1.0) -> float:
    """Dynamic hold-up [m3 liquid / m3 bed] from film flow (Billet-Schultes form).

    h_d = (12 nu u_L a^2 / g)^(1/3) (a_h/a)^(2/3);  nu = mu / rho.  Zero when
    nothing flows.
    """
    if u_L_m_s <= 0.0:
        return 0.0
    nu = fluid.mu_L * 1.0e-3 / fluid.rho_L                    # kinematic viscosity [m2/s]
    return (12.0 * nu * u_L_m_s * geom.a_m2_m3 ** 2 / G) ** (1.0 / 3.0) * wetted_fraction ** (2.0 / 3.0)


# --------------------------------------------------------------------------
# Result and entry point
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class HoldupResult:
    """Liquid hold-up on the packing and the liquid residence time."""

    u_L_m_s: float          # superficial liquid velocity [m/s]
    h_static: float         # [m3 liquid / m3 bed]
    h_dynamic: float        # [m3 liquid / m3 bed]
    h_total: float          # liquid hold-up ratio, static + dynamic [m3/m3 bed]
    saturation: float       # h_total / void fraction [-]
    V_liquid_m3: float      # total liquid on the packing [m3]
    mass_kg: float          # total liquid on the packing [kg] (lost to recovery)
    tau_dynamic_s: float    # film transit time: dynamic volume / liquid flow [s]
    tau_total_s: float      # (static + dynamic) volume / liquid flow [s]


def liquid_holdup(
    geom: ColumnGeometry,
    fluid: FluidProps,
    flows: InternalFlows,
    model: HoldupModel = HoldupModel(),
) -> HoldupResult:
    """Hold-up, liquid on the packing and liquid residence time.

    Parameters
    ----------
    geom  : ``packing.column_geometry(...)`` result.
    fluid : ``properties.fluids.mixture_properties(...)`` at column conditions
            (liquid density, viscosity and surface tension are used).
    flows : ``internal_flows(...)`` result (reflux sets the liquid load).
    model : calibration parameters.
    """
    Q_L = flows.m_reflux / fluid.rho_L                      # liquid volumetric flow [m3/s]
    u_L = Q_L / geom.area_m2                                # superficial velocity [m/s]

    h_s = model.k_static * static_holdup(geom, fluid)
    h_d = model.k_dynamic * dynamic_holdup(geom, fluid, u_L, model.wetted_fraction)
    h_t = h_s + h_d
    saturation = h_t / geom.eps

    if saturation > SATURATION_WARN:
        warnings.warn(f"Liquid fills {saturation:.0%} of the void volume: the bed is at or near "
                      "flooding. Reduce heater power or reflux.", stacklevel=2)

    V_liq = h_t * geom.V_bed_m3
    # Residence times are undefined with no liquid flow (R = 0): report infinity.
    tau_d = h_d * geom.V_bed_m3 / Q_L if Q_L > 0 else math.inf
    tau_t = V_liq / Q_L if Q_L > 0 else math.inf

    return HoldupResult(u_L_m_s=u_L, h_static=h_s, h_dynamic=h_d, h_total=h_t,
                        saturation=saturation, V_liquid_m3=V_liq, mass_kg=V_liq * fluid.rho_L,
                        tau_dynamic_s=tau_d, tau_total_s=tau_t)


# --------------------------------------------------------------------------
# Demo / smoke test:  python -m fdc.holdup
# --------------------------------------------------------------------------

if __name__ == "__main__":
    from .packing import PackingSpec, column_geometry
    from .properties.feed import DEFAULT_WPPO_TBP, TBPCurve, characterise_feed
    from .properties.fluids import Components, mixture_properties
    from .properties.vle import analyse_cut

    feed = characterise_feed(TBPCurve.from_points(DEFAULT_WPPO_TBP), n_cuts=20, sg_feed=0.80)
    comps = Components.from_feed(feed)
    p_mbar, T_lo, T_hi = 20.0, 150.0, 250.0

    cut = analyse_cut(feed, T_lo, T_hi, p_mbar)
    _, x_cut = feed.cut_composition(T_lo, T_hi)
    fluid = mixture_properties(comps, x_cut, cut.T_mid_c, p_mbar)

    q_w, R = 60.0, 5.0
    flows = internal_flows(q_w, fluid.latent_kj_kg, R)
    print(f"Cut {T_lo:.0f}-{T_hi:.0f} C at {p_mbar:.0f} mbar (head {cut.T_mid_c:.0f} C), "
          f"heater {q_w:.0f} W, R = {R:.0f}")
    print(f"  vapour {flows.m_vapour * 3600:.2f} kg/h | reflux {flows.m_reflux * 3600:.2f} kg/h "
          f"| distillate {flows.m_distillate * 3600:.2f} kg/h\n")

    for label, spec in (("glass beads 3 mm", PackingSpec("glass_beads", 3.0)),
                        ("steel wool 0.05 mm", PackingSpec("steel_wool", 0.05))):
        g = column_geometry(spec, D_mm=30.0, H_pack_m=0.30)
        h = liquid_holdup(g, fluid, flows)
        print(f"{label}: u_L {h.u_L_m_s * 1000:.2f} mm/s | static {h.h_static:.3f} + dynamic {h.h_dynamic:.3f} "
              f"= {h.h_total:.3f} m3/m3 bed ({h.saturation:.0%} of voids)")
        print(f"  liquid on packing {h.V_liquid_m3 * 1e6:.1f} mL = {h.mass_kg * 1000:.1f} g | "
              f"residence time {h.tau_dynamic_s:.0f} s (film) / {h.tau_total_s:.0f} s (total)\n")
