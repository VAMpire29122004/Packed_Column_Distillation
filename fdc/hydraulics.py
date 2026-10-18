"""
fdc.hydraulics
==============

Vapour load, pressure drop across the packed bed and the margin to flooding.

Corr-map position
-----------------
Drivers : V06 / V07 (heater power and diameter, through the vapour flow),
          V12 (fluid properties), V14 / V08 (packing geometry),
          V09 (H_pack), V18 (liquid hold-up), V05 (R, through the liquid load)
Solves  : V16 (vapour velocity, F-factor, % flooding) and V17 (pressure drop).
          The pot pressure P_top + dP then feeds back to the pot temperature
          (the 'F' link into V10 in the corr map).

Model
-----
1. Vapour load:  u_V = m_V / (rho_V A),  F = u_V sqrt(rho_V)  [Pa^0.5].
2. Dry bed:  Ergun equation,
       dP/H = c1 (1-eps)^2 mu_V u / (eps^3 d^2) + c2 (1-eps) rho_V u^2 / (eps^3 d)
   with c1 = 150, c2 = 1.75 for granular beds (d = sphere-equivalent diameter).
3. Irrigated bed:  Stichlmair-type loading model.  Liquid occupies part of the
   void space, so the bed behaves like one with a smaller void fraction
   eps_e = eps - h and larger effective particles
   d_e = d ((1 - eps_e) / (1 - eps))^(1/3).  The hold-up itself grows with the
   pressure drop,
       h = h0 (1 + 20 (dP/H / (rho_L g))^2),
   so dP/H is the fixed point of the Ergun equation evaluated with that
   hold-up.  This reproduces the steepening pressure drop of the loading
   region.
4. Flooding:  the fixed point ceases to exist (the bed cannot carry that vapour
   rate at that liquid load).  The flooding vapour rate is found by raising
   the boil-up at constant reflux fraction until no solution remains;
   % flooding = actual vapour rate / flooding vapour rate.

Conventions
-----------
SI throughout inside; results also give pressure drop in mbar and the pot
pressure, matching the pressure unit used elsewhere in the project.

Main references
---------------
* Ergun (1952), Chem. Eng. Prog. 48(2), 89-94.
* Stichlmair, Bravo & Fair (1989), Gas Sep. Purif. 3, 19-28 (loading and
  flooding of packed columns; the effective-voidage and hold-up relations).

Known limitations
-----------------
* Ergun constants are for granular beds.  Steel wool is a very different
  structure; use ``HydraulicsModel`` to fit c1, c2 to a measured dry pressure
  drop.  Expect the wool numbers to be indicative only.
* The loading coefficient (20) comes from industrial packings.
* Constant L/V is assumed when searching for the flooding point.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import brentq

from .holdup import G, HoldupModel, HoldupResult, InternalFlows, liquid_holdup
from .packing import ColumnGeometry
from .properties.fluids import FluidProps

# --------------------------------------------------------------------------
# Constants and defaults
# --------------------------------------------------------------------------

PERCENT_FLOOD_WARN = 85.0       # above this the column is run uncomfortably close to flooding
MBAR_PER_PA = 0.01
_H_CAP = 0.98                   # hold-up is capped at this fraction of the void volume


@dataclass(frozen=True)
class HydraulicsModel:
    """Pressure-drop model constants, adjustable for calibration.

    c_viscous, c_inertial : Ergun constants (150, 1.75 for granular beds).
    loading_coeff         : coefficient in h = h0 (1 + coeff (dP/H / rho_L g)^2).
    """

    c_viscous: float = 150.0
    c_inertial: float = 1.75
    loading_coeff: float = 20.0

    def __post_init__(self):
        if min(self.c_viscous, self.c_inertial, self.loading_coeff) <= 0:
            raise ValueError("Model constants must be positive.")


# --------------------------------------------------------------------------
# Pressure drop
# --------------------------------------------------------------------------

def _ergun(eps: float, d: float, u: float, rho_V: float, mu_V: float, m: HydraulicsModel) -> float:
    """Ergun pressure gradient [Pa/m] for voidage ``eps``, particle size ``d``,
    superficial velocity ``u``."""
    return (m.c_viscous * (1.0 - eps) ** 2 * mu_V * u / (eps ** 3 * d ** 2)
            + m.c_inertial * (1.0 - eps) * rho_V * u ** 2 / (eps ** 3 * d))


def dry_pressure_gradient(geom: ColumnGeometry, fluid: FluidProps, u_V: float,
                          model: HydraulicsModel = HydraulicsModel()) -> float:
    """Pressure gradient [Pa/m] through the dry bed at superficial velocity ``u_V``."""
    return _ergun(geom.eps, geom.d_equiv_m, u_V, fluid.rho_V, fluid.mu_V * 1e-3, model)


def _wet_gradient(x: float, h0: float, geom: ColumnGeometry, fluid: FluidProps,
                  u_V: float, model: HydraulicsModel) -> float:
    """Ergun gradient of the irrigated bed if the pressure gradient were ``x``.

    The assumed gradient sets the hold-up (loading relation); the hold-up sets
    the effective voidage and particle size; Ergun then gives the gradient.
    """
    eps = geom.eps
    h = min(h0 * (1.0 + model.loading_coeff * (x / (fluid.rho_L * G)) ** 2), _H_CAP * eps)
    eps_e = eps - h
    d_e = geom.d_equiv_m * ((1.0 - eps_e) / (1.0 - eps)) ** (1.0 / 3.0)
    return _ergun(eps_e, d_e, u_V, fluid.rho_V, fluid.mu_V * 1e-3, model)


def irrigated_pressure_gradient(h0: float, geom: ColumnGeometry, fluid: FluidProps,
                                u_V: float, model: HydraulicsModel = HydraulicsModel()):
    """Pressure gradient [Pa/m] of the irrigated bed, or ``None`` if flooded.

    Solves x = _wet_gradient(x).  G(x) - x starts positive (a wet bed always
    drops more than a dry one); a root exists only if it dips below zero
    somewhere, which a log-spaced scan detects.  No dip means no stable
    operating point: the bed is flooded.
    """
    if u_V <= 0.0:
        return 0.0

    def gap(x):
        return _wet_gradient(x, h0, geom, fluid, u_V, model) - x

    grid = np.geomspace(1.0e-2, 20.0 * fluid.rho_L * G, 400)
    vals = np.array([gap(x) for x in grid])
    below = np.nonzero(vals < 0.0)[0]
    if below.size == 0:
        return None                                   # no operating point: flooded
    i = below[0]
    lo = grid[i - 1] if i > 0 else 1.0e-9
    return float(brentq(gap, lo, grid[i], xtol=1e-9, rtol=1e-10))


# --------------------------------------------------------------------------
# Flooding search
# --------------------------------------------------------------------------

def _state_at_vapour_rate(m_V: float, reflux_fraction: float, geom: ColumnGeometry,
                          fluid: FluidProps, hold_model: HoldupModel, model: HydraulicsModel):
    """Hold-up and irrigated gradient when the vapour flow is ``m_V`` [kg/s]
    and the reflux fraction L/V is held constant."""
    flows = InternalFlows(m_V, m_V * reflux_fraction, m_V * (1.0 - reflux_fraction))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")               # hold-up's own flooding warning
        hold = liquid_holdup(geom, fluid, flows, hold_model)
    u_V = m_V / (fluid.rho_V * geom.area_m2)
    return hold, u_V, irrigated_pressure_gradient(hold.h_total, geom, fluid, u_V, model)


def flooding_vapour_rate(reflux_fraction: float, geom: ColumnGeometry, fluid: FluidProps,
                         hold_model: HoldupModel = HoldupModel(),
                         model: HydraulicsModel = HydraulicsModel()) -> float:
    """Vapour flow [kg/s] at which the bed floods, at constant L/V.

    Brackets the flooding point by doubling, then bisects.
    """
    def flooded(m_V):
        return _state_at_vapour_rate(m_V, reflux_fraction, geom, fluid, hold_model, model)[2] is None

    lo = 1.0e-9
    hi = 1.0e-6
    while not flooded(hi):
        lo, hi = hi, hi * 2.0
        if hi > 1.0e3:
            raise RuntimeError("No flooding point found below 1000 kg/s; check the inputs.")
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if flooded(mid):
            hi = mid
        else:
            lo = mid
    return lo


# --------------------------------------------------------------------------
# Result and entry point
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class HydraulicsResult:
    """Vapour load, pressure drop and flooding margin at one operating point."""

    u_V_m_s: float              # superficial vapour velocity [m/s]
    F_factor: float             # u_V sqrt(rho_V) [Pa^0.5]
    dp_dry_pa_m: float          # dry-bed gradient [Pa/m]
    dp_pa_m: float              # irrigated gradient [Pa/m] (nan if flooded)
    dp_total_pa: float          # across the packed height [Pa] (nan if flooded)
    dp_total_mbar: float        # same, in mbar (nan if flooded)
    p_pot_mbar: float           # pot pressure P_top + dP [mbar] (nan if flooded)
    holdup_loaded: float        # hold-up including loading by the vapour [m3/m3 bed] (nan if flooded)
    flooded: bool
    m_flood_kg_s: float         # flooding vapour flow at this L/V [kg/s]
    percent_flood: float        # actual / flooding vapour flow x 100
    F_flood: float              # F-factor at flooding [Pa^0.5]


def column_hydraulics(
    geom: ColumnGeometry,
    fluid: FluidProps,
    flows: InternalFlows,
    hold_model: HoldupModel = HoldupModel(),
    model: HydraulicsModel = HydraulicsModel(),
) -> HydraulicsResult:
    """Vapour velocity, F-factor, pressure drop, pot pressure and % flooding.

    Parameters
    ----------
    geom       : ``packing.column_geometry(...)`` result.
    fluid      : ``properties.fluids.mixture_properties(...)`` at column
                 conditions (needs rho_V, mu_V, rho_L, mu_L, sigma).
    flows      : ``holdup.internal_flows(...)`` (sets V and the L/V split).
    hold_model, model : calibration parameters for hold-up and pressure drop.
    """
    if flows.m_vapour <= 0.0:
        raise ValueError("No vapour flow: heater power must be positive.")
    phi = flows.m_reflux / flows.m_vapour                 # reflux fraction L/V

    hold, u_V, dpdz = _state_at_vapour_rate(flows.m_vapour, phi, geom, fluid, hold_model, model)
    m_flood = flooding_vapour_rate(phi, geom, fluid, hold_model, model)
    pct = 100.0 * flows.m_vapour / m_flood
    F = u_V * math.sqrt(fluid.rho_V)
    F_flood = (m_flood / (fluid.rho_V * geom.area_m2)) * math.sqrt(fluid.rho_V)

    nan = float("nan")
    flooded = dpdz is None
    if flooded:
        warnings.warn("The column is flooded at this heater power and reflux: no stable "
                      "operating point. Lower the power, use larger packing or a wider column.",
                      stacklevel=2)
    elif pct > PERCENT_FLOOD_WARN:
        warnings.warn(f"Operating at {pct:.0f}% of flooding: expect unstable operation.", stacklevel=2)

    dp_total = nan if flooded else dpdz * geom.H_pack_m
    # Hold-up at this vapour load: the base hold-up grown by the loading relation
    # (the same relation that was used to solve for the pressure gradient).
    h_loaded = nan if flooded else min(
        hold.h_total * (1.0 + model.loading_coeff * (dpdz / (fluid.rho_L * G)) ** 2),
        _H_CAP * geom.eps)
    return HydraulicsResult(
        u_V_m_s=u_V, F_factor=F,
        dp_dry_pa_m=dry_pressure_gradient(geom, fluid, u_V, model),
        dp_pa_m=nan if flooded else dpdz,
        dp_total_pa=dp_total, dp_total_mbar=nan if flooded else dp_total * MBAR_PER_PA,
        p_pot_mbar=nan if flooded else fluid.p_mbar + dp_total * MBAR_PER_PA,
        holdup_loaded=h_loaded,
        flooded=flooded, m_flood_kg_s=m_flood, percent_flood=pct, F_flood=F_flood,
    )


# --------------------------------------------------------------------------
# Demo / smoke test:  python -m fdc.hydraulics
# --------------------------------------------------------------------------

if __name__ == "__main__":
    from .holdup import internal_flows
    from .packing import PackingSpec, column_geometry
    from .properties.feed import DEFAULT_WPPO_TBP, TBPCurve, characterise_feed
    from .properties.fluids import Components, mixture_properties
    from .properties.vle import analyse_cut

    feed = characterise_feed(TBPCurve.from_points(DEFAULT_WPPO_TBP), n_cuts=20, sg_feed=0.80)
    comps = Components.from_feed(feed)
    p_mbar, T_lo, T_hi, R = 20.0, 150.0, 250.0, 5.0

    cut = analyse_cut(feed, T_lo, T_hi, p_mbar)
    _, x_cut = feed.cut_composition(T_lo, T_hi)
    fluid = mixture_properties(comps, x_cut, cut.T_mid_c, p_mbar)
    print(f"Cut {T_lo:.0f}-{T_hi:.0f} C at P_top = {p_mbar:.0f} mbar, R = {R:.0f}, 40 mm column, 0.30 m bed\n")

    for label, spec in (("glass beads 3 mm", PackingSpec("glass_beads", 3.0)),
                        ("glass beads 5 mm", PackingSpec("glass_beads", 5.0))):
        g = column_geometry(spec, D_mm=40.0, H_pack_m=0.30)
        print(f"{label}:")
        print(f"  {'Q/W':>5} {'u_V m/s':>8} {'F':>5} {'dP mbar':>8} {'P_pot mbar':>10} {'% flood':>8}")
        for q in (5, 10, 20, 40, 60):
            fl = internal_flows(q, fluid.latent_kj_kg, R)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                h = column_hydraulics(g, fluid, fl)
            tag = "  FLOODED" if h.flooded else ""
            print(f"  {q:5.0f} {h.u_V_m_s:8.2f} {h.F_factor:5.2f} {h.dp_total_mbar:8.2f} "
                  f"{h.p_pot_mbar:10.2f} {h.percent_flood:8.0f}{tag}")
        print(f"  flooding at F = {h.F_flood:.2f} Pa^0.5\n")
