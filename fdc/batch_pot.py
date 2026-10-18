"""
fdc.batch_pot
=============

Time-stepping simulation of the batch run: the flask (pot) is heated, the
packed column fractionates the vapour, and distillate is switched between
receivers as the head temperature crosses the cut boundaries.

Corr-map position
-----------------
Drivers : V01 (charge), V02 (feed curve), V03 (cut edges), V04 (P_top),
          V05 (R), V06 (heater power), V09 / V20 (H_pack / HETP -> stages),
          V17 (pressure drop), V18 (hold-up)
Solves  : V10 (pot temperature history), V23 (thermal residence time) and
          V24 (cut masses, purity, recovery, batch time).  This is the module
          that brings every earlier one together.

Model
-----
* Pot: the charge is a set of pseudo-components (``feed.py``).  Its bubble
  point at the pot pressure P_top + dP gives T_pot.
* Column: quasi-steady, ideal-stage model.  The packed bed provides
  N = H_pack / HETP + 1 theoretical stages (the pot is one).  Constant molar
  overflow, total condenser, Raoult's law with vapour pressures at the mean of
  the pot and head temperatures.  For a given pot composition and reflux ratio
  the distillate composition is found stage by stage from the top down, with a
  multiplicative correction of the distillate composition until the computed
  pot composition matches the real one.  Fractional stage counts blend the
  results for the two neighbouring integers.
* Rates: the heater power, times the efficiency, boils V = eta Q / latent heat;
  the distillate rate is V / (R + 1).  The run is advanced in small distillate
  increments, so every step has similar resolution regardless of how fast the
  composition changes.
* Hydraulics: each step the pressure drop (``hydraulics.py``) sets the pot
  pressure, which feeds back on T_pot.  Flooding stops the run.
* Cuts: the receiver is switched when the head temperature reaches the cut
  edge converted to P_top (``vle.boiling_point_at_pressure``), exactly as an
  operator would watch TT-102.  The run ends when the last edge is reached.
* Hold-up: the liquid on the packing (``holdup.py``) is taken out of the pot
  at the start (at feed composition) and returned to the residue at the end.
  Its dynamic lag in composition is not modelled.
* Thermal exposure: equivalent minutes at a reference temperature using an
  Arrhenius weighting, plus the time spent above a threshold.

Metrics
-------
Cut purity is the mass fraction of the collected material that lies inside the
cut's own atmospheric-equivalent range; recovery is the in-range mass collected
over the in-range mass available in the charge.  Each pseudo-component counts
as spread uniformly over its slice of the TBP curve, so n_cuts of 30 or more
reduces the discretisation noise.

Conventions
-----------
Cut edges are the *upper* atmospheric-equivalent limits of consecutive cuts,
ascending; cut 0 starts at the IBP.  Whatever lies above the last edge stays in
the pot as residue.  Times in minutes, masses in kg, temperatures in degC.

Known limitations
-----------------
* Quasi-steady column: no start-up transient, no hold-up dynamics.
* Constant relative volatility within a step; constant R during the run.
* Mass and moles are tied by the pseudo-component molecular weights, so small
  inconsistencies of a few percent in M carry through to masses.
* ASSUMPTIONS: thermal model constants (``ThermalModel``) and the heat-up cp
  are typical values, not WPPO measurements.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np

from .holdup import HoldupModel, InternalFlows, internal_flows, liquid_holdup
from .hydraulics import HydraulicsModel, irrigated_pressure_gradient
from .hetp import HETPModel, hetp as hetp_model, stages_available
from .packing import ColumnGeometry
from .properties.feed import FeedCharacterisation, KELVIN_OFFSET
from .properties.fluids import Components, mixture_properties
from .properties.vle import (MBAR_PER_BAR, VLEProps, analyse_cut,
                             boiling_point_at_pressure)

R_GAS = 8.314462618            # J/(mol K)
PA_TO_MBAR = 0.01


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class BatchSpec:
    """Operating inputs of the run.

    m_feed_kg   : charge to the pot [kg].
    p_top_mbar  : head (top) pressure [mbar abs].
    R           : constant reflux ratio (finite).
    q_heater_w  : heater power [W].
    cut_edges_c : ascending upper limits of consecutive cuts, atmospheric-equivalent
                  degC; the run stops when the head reaches the last edge.
    efficiency  : fraction of heater power that evaporates liquid (ASSUMPTION 0.70).
    target      : index of the cut whose purity/recovery you care about
                  (default -1 = the last cut).
    T_condenser_c: coolant temperature of the condensers [degC] (ASSUMPTION 15);
                  vapour that must condense below this temperature escapes to the
                  cold trap, so the run is flagged when the head is that cold.
    """

    m_feed_kg: float
    p_top_mbar: float
    R: float
    q_heater_w: float
    cut_edges_c: tuple[float, ...]
    efficiency: float = 0.70
    target: int = -1
    T_condenser_c: float = 15.0

    def __post_init__(self):
        if self.m_feed_kg <= 0 or self.p_top_mbar <= 0 or self.q_heater_w <= 0:
            raise ValueError("Charge, pressure and heater power must be positive.")
        if not self.R >= 0 or math.isinf(self.R):
            raise ValueError("R must be finite and non-negative (total reflux collects nothing).")
        if not 0.0 < self.efficiency <= 1.0:
            raise ValueError("Efficiency must be in (0, 1].")
        edges = list(self.cut_edges_c)
        if len(edges) < 1 or any(b <= a for a, b in zip(edges, edges[1:])):
            raise ValueError("cut_edges_c must be a non-empty ascending sequence.")
        if not -len(edges) <= self.target < len(edges):
            raise ValueError("target must index one of the cuts.")


@dataclass(frozen=True)
class ThermalModel:
    """Pot thermal-exposure and heat-up constants (all ASSUMPTIONS, not measured).

    Ea_kj_mol  : activation energy of thermal degradation for the weighting.
    T_ref_c    : reference temperature for 'equivalent minutes'.
    T_limit_c  : pot temperature at which the run is stopped and flagged.
    cp_kj_kgk  : liquid heat capacity for the heat-up time.
    T_ambient_c: starting temperature of the charge.
    """

    Ea_kj_mol: float = 200.0
    T_ref_c: float = 300.0
    T_limit_c: float = 300.0
    cp_kj_kgk: float = 2.2
    T_ambient_c: float = 25.0


@dataclass(frozen=True)
class SimSettings:
    """Numerical settings.

    step_fraction       : distillate removed per step, as a fraction of the charge in moles.
    min_pot_fraction    : stop when the pot falls below this fraction of the charge.
    max_steps           : safety limit on the number of steps.
    include_pressure_drop: feed the bed pressure drop back to the pot pressure.
    refine_margin_c     : within this many degC below the next cut edge the step is
                          shortened, so receivers switch at the right moment.
    refine_factor       : the shortened step as a fraction of the normal one.
    """

    step_fraction: float = 0.005
    min_pot_fraction: float = 0.05
    max_steps: int = 3000
    include_pressure_drop: bool = True
    refine_margin_c: float = 8.0
    refine_factor: float = 0.2


# --------------------------------------------------------------------------
# Column model: distillate composition for a given pot composition
# --------------------------------------------------------------------------

def _distillate_integer_stages(x_pot: np.ndarray, Pv: np.ndarray, N: int, R: float,
                               tol: float = 1e-9, max_iter: int = 300) -> np.ndarray:
    """Distillate mole fractions for ``N`` theoretical stages (pot included).

    Top-down from a guessed distillate composition: the top vapour equals the
    distillate (total condenser), each stage's liquid is in equilibrium with its
    vapour (y proportional to Pv x), and the material balance above a stage gives
    y_n = (R x_{n+1} + x_D)/(R + 1).  The pot liquid so obtained is compared
    with the real one and x_D is corrected component by component by the ratio
    of the two until they agree.  Top-down is the numerically stable direction
    for the heavy components.
    """
    if N <= 1:
        x_d = Pv * x_pot
        return x_d / x_d.sum()
    x_d = Pv * x_pot
    x_d /= x_d.sum()                                   # one-stage guess
    for _ in range(max_iter):
        y = x_d
        for _ in range(N):
            x = y / Pv
            x /= x.sum()
            y = (R * x + x_d) / (R + 1.0)
        # x is now the computed pot liquid
        if np.max(np.abs(x - x_pot)) < tol:
            break
        ratio = np.clip(x_pot / np.maximum(x, 1e-300), 1e-6, 1e6)
        x_d = x_d * ratio
        x_d /= x_d.sum()
    return x_d


def column_distillate(x_pot: np.ndarray, Pv: np.ndarray, N_total: float, R: float) -> np.ndarray:
    """Distillate composition for a possibly fractional number of stages.

    Blends the solutions for the two neighbouring integer stage counts, which
    keeps results continuous as the packed height changes.
    """
    if N_total < 1.0:
        raise ValueError("A column has at least one stage (the pot).")
    n0 = int(math.floor(N_total))
    w = N_total - n0
    x0 = _distillate_integer_stages(x_pot, Pv, n0, R)
    if w < 1e-9:
        return x0
    return (1.0 - w) * x0 + w * _distillate_integer_stages(x_pot, Pv, n0 + 1, R)


# --------------------------------------------------------------------------
# Fast bubble / dew points (Newton from the previous temperature)
# --------------------------------------------------------------------------

def _fast_bubble(props: VLEProps, x: np.ndarray, p_bar: float, T0: float | None) -> float:
    """Bubble-point temperature [K]; Newton from ``T0``, falling back to the
    robust bracketed solver in ``vle`` if it does not converge."""
    if T0 is not None:
        f = lambda T: math.log(float(np.dot(x, props.psat_bar(T))) / p_bar)
        T, fT = T0, f(T0)
        for _ in range(40):
            d = (f(T + 0.5) - fT) / 0.5
            if d <= 0:
                break
            T += max(-40.0, min(40.0, -fT / d))
            fT = f(T)
            if abs(fT) < 1e-10:
                return T
    return props.bubble_point(x, p_bar)[0]


def _fast_dew(props: VLEProps, y: np.ndarray, p_bar: float, T0: float | None) -> float:
    """Dew-point temperature [K], same strategy as ``_fast_bubble``."""
    if T0 is not None:
        f = lambda T: math.log(p_bar * float(np.dot(y, 1.0 / props.psat_bar(T))))
        T, fT = T0, f(T0)
        for _ in range(40):
            d = (f(T + 0.5) - fT) / 0.5
            if d >= 0:
                break
            T += max(-40.0, min(40.0, -fT / d))
            fT = f(T)
            if abs(fT) < 1e-10:
                return T
    return props.dew_point(y, p_bar)[0]


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CutResult:
    """What one receiver collected."""

    index: int
    T_lo_atm_c: float          # lower edge of the cut's range (atmospheric equivalent)
    T_hi_atm_c: float          # upper edge
    T_head_lo_c: float         # head-temperature window at P_top
    T_head_hi_c: float
    mass_kg: float
    purity: float              # mass fraction inside its own range (nan if empty)
    recovery: float            # in-range mass collected / in-range mass in the charge
    t_start_min: float
    t_end_min: float


@dataclass(frozen=True)
class BatchResult:
    """Outcome of a simulated run."""

    cuts: tuple[CutResult, ...]
    target: CutResult
    residue_kg: float                # pot residue plus liquid returned from the packing
    holdup_kg: float                 # liquid held on the packing at the start
    stop_reason: str
    t_heatup_min: float
    t_collect_min: float
    t_batch_min: float               # heat-up plus collection
    T_pot_max_c: float
    tau_equiv_min: float             # equivalent minutes at ThermalModel.T_ref_c
    N_total: float                   # theoretical stages used (pot included)
    hetp_m: float
    mass_balance_error_kg: float     # should be ~1e-12
    light_ends_min: float            # minutes the head was colder than the condenser coolant
    # time series, one entry per step (t from heater switch-on)
    t_min: np.ndarray = field(repr=False)
    T_pot_c: np.ndarray = field(repr=False)
    T_head_c: np.ndarray = field(repr=False)
    p_pot_mbar: np.ndarray = field(repr=False)
    cut_index: np.ndarray = field(repr=False)
    distillate_kg: np.ndarray = field(repr=False)      # cumulative, all cuts [kg]

    def minutes_head_below(self, T_c: float) -> float:
        """Minutes the head temperature spent below ``T_c`` during collection."""
        below = self.T_head_c < T_c
        dt = np.diff(self.t_min, prepend=self.t_min[0])
        return float(np.sum(dt[below]))

    def minutes_above(self, T_c: float) -> float:
        """Minutes the pot spent above ``T_c`` during collection."""
        above = self.T_pot_c > T_c
        dt = np.diff(self.t_min, prepend=self.t_min[0])
        return float(np.sum(dt[above]))


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _range_weights(feed: FeedCharacterisation, T_lo_c: float, T_hi_c: float) -> np.ndarray:
    """Fraction of each pseudo-component's slice of the TBP curve that lies inside
    a range (same overlap logic as ``FeedCharacterisation.cut_composition``)."""
    w_lo = float(feed.curve.frac_below(T_lo_c))
    w_hi = float(feed.curve.frac_below(T_hi_c))
    edges = np.linspace(0.0, 1.0, feed.n_cuts + 1)
    overlap = np.clip(np.minimum(edges[1:], w_hi) - np.maximum(edges[:-1], w_lo), 0.0, None)
    return overlap * feed.n_cuts


def arrhenius_factor(T_K: float, Ea_kj_mol: float, T_ref_K: float) -> float:
    """Rate relative to the reference temperature, exp(-Ea/R (1/T - 1/T_ref))."""
    return math.exp(-Ea_kj_mol * 1000.0 / R_GAS * (1.0 / T_K - 1.0 / T_ref_K))


def _estimate_hetp(feed, comps, geom, spec: BatchSpec, lo: float, hi: float) -> float:
    """HETP at the start of the target cut (used when the caller does not supply one)."""
    cut = analyse_cut(feed, lo, hi, spec.p_top_mbar)
    _, x_cut = feed.cut_composition(lo, hi)
    fl = mixture_properties(comps, x_cut, cut.T_mid_c, spec.p_top_mbar)
    flows = internal_flows(spec.q_heater_w, fl.latent_kj_kg, spec.R, spec.efficiency)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return hetp_model(geom, fl, flows, cut.alpha_min, HETPModel()).hetp_m


# --------------------------------------------------------------------------
# The simulation
# --------------------------------------------------------------------------

def simulate_batch(
    feed: FeedCharacterisation,
    geom: ColumnGeometry,
    spec: BatchSpec,
    hetp_m: float | None = None,
    thermal: ThermalModel = ThermalModel(),
    settings: SimSettings = SimSettings(),
    hold_model: HoldupModel = HoldupModel(),
    hyd_model: HydraulicsModel = HydraulicsModel(),
) -> BatchResult:
    """Simulate the batch run and return cut masses, purities, recoveries and timing.

    Parameters
    ----------
    feed, geom : characterised feed and packed-column geometry.
    spec       : operating inputs (``BatchSpec``).
    hetp_m     : HETP [m].  If omitted it is estimated with ``hetp.py`` at the
                 start conditions of the target cut; pass your calibrated value
                 when you have one.
    """
    comps = Components.from_feed(feed)
    props = comps.vle
    k_w = feed.k_watson
    n_comp = feed.n_cuts
    p_top = spec.p_top_mbar
    p_top_bar = p_top / MBAR_PER_BAR
    edges = list(spec.cut_edges_c)
    n_cut = len(edges)
    tgt = spec.target % n_cut
    ibp, fbp = feed.curve.ibp_c, feed.curve.fbp_c

    # ---- cut definitions ----------------------------------------------------
    lows = [ibp] + edges[:-1]
    highs = edges
    T_head_edge = [float(boiling_point_at_pressure(e, p_top, k_w)) for e in edges]
    T_head_low = [float(boiling_point_at_pressure(max(l, ibp), p_top, k_w)) for l in lows]
    weights = [_range_weights(feed, l, min(h, fbp)) for l, h in zip(lows, highs)]

    # ---- column: stages ------------------------------------------------------
    if hetp_m is None:
        hetp_m = _estimate_hetp(feed, comps, geom, spec, lows[tgt], min(highs[tgt], fbp))
    N_total = stages_available(geom.H_pack_m, hetp_m)

    # ---- charge in moles --------------------------------------------------------
    n_feed = 1000.0 * spec.m_feed_kg * feed.w / feed.mw           # mol
    n_feed_total = float(n_feed.sum())

    # hold-up taken out of the pot at the start (feed composition), returned at the end
    x_feed = n_feed / n_feed_total
    T0_guess = props.bubble_point(x_feed, p_top_bar)[0]
    fl0 = mixture_properties(comps, x_feed, T0_guess - KELVIN_OFFSET, p_top)
    flows0 = internal_flows(spec.q_heater_w, fl0.latent_kj_kg, spec.R, spec.efficiency)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        holdup_kg = liquid_holdup(geom, fl0, flows0, hold_model).mass_kg
    if holdup_kg >= 0.5 * spec.m_feed_kg:
        raise ValueError(f"The column holds {holdup_kg * 1000:.0f} g of liquid, more than half "
                         f"of the {spec.m_feed_kg * 1000:.0f} g charge; use a larger charge or a smaller column.")
    inventory = n_feed * (holdup_kg / spec.m_feed_kg)             # mol, feed composition
    n_pot = n_feed - inventory

    # ---- state and history ----------------------------------------------------
    dist = np.zeros((n_cut, n_comp))                              # mol collected per cut
    t_start = [math.nan] * n_cut
    t_end = [math.nan] * n_cut
    hist = {k: [] for k in ("t", "Tp", "Th", "pp", "ci", "dist")}
    cur = 0
    T_pot_K = T_head_K = None
    dP_mbar = 0.0
    t_s = 0.0
    stop = "max steps"
    step_mol = settings.step_fraction * n_feed_total
    min_pot = settings.min_pot_fraction * n_feed_total
    t_heatup_s = 0.0

    for step in range(settings.max_steps):
        x_pot = n_pot / n_pot.sum()
        p_pot_mbar = p_top + dP_mbar

        # pot: bubble point and equilibrium vapour at the pot pressure
        T_pot_K = _fast_bubble(props, x_pot, p_pot_mbar / MBAR_PER_BAR, T_pot_K)
        T_ref = 0.5 * (T_pot_K + (T_head_K if T_head_K is not None else T_pot_K))

        # column: distillate composition and head temperature
        Pv = props.psat_bar(T_ref)
        x_d = column_distillate(x_pot, Pv, N_total, spec.R)
        T_head_K = _fast_dew(props, x_d, p_top_bar, T_head_K if T_head_K is not None else T_pot_K)
        T_head_c = T_head_K - KELVIN_OFFSET
        T_pot_c = T_pot_K - KELVIN_OFFSET

        # boil-up from the heater power, evaluated at the pot
        y_pot = Pv * x_pot / float(np.dot(Pv, x_pot))
        fl_pot = mixture_properties(comps, x_pot, T_pot_c, p_pot_mbar, y_vap=y_pot)
        V_mol = spec.efficiency * spec.q_heater_w / (fl_pot.latent_kj_kg * 1000.0 * fl_pot.mw_vapour / 1000.0)
        D_mol = V_mol / (spec.R + 1.0)

        # hydraulics at the head: pressure drop sets the pot pressure for the next step
        if settings.include_pressure_drop:
            fl_head = mixture_properties(comps, x_d, T_head_c, p_top, y_vap=x_d)
            m_v = V_mol * fl_head.mw_vapour / 1000.0
            fr = spec.R / (spec.R + 1.0)
            flows = InternalFlows(m_v, m_v * fr, m_v * (1.0 - fr))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hold = liquid_holdup(geom, fl_head, flows, hold_model)
            dpdz = irrigated_pressure_gradient(hold.h_total, geom, fl_head,
                                               m_v / (fl_head.rho_V * geom.area_m2), hyd_model)
            if dpdz is None:
                stop = "flooded"
                break
            dP_mbar = dpdz * geom.H_pack_m * PA_TO_MBAR

        # start of collection: heat-up time to reach the first boiling point
        if step == 0:
            t_heatup_s = (spec.m_feed_kg * thermal.cp_kj_kgk * 1000.0
                          * max(0.0, T_pot_c - thermal.T_ambient_c)
                          / (spec.efficiency * spec.q_heater_w))
            t_s = t_heatup_s

        # receiver switching on head temperature
        while cur < n_cut and T_head_c >= T_head_edge[cur]:
            cur += 1

        # end-of-run checks, made before drawing off any more distillate
        if cur >= n_cut:
            end = "target range complete"
        elif T_pot_c > thermal.T_limit_c:
            end = "pot temperature limit"
        elif n_pot.sum() <= min_pot:
            end = "pot exhausted"
        else:
            end = None
        if end is not None:
            stop = end
            _record(hist, (t_s / 60.0, T_pot_c, T_head_c, p_pot_mbar, min(cur, n_cut - 1)),
                    _kg(dist.sum(axis=0), feed))
            break

        # draw one increment of distillate into the current receiver; a component
        # can never be drawn off faster than the pot holds it
        near_edge = T_head_c > T_head_edge[cur] - settings.refine_margin_c
        dn = min(step_mol * (settings.refine_factor if near_edge else 1.0), n_pot.sum() - min_pot)
        dn_i = np.minimum(dn * x_d, n_pot)
        removed = float(dn_i.sum())
        if removed <= 1e-12 * n_feed_total:
            stop = "stalled"
            break
        n_pot = n_pot - dn_i
        dist[cur] += dn_i
        if math.isnan(t_start[cur]):
            t_start[cur] = t_s / 60.0
        _record(hist, (t_s / 60.0, T_pot_c, T_head_c, p_pot_mbar, cur), _kg(dist.sum(axis=0), feed))
        t_s += removed / D_mol
        t_end[cur] = t_s / 60.0

    # ---- results ---------------------------------------------------------------
    residue_mol = n_pot + inventory
    kg = lambda n_vec: _kg(n_vec, feed)
    cuts = []
    for i in range(n_cut):
        m_i = kg(dist[i])
        hi_i = min(highs[i], fbp)
        in_range = float(np.dot(dist[i] * feed.mw, weights[i])) / 1000.0
        available = spec.m_feed_kg * feed.cut_mass_fraction(max(lows[i], ibp), hi_i)
        cuts.append(CutResult(
            index=i, T_lo_atm_c=lows[i], T_hi_atm_c=highs[i],
            T_head_lo_c=T_head_low[i], T_head_hi_c=T_head_edge[i], mass_kg=m_i,
            purity=in_range / m_i if m_i > 0 else math.nan,
            recovery=in_range / available if available > 0 else math.nan,
            t_start_min=t_start[i], t_end_min=t_end[i]))

    t_array = np.array(hist["t"])
    Tp_array = np.array(hist["Tp"])
    t_collect = (t_s - t_heatup_s) / 60.0

    # thermal exposure: heat-up ramp (linear in T) plus collection, trapezoid rule
    Ea, T_ref_K = thermal.Ea_kj_mol, thermal.T_ref_c + KELVIN_OFFSET
    ramp_T = np.linspace(thermal.T_ambient_c, Tp_array[0] if len(Tp_array) else thermal.T_ambient_c, 21) + KELVIN_OFFSET
    ramp_rate = np.array([arrhenius_factor(T, Ea, T_ref_K) for T in ramp_T])
    tau = float(np.trapezoid(ramp_rate, np.linspace(0.0, t_heatup_s / 60.0, 21)))
    if len(t_array) > 1:
        rate = np.array([arrhenius_factor(T + KELVIN_OFFSET, Ea, T_ref_K) for T in Tp_array])
        tau += float(np.trapezoid(rate, t_array))

    balance = spec.m_feed_kg - (sum(c.mass_kg for c in cuts) + kg(residue_mol))
    head = np.array(hist["Th"])
    dt_min = np.diff(t_array, prepend=t_array[0]) if len(t_array) else np.array([])
    light_min = float(np.sum(dt_min[head < spec.T_condenser_c])) if len(head) else 0.0
    if stop in ("flooded", "pot temperature limit", "stalled"):
        warnings.warn(f"Run stopped early: {stop}. Lower the heater power or choose larger packing.",
                      stacklevel=2)
    if light_min > 0.0:
        warnings.warn(f"The head was colder than the {spec.T_condenser_c:.0f} C condenser coolant for "
                      f"{light_min:.1f} min: light ends will not condense and go to the cold trap. "
                      "Collect the forerun at higher pressure.", stacklevel=2)

    return BatchResult(
        cuts=tuple(cuts), target=cuts[tgt], residue_kg=kg(residue_mol), holdup_kg=holdup_kg,
        stop_reason=stop, t_heatup_min=t_heatup_s / 60.0, t_collect_min=t_collect,
        t_batch_min=t_heatup_s / 60.0 + t_collect,
        T_pot_max_c=float(Tp_array.max()) if len(Tp_array) else math.nan, tau_equiv_min=tau,
        N_total=N_total, hetp_m=hetp_m, mass_balance_error_kg=balance, light_ends_min=light_min,
        t_min=t_array, T_pot_c=Tp_array, T_head_c=np.array(hist["Th"]),
        p_pot_mbar=np.array(hist["pp"]), cut_index=np.array(hist["ci"], dtype=int),
        distillate_kg=np.array(hist["dist"]))


def _kg(n_vec: np.ndarray, feed: FeedCharacterisation) -> float:
    """Mass [kg] of a vector of moles per pseudo-component."""
    return float(np.dot(n_vec, feed.mw)) / 1000.0


def _record(hist: dict, record: tuple, cumulative_kg: float) -> None:
    """Append one history sample (time, temperatures, pressure, cut, cumulative distillate)."""
    t, Tp, Th, pp, ci = record
    hist["t"].append(t)
    hist["Tp"].append(Tp)
    hist["Th"].append(Th)
    hist["pp"].append(pp)
    hist["ci"].append(ci)
    hist["dist"].append(cumulative_kg)


# --------------------------------------------------------------------------
# Demo / smoke test:  python -m fdc.batch_pot
# --------------------------------------------------------------------------

if __name__ == "__main__":
    from .packing import PackingSpec, column_geometry
    from .properties.feed import DEFAULT_WPPO_TBP, TBPCurve, characterise_feed

    warnings.simplefilter("ignore")                       # heavy-fraction notices from vle
    feed = characterise_feed(TBPCurve.from_points(DEFAULT_WPPO_TBP), n_cuts=30, sg_feed=0.80)
    geom = column_geometry(PackingSpec("glass_beads", 5.0), D_mm=50.0, H_pack_m=0.60)
    spec = BatchSpec(m_feed_kg=1.0, p_top_mbar=100.0, R=5.0, q_heater_w=100.0,
                     cut_edges_c=(150.0, 250.0))

    res = simulate_batch(feed, geom, spec)
    print(f"1.0 kg charge, P_top {spec.p_top_mbar:.0f} mbar, R = {spec.R:.0f}, {spec.q_heater_w:.0f} W, "
          f"5 mm beads, 50 mm x 0.60 m")
    print(f"  stages {res.N_total:.1f} (HETP {res.hetp_m * 1000:.0f} mm) | hold-up {res.holdup_kg * 1000:.0f} g | "
          f"stop: {res.stop_reason}")
    print(f"  heat-up {res.t_heatup_min:.0f} min + collection {res.t_collect_min:.0f} min = "
          f"{res.t_batch_min:.0f} min | pot max {res.T_pot_max_c:.0f} C | "
          f"thermal exposure {res.tau_equiv_min:.4f} min-equivalent at {ThermalModel().T_ref_c:.0f} C\n")
    for c in res.cuts:
        print(f"  cut {c.index}: {c.T_lo_atm_c:.0f}-{c.T_hi_atm_c:.0f} C atm-equiv "
              f"(head {c.T_head_lo_c:.0f} to {c.T_head_hi_c:.0f} C): {c.mass_kg * 1000:.0f} g, "
              f"purity {c.purity:.2f}, recovery {c.recovery:.2f}")
    print(f"  residue (pot + packing) {res.residue_kg * 1000:.0f} g; mass balance error "
          f"{res.mass_balance_error_kg:.1e} kg")
    print(f"  head colder than the {spec.T_condenser_c:.0f} C coolant for {res.light_ends_min:.0f} min "
          "(light ends would escape to the cold trap)\n")

    print("  time/min  T_pot/C  T_head/C  P_pot/mbar  cut  collected/g")
    for i in np.linspace(0, len(res.t_min) - 1, 8).astype(int):
        print(f"  {res.t_min[i]:8.0f} {res.T_pot_c[i]:8.1f} {res.T_head_c[i]:9.1f} {res.p_pot_mbar[i]:11.1f} "
              f"{res.cut_index[i]:4d} {res.distillate_kg[i] * 1000:11.0f}")
