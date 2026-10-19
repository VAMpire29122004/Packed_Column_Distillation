"""
app.plots
=========

Figure builders for the Streamlit app.  Every function takes engine objects
and returns a ``matplotlib.figure.Figure``; the app displays them with
``st.pyplot(fig)``.

Figures are built with the object-oriented ``Figure`` API (no ``pyplot``), so
there is no hidden global state and they are safe to create from Streamlit's
script threads.

Figures
-------
* ``feed_curve_plot``           feed boiling-point curve with the cuts shaded
* ``boiling_point_plot``        atmospheric-equivalent versus vacuum boiling temperature
* ``stages_vs_reflux_plot``     stages needed against reflux ratio (vacuum vs atmospheric)
* ``hydraulics_vs_power_plot``  pressure drop and % flooding against heater power
* ``batch_history_plot``        pot and head temperature, collected cuts and pot pressure over time
* ``cut_summary_plot``          mass, purity and recovery of each cut
* ``design_sweep_plot``         diameter, packed height and stages across packing sizes or pressures

Conventions
-----------
Temperatures in degC (cut limits atmospheric-equivalent), pressures in mbar,
masses in g, times in min.  Colours match the process schematic.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from matplotlib.figure import Figure

from fdc.holdup import internal_flows, liquid_holdup
from fdc.hydraulics import flooding_vapour_rate, irrigated_pressure_gradient
from fdc.properties.vle import boiling_point_at_pressure
from fdc.stages import SeparationSpec, cut_stage_requirement, stages_at_reflux

# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------

C_POT, C_HEAD, C_FLOOD, C_DP = "#c0392b", "#2f6db5", "#e07b6b", "#7a1f1f"
C_GREY = "#777777"
CUT_COLORS = ["#a5c8ea", "#f4cccc", "#d6e4c2", "#f1e3b5", "#d9c3e8", "#cfd8dc"]


def _new_fig(w: float = 7.2, h: float = 3.7) -> Figure:
    return Figure(figsize=(w, h), dpi=100, constrained_layout=True)


def _style(ax) -> None:
    ax.grid(True, alpha=0.25, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _cut_label(lo: float, hi: float) -> str:
    return f"{lo:.0f}\u2013{hi:.0f} \u00b0C"


# --------------------------------------------------------------------------
# Feed curve
# --------------------------------------------------------------------------

def feed_curve_plot(feed, cuts=None) -> Figure:
    """TBP curve (temperature against cumulative mass %), pseudo-components and cuts.

    ``cuts`` is a list of ``(T_lo, T_hi)`` atmospheric-equivalent limits; each is
    shaded and labelled with the share of the feed that boils inside it.
    """
    fig = _new_fig()
    ax = fig.subplots()
    w = np.linspace(0.0, 1.0, 200)
    ax.plot(w * 100, feed.curve.temp_at(w), color="#222222", lw=2, label="TBP curve")
    mid = (np.arange(feed.n_cuts) + 0.5) / feed.n_cuts
    ax.plot(mid * 100, feed.Tb_K - 273.15, "o", ms=3.5, color=C_GREY, label=f"{feed.n_cuts} pseudo-components")
    ibp, fbp = feed.curve.ibp_c, feed.curve.fbp_c
    for i, (lo, hi) in enumerate(cuts or []):
        lo_c, hi_c = max(lo, ibp), min(hi, fbp)
        if hi_c <= lo_c:
            continue
        col = CUT_COLORS[i % len(CUT_COLORS)]
        ax.axhspan(lo_c, hi_c, color=col, alpha=0.55, lw=0)
        w_lo, w_hi = float(feed.curve.frac_below(lo_c)), float(feed.curve.frac_below(hi_c))
        ax.vlines([w_lo * 100, w_hi * 100], ibp - 20, [lo_c, hi_c], colors=C_GREY, linestyles=":", lw=1)
        ax.text(99, (lo_c + hi_c) / 2, f"{_cut_label(lo, hi)}: {100 * (w_hi - w_lo):.1f} % of feed",
                ha="right", va="center", fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_ylim(ibp - 20, fbp + 20)
    ax.set_xlabel("Cumulative mass distilled (%)")
    ax.set_ylabel("Atmospheric-equivalent boiling point (\u00b0C)")
    ax.set_title("Feed boiling-point curve and cuts", fontsize=11)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    _style(ax)
    return fig


# --------------------------------------------------------------------------
# Vacuum boiling points
# --------------------------------------------------------------------------

def _bp_line(T_atm: np.ndarray, p_mbar: float, k_watson: float) -> np.ndarray:
    """Boiling temperature at ``p_mbar`` for each atmospheric-equivalent T; nan where
    the correlation does not cover the pressure."""
    out = np.full(T_atm.shape, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, T in enumerate(T_atm):
            try:
                out[i] = boiling_point_at_pressure(float(T), p_mbar, k_watson)
            except ValueError:
                pass
    return out


def boiling_point_plot(feed, pressures_mbar, p_op_mbar: float | None = None, cut=None,
                       T_condenser_c: float | None = None) -> Figure:
    """Boiling temperature at each pressure against the atmospheric-equivalent value.

    ``p_op_mbar`` highlights the operating line and, with ``cut = (lo, hi)``, shades
    the head-temperature window an operator would watch on TT-102.
    """
    fig = _new_fig()
    ax = fig.subplots()
    ibp, fbp = feed.curve.ibp_c, feed.curve.fbp_c
    T = np.linspace(ibp, fbp, 40)
    others = iter(["#8c8c8c", "#e69f00", "#009e73", "#cc79a7", "#56b4e9"])
    for p in sorted(set(pressures_mbar) | ({p_op_mbar} if p_op_mbar else set()), reverse=True):
        y = _bp_line(T, p, feed.k_watson)
        op = p_op_mbar is not None and math.isclose(p, p_op_mbar)
        ax.plot(T, y, lw=2.6 if op else 1.3, color=C_HEAD if op else next(others),
                label=f"{p:g} mbar" + (" (operating)" if op else ""))
    if cut is not None:
        lo, hi = max(cut[0], ibp), min(cut[1], fbp)
        ax.axvspan(lo, hi, color=CUT_COLORS[0], alpha=0.45, lw=0)
        if p_op_mbar is not None:
            lo_p, hi_p = (float(boiling_point_at_pressure(x, p_op_mbar, feed.k_watson)) for x in (lo, hi))
            ax.axhspan(lo_p, hi_p, xmin=0, xmax=1, color=CUT_COLORS[0], alpha=0.25, lw=0)
            ax.annotate(f"head window {lo_p:.0f}\u2013{hi_p:.0f} \u00b0C at {p_op_mbar:g} mbar",
                        xy=(hi, hi_p), xytext=(ibp + 5, hi_p + 25), fontsize=8,
                        arrowprops=dict(arrowstyle="->", color=C_GREY, lw=0.8))
    if T_condenser_c is not None:
        ax.axhline(T_condenser_c, color=C_POT, ls="--", lw=1)
        ax.text(fbp, T_condenser_c + 4, "condenser coolant", color=C_POT, fontsize=8, ha="right")
    ax.set_xlabel("Atmospheric-equivalent boiling point (\u00b0C)")
    ax.set_ylabel("Boiling temperature at that pressure (\u00b0C)")
    ax.set_title("What vacuum does to the boiling points", fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    _style(ax)
    return fig


# --------------------------------------------------------------------------
# Stages against reflux
# --------------------------------------------------------------------------

def stages_vs_reflux_plot(feed, lo: float, hi: float, p_mbar: float, sep: SeparationSpec = SeparationSpec(),
                          R_current: float | None = None, N_available: float | None = None,
                          compare_mbar=(1013.25,)) -> Figure:
    """Total theoretical stages against reflux ratio for the operating pressure and
    for comparison pressures (default atmospheric).

    ``R_current`` marks the chosen reflux ratio; ``N_available`` draws the stages a
    given column provides (rating mode).  If that is below the minimum possible
    for the cut (infinite reflux), the label says the cut cannot be made.
    """
    fig = _new_fig()
    ax = fig.subplots()
    pressures = [p_mbar] + [p for p in compare_mbar if not math.isclose(p, p_mbar)]

    # boundaries at each pressure (computed once); one common reflux range for all curves
    probes = []
    for p in pressures:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probe = cut_stage_requirement(feed, lo, hi, p, 1.0e6, sep)
        if not probe.boundaries:
            raise ValueError("This cut takes the whole feed: no separation, no stage curve.")
        probes.append(probe)
    R_hi = 6.0 * max(pr.R_min for pr in probes)

    def n_at(probe, R):
        return max(stages_at_reflux(b.N_min, b.R_min, R) for b in probe.boundaries)

    y_top = 0.0
    for i, (p, probe) in enumerate(zip(pressures, probes)):
        op = i == 0
        col = C_HEAD if op else C_GREY
        R_start = probe.R_min * 1.01
        R_grid = R_start + (R_hi - R_start) * np.linspace(0.0, 1.0, 120) ** 2.5     # dense near R_min
        N = np.array([n_at(probe, R) for R in R_grid])
        ax.plot(R_grid, N, lw=2.6 if op else 1.4, ls="-" if op else "--", color=col,
                label=f"{p:g} mbar" + (" (operating)" if op else ""))
        ax.axvline(probe.R_min, color=col, ls=":", lw=1, label=f"R_min = {probe.R_min:.2f} at {p:g} mbar")
        y_top = max(y_top, 1.25 * n_at(probe, 1.6 * probe.R_min))
    op_probe = probes[0]
    if R_current is not None and R_current > op_probe.R_min:
        n_cur = n_at(op_probe, R_current)
        ax.plot([R_current], [n_cur], "o", color=C_POT, ms=7, zorder=5)
        ax.annotate(f"R = {R_current:.2f}: {n_cur:.1f} stages", xy=(R_current, n_cur),
                    xytext=(R_current + 0.05 * R_hi, n_cur + 0.12 * y_top), fontsize=8,
                    arrowprops=dict(arrowstyle="->", color=C_POT, lw=0.8))
    if N_available is not None:
        n_floor = n_at(op_probe, math.inf)                                       # N_min at total reflux
        note = "" if N_available > n_floor else f" (below the minimum {n_floor:.1f}: cannot make this cut)"
        ax.axhline(N_available, color=C_POT, ls="--", lw=1.2)
        ax.text(0.15, N_available - 0.012 * y_top, f"stages the column provides: {N_available:.1f}{note}",
                color=C_POT, fontsize=8, ha="left", va="top")
        y_top = max(y_top, 1.15 * N_available)
    ax.set_ylim(0, y_top)
    ax.set_xlim(left=0)
    ax.set_xlabel("Reflux ratio R")
    ax.set_ylabel("Theoretical stages (pot included)")
    ax.set_title(f"Stages for the {_cut_label(lo, hi)} cut", fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    _style(ax)
    return fig


# --------------------------------------------------------------------------
# Hydraulics against heater power
# --------------------------------------------------------------------------

def hydraulics_vs_power_plot(geom, fluid, R: float, efficiency: float = 0.70, q_current: float | None = None,
                             hold_model=None, hyd_model=None, n_points: int = 40) -> Figure:
    """Bed pressure drop and % flooding against heater power at constant reflux ratio.

    % flooding is linear in power (the flooding vapour rate depends only on L/V),
    so it is found from a single flooding search; the pressure drop is solved at
    each power below the flooding point.
    """
    from fdc.holdup import HoldupModel
    from fdc.hydraulics import HydraulicsModel
    hold_model = hold_model or HoldupModel()
    hyd_model = hyd_model or HydraulicsModel()
    phi = R / (R + 1.0)
    V_flood = flooding_vapour_rate(phi, geom, fluid, hold_model, hyd_model)           # kg/s
    q_flood = V_flood * fluid.latent_kj_kg * 1000.0 / efficiency                      # W at 100 %
    powers = np.linspace(0.04, 1.30, n_points) * q_flood
    dp = np.full(n_points, np.nan)
    for i, q in enumerate(powers):
        flows = internal_flows(float(q), fluid.latent_kj_kg, R, efficiency)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            h0 = liquid_holdup(geom, fluid, flows, hold_model).h_total
        g = irrigated_pressure_gradient(h0, geom, fluid, flows.m_vapour / (fluid.rho_V * geom.area_m2), hyd_model)
        if g is not None:
            dp[i] = g * geom.H_pack_m * 0.01                                          # Pa -> mbar
    pct = 100.0 * powers / q_flood

    fig = _new_fig()
    ax = fig.subplots()
    ax2 = ax.twinx()
    ax.plot(powers, dp, color=C_DP, lw=2.4, label="bed pressure drop")
    ax2.plot(powers, pct, color=C_FLOOD, lw=1.6, ls="--", label="% of flooding")
    ax2.axhspan(85, 100, color=C_FLOOD, alpha=0.15, lw=0)
    ax2.axhline(100, color=C_FLOOD, lw=1)
    ax2.text(powers[0], 101.5, "flooding", color=C_FLOOD, fontsize=8)
    if q_current is not None:
        pc = 100.0 * q_current / q_flood
        ax.axvline(q_current, color=C_POT, ls=":", lw=1.4)
        ax2.plot([q_current], [pc], "o", color=C_POT, ms=7, zorder=5)
        ax2.annotate(f"{q_current:.0f} W: {pc:.0f} % of flooding", xy=(q_current, pc),
                     xytext=(q_current + 0.05 * powers[-1], max(pc - 22, 6)), fontsize=8,
                     arrowprops=dict(arrowstyle="->", color=C_POT, lw=0.8))
    ax.set_xlabel("Heater power (W)")
    ax.set_ylabel("Pressure drop across the bed (mbar)", color=C_DP)
    ax2.set_ylabel("% of flooding", color=C_FLOOD)
    ax2.set_ylim(0, 135)
    ax.set_ylim(0, None)
    ax.set_title(f"Hydraulics at R = {R:.2f} (flooding at {q_flood:.0f} W)", fontsize=11)
    _style(ax)
    ax2.spines["top"].set_visible(False)
    return fig


# --------------------------------------------------------------------------
# Batch history
# --------------------------------------------------------------------------

def _segments(cut_index: np.ndarray):
    """(start, end, cut) index ranges of contiguous cut assignments."""
    edges = np.flatnonzero(np.diff(cut_index)) + 1
    starts = np.concatenate([[0], edges])
    ends = np.concatenate([edges, [len(cut_index)]])
    return [(int(a), int(b), int(cut_index[a])) for a, b in zip(starts, ends)]


def batch_history_plot(res, spec=None) -> Figure:
    """Pot and head temperature, cumulative distillate by cut and pot pressure over time.

    Background bands show which receiver is being filled.  With ``spec`` the
    condenser coolant temperature and P_top are drawn for reference.
    """
    fig = _new_fig(7.2, 6.6)
    ax1, ax2, ax3 = fig.subplots(3, 1, sharex=True, gridspec_kw={"height_ratios": [3, 2.2, 1.3]})
    t = res.t_min
    nc = len(res.cuts)
    for a, b, c in _segments(res.cut_index):
        for ax in (ax1, ax2, ax3):
            ax.axvspan(t[a], t[min(b, len(t) - 1)], color=CUT_COLORS[c % len(CUT_COLORS)], alpha=0.30, lw=0)
    ax1.plot(t, res.T_pot_c, color=C_POT, lw=2, label="pot (TT-101)")
    ax1.plot(t, res.T_head_c, color=C_HEAD, lw=2, label="head (TT-102)")
    for c in res.cuts:
        ax1.axhline(c.T_head_hi_c, color=C_GREY, ls=":", lw=0.8)
    if spec is not None:
        ax1.axhline(spec.T_condenser_c, color=C_POT, ls="--", lw=0.9)
        ax1.text(t[-1], spec.T_condenser_c + 3, "condenser coolant", color=C_POT, fontsize=8, ha="right")
    ax1.set_ylabel("Temperature (\u00b0C)")
    ax1.legend(fontsize=8, frameon=False, loc="upper left")
    ax1.set_title("Simulated batch (bands: receiver being filled)", fontsize=11)

    inc = np.diff(res.distillate_kg, prepend=0.0) * 1000.0
    layers = [np.cumsum(inc * (res.cut_index == c)) for c in range(nc)]
    ax2.stackplot(t, *layers, colors=[CUT_COLORS[c % len(CUT_COLORS)] for c in range(nc)],
                  labels=[_cut_label(c.T_lo_atm_c, c.T_hi_atm_c) for c in res.cuts], edgecolor="#555555", linewidth=0.6)
    ax2.set_ylabel("Collected (g)")
    ax2.legend(fontsize=8, frameon=False, loc="upper left", ncol=min(nc, 3))

    ax3.plot(t, res.p_pot_mbar, color=C_DP, lw=1.8)
    if spec is not None:
        ax3.axhline(spec.p_top_mbar, color=C_GREY, ls=":", lw=1)
        ax3.text(t[0], spec.p_top_mbar, " P_top", color=C_GREY, fontsize=8, va="bottom")
    ax3.set_ylabel("Pot pressure (mbar)")
    ax3.set_xlabel("Time since heater on (min)")
    for ax in (ax1, ax2, ax3):
        _style(ax)
    return fig


# --------------------------------------------------------------------------
# Cut summary
# --------------------------------------------------------------------------

def cut_summary_plot(res) -> Figure:
    """Mass of each cut and the residue, annotated with purity and recovery."""
    fig = _new_fig(7.2, 3.4)
    ax = fig.subplots()
    labels = [_cut_label(c.T_lo_atm_c, c.T_hi_atm_c) for c in res.cuts] + ["residue\n(pot + packing)"]
    masses = [c.mass_kg * 1000.0 for c in res.cuts] + [res.residue_kg * 1000.0]
    colors = [CUT_COLORS[i % len(CUT_COLORS)] for i in range(len(res.cuts))] + ["#bdbdbd"]
    bars = ax.bar(range(len(masses)), masses, color=colors, edgecolor="#555555", lw=0.8)
    for i, (b, c) in enumerate(zip(bars, list(res.cuts) + [None])):
        txt = f"{masses[i]:.0f} g" if c is None else f"{masses[i]:.0f} g\npurity {c.purity:.2f}\nrecovery {c.recovery:.2f}"
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), txt, ha="center", va="bottom", fontsize=8)
    tgt = res.target.index
    bars[tgt].set_linewidth(2.4)
    bars[tgt].set_edgecolor(C_HEAD)
    ax.set_xticks(range(len(masses)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Mass (g)")
    ax.set_ylim(0, max(masses) * 1.32)
    ax.set_title("Cuts collected (target outlined)", fontsize=11)
    _style(ax)
    return fig


# --------------------------------------------------------------------------
# Design sweeps
# --------------------------------------------------------------------------

def design_sweep_plot(xs, results, xlabel: str, log_x: bool = False) -> Figure:
    """Diameter, packed height and stages of designed columns across ``xs``.

    ``results`` holds one ``DesignResult`` (or ``None`` for an infeasible design)
    per x value, e.g. from ``fdc.design_mode.compare_packings`` or a pressure sweep.
    Infeasible points are marked on the axis.
    """
    fig = _new_fig(7.2, 6.0)
    axes = fig.subplots(3, 1, sharex=True)
    xs = np.asarray(xs, dtype=float)
    fields = [("D_mm", "Column diameter (mm)", C_HEAD), ("H_pack_m", "Packed height (m)", C_DP),
              ("N_total", "Theoretical stages", C_GREY)]
    ok = np.array([r is not None for r in results])
    for ax, (name, lab, col) in zip(axes, fields):
        y = np.array([getattr(r, name) if r is not None else np.nan for r in results])
        ax.plot(xs, y, "o-", color=col, lw=2, ms=5)
        for x_bad in xs[~ok]:
            ax.axvline(x_bad, color=C_FLOOD, ls=":", lw=1)
            ax.text(x_bad, ax.get_ylim()[1], " infeasible", color=C_FLOOD, fontsize=8, rotation=90, va="top")
        ax.set_ylabel(lab, fontsize=9)
        ax.set_ylim(0, None)
        _style(ax)
    axes[-1].set_xlabel(xlabel)
    if log_x:
        axes[-1].set_xscale("log")
    axes[0].set_title("Designed column across the sweep", fontsize=11)
    return fig


# --------------------------------------------------------------------------
# Demo:  python -m app.plots   (writes PNGs of every figure to ./plots_demo/)
# --------------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path

    from fdc.batch_pot import BatchSpec, simulate_batch
    from fdc.design_mode import compare_packings, design_column
    from fdc.packing import PackingSpec, column_geometry
    from fdc.properties.feed import DEFAULT_WPPO_TBP, TBPCurve, characterise_feed
    from fdc.properties.fluids import Components, mixture_properties
    from fdc.properties.vle import analyse_cut

    warnings.simplefilter("ignore")
    out = Path("plots_demo")
    out.mkdir(exist_ok=True)

    feed = characterise_feed(TBPCurve.from_points(DEFAULT_WPPO_TBP), n_cuts=30, sg_feed=0.80)
    geom = column_geometry(PackingSpec("glass_beads", 5.0), D_mm=50.0, H_pack_m=0.60)
    spec = BatchSpec(m_feed_kg=1.0, p_top_mbar=100.0, R=5.0, q_heater_w=100.0, cut_edges_c=(150.0, 250.0))
    res = simulate_batch(feed, geom, spec)

    cut = analyse_cut(feed, 150.0, 250.0, 100.0)
    _, x_cut = feed.cut_composition(150.0, 250.0)
    fluid = mixture_properties(Components.from_feed(feed), x_cut, cut.T_hi_c, 100.0)

    figs = {
        "feed_curve": feed_curve_plot(feed, cuts=[(feed.curve.ibp_c, 150.0), (150.0, 250.0)]),
        "boiling_points": boiling_point_plot(feed, [1013.25, 100.0, 20.0], 100.0, (150.0, 250.0), 15.0),
        "stages_vs_reflux": stages_vs_reflux_plot(feed, 150.0, 250.0, 100.0, R_current=5.0, N_available=res.N_total),
        "hydraulics": hydraulics_vs_power_plot(geom, fluid, 5.0, q_current=100.0),
        "batch_history": batch_history_plot(res, spec),
        "cut_summary": cut_summary_plot(res),
    }
    sizes = (3, 4, 5, 6, 8)
    cmp = compare_packings(sizes, feed=feed, T_lo_c=150.0, T_hi_c=250.0, p_top_mbar=100.0, q_heater_w=100.0, verify=False)
    figs["design_sweep"] = design_sweep_plot(sizes, [r for _, r, _ in cmp], "Bead diameter (mm)")
    for name, fig in figs.items():
        fig.savefig(out / f"{name}.png")
    print(f"wrote {len(figs)} figures to {out}/")
