"""
app.schematic
=============

Parametric, live version of the process schematic (``assets/fdc_schematic.svg``).
``render_schematic`` turns a ``SchematicState`` into a self-contained SVG string
that the Streamlit app displays; every drawn quantity comes from the model.

What is scaled from the inputs
------------------------------
* column height and packed-bed height (one common scale, so a taller bed looks
  taller), column diameter, bead size relative to the diameter (or a wool
  texture for steel wool);
* liquid level in the pot, hold-up shading on the bed (from the liquid
  saturation of the voids), number and fill of the cut receivers;
* live readouts on the instruments (TT-101 pot, TT-102 head, PT-101 top and
  PT-102 pot pressure) and a flooding-margin gauge, with a red FLOODED state.

Two helpers build a state from engine results so the app can scrub through a
simulated batch: ``state_from_design`` (a designed column at rest) and
``state_from_batch`` (any step of a ``BatchResult``).

Conventions
-----------
* The SVG uses only inline attributes (no <style> block, no external files) so
  it displays identically inside Streamlit's markdown or an HTML component, and
  it carries its own white background so it reads well in light and dark themes.
* Drawing units are pixels of a 1100-wide canvas; the height grows with the
  column.  Geometry is schematic (the flask size is fixed): only the column and
  bed follow real proportions.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SchematicState:
    """Everything the drawing needs.  Every field has a sensible default."""

    # column geometry (real proportions)
    D_mm: float = 50.0
    H_pack_m: float = 0.60
    H_col_m: float = 0.75
    packing_kind: str = "glass_beads"      # or "steel_wool"
    packing_size_mm: float = 5.0           # bead or wire diameter
    # operating labels
    R: float | None = 5.0
    q_heater_w: float | None = 100.0
    p_top_mbar: float | None = None
    # live readouts (None shows an em dash)
    T_pot_c: float | None = None
    T_head_c: float | None = None
    p_pot_mbar: float | None = None
    time_min: float | None = None
    # column state
    saturation: float = 0.0                # liquid fill of the voids, 0..1
    percent_flood: float | None = None
    flooded: bool = False
    # pot
    pot_fill: float = 0.45                 # liquid level in the flask, 0..1
    # receivers
    cut_labels: tuple[str, ...] = ("Cut 1", "Cut 2")
    cut_masses_g: tuple[float, ...] = (0.0, 0.0)
    active_cut: int | None = None
    fill_ref_g: float | None = None        # mass that fills a flask (default: largest cut)
    # flags
    light_ends: bool = False


# --------------------------------------------------------------------------
# Small SVG helpers (inline attributes only)
# --------------------------------------------------------------------------

FONT = 'font-family="Arial, Helvetica, sans-serif"'
C_TEXT, C_TAG, C_GLASS = "#222222", "#7a1f1f", "#555555"
C_VAP, C_LIQ, C_CW, C_VAC = "#e8943a", "#2f6db5", "#17a2b8", "#777777"


def _t(x, y, s, size=12, fill=C_TEXT, anchor="start", weight="normal", italic=False) -> str:
    style = ' font-style="italic"' if italic else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}"{style} {FONT}>{html.escape(str(s))}</text>')


def _line(x1, y1, x2, y2, stroke, w=2.5, dash=None, marker=None) -> str:
    d = f' stroke-dasharray="{dash}"' if dash else ""
    m = f' marker-end="url(#{marker})"' if marker else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" '
            f'stroke-width="{w}"{d}{m}/>')


def _path(d, stroke, w=2.5, dash=None, marker=None, fill="none") -> str:
    da = f' stroke-dasharray="{dash}"' if dash else ""
    m = f' marker-end="url(#{marker})"' if marker else ""
    return f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{w}"{da}{m}/>'


def _rect(x, y, w, h, fill="#ffffff", stroke=C_GLASS, sw=1.4, rx=0, extra="") -> str:
    r = f' rx="{rx}"' if rx else ""
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"{r}{extra}/>')


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _fmt(v, unit: str, nd: int = 1) -> str:
    return "\u2014" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f} {unit}"


def _sensor(x, y, label, tag_y_off=0) -> str:
    """Circular instrument bubble with a letter."""
    return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="#ffffff" stroke="{C_TAG}" stroke-width="1.4"/>'
            + _t(x, y + 4, label, 10, C_TAG, "middle", "bold"))


def _defs(bead_px: float, wool: bool) -> str:
    def marker(i, c):
        return (f'<marker id="{i}" markerUnits="userSpaceOnUse" markerWidth="10" markerHeight="10" refX="5" '
                f'refY="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>')
    t = bead_px
    if wool:
        packing = ('<pattern id="packing" width="14" height="14" patternUnits="userSpaceOnUse">'
                   '<path d="M0,3 C4,-2 8,8 14,3 M0,10 C5,5 9,15 14,10 M3,0 C-2,5 8,9 3,14" fill="none" '
                   'stroke="#8d8d8d" stroke-width="0.9"/></pattern>')
    else:
        packing = (f'<pattern id="packing" width="{t:.2f}" height="{t:.2f}" patternUnits="userSpaceOnUse">'
                   f'<circle cx="{t / 2:.2f}" cy="{t / 2:.2f}" r="{max(t / 2 - 0.7, 1):.2f}" fill="#ffffff" '
                   f'stroke="#666" stroke-width="0.9"/></pattern>')
    return ("<defs>" + marker("aO", C_VAP) + marker("aB", C_LIQ) + marker("aC", C_CW) + marker("aG", C_VAC)
            + '<pattern id="insul" width="8" height="8" patternUnits="userSpaceOnUse" '
              'patternTransform="rotate(45)"><rect width="8" height="8" fill="#f1ece0"/>'
              '<line x1="0" y1="0" x2="0" y2="8" stroke="#cfc6ae" stroke-width="2"/></pattern>'
            + packing + "</defs>")


# --------------------------------------------------------------------------
# Main renderer
# --------------------------------------------------------------------------

def render_schematic(s: SchematicState) -> str:
    """Return the schematic for state ``s`` as an SVG string."""
    if s.D_mm <= 0 or s.H_pack_m <= 0 or s.H_col_m < s.H_pack_m:
        raise ValueError("Need D > 0 and 0 < H_pack <= H_col.")
    n_cut = max(1, min(len(s.cut_labels), 6))

    # ---- layout: one pixel scale for column and bed heights ----------------------
    Xc = 260.0
    scale = min(280.0, 470.0 / s.H_col_m)                 # px per metre
    Hc = max(150.0, s.H_col_m * scale)
    scale = Hc / s.H_col_m
    Wc = _clamp(s.D_mm * 1.0, 26.0, 120.0)
    k = Wc / s.D_mm                                       # px per mm (same for beads)
    bead_px = _clamp(s.packing_size_mm * k, 4.0, 16.0)

    y_head_top, head_h = 58.0, 62.0
    y_col_top = y_head_top + head_h
    y_col_bot = y_col_top + Hc
    Hp = s.H_pack_m * scale
    extra = Hc - Hp
    y_bed_bot = y_col_bot - extra / 3.0
    y_bed_top = y_bed_bot - Hp
    cy = y_col_bot + 12 + 100                             # flask centre
    H_canvas = max(710.0, cy + 215.0)
    x0, x1 = Xc - Wc / 2, Xc + Wc / 2
    hw = max(88.0, Wc + 36.0)
    hx0, hx1 = Xc - hw / 2, Xc + hw / 2
    wool = s.packing_kind == "steel_wool"

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 {H_canvas:.0f}" width="100%" role="img" '
         f'aria-label="Packed column schematic">',
         _defs(bead_px, wool),
         f'<rect x="1" y="1" width="1098" height="{H_canvas - 2:.0f}" rx="10" fill="#ffffff" stroke="#d0d0d0"/>',
         _t(20, 28, "Packed-column batch fractionation of WPPO", 17, C_TEXT, weight="bold")]

    # ---- heating mantle and flask ---------------------------------------------------
    o.append(f'<path d="M135,{cy + 160:.1f} Q135,{cy + 80:.1f} 260,{cy + 80:.1f} Q385,{cy + 80:.1f} 385,{cy + 160:.1f} '
             f'L385,{cy + 188:.1f} L135,{cy + 188:.1f} Z" fill="#f4cccc" stroke="{C_GLASS}" stroke-width="1.4"/>')
    heater = "Heating mantle + stirrer" if s.q_heater_w is None else f"Heater {s.q_heater_w:.0f} W (mantle + stirrer)"
    o.append(_t(260, cy + 176, heater, 12, C_TEXT, "middle"))
    o.append(_rect(385, cy + 118, 14, 24, "#ffffff", C_TAG, 1.4))
    o.append(_line(399, cy + 130, 440, cy + 130, C_TAG, 1, "3 3"))
    o.append(_t(444, cy + 126, "TSH-101", 11, C_TAG, weight="bold") + _t(444, cy + 140, "mantle high-T cut-out", 12))

    level_y = cy + 100 - 200 * _clamp(s.pot_fill, 0.0, 0.95)       # flask bottom is cy + 100
    o.append(f'<clipPath id="flaskclip"><circle cx="260" cy="{cy:.1f}" r="99"/></clipPath>')
    o.append(f'<circle cx="260" cy="{cy:.1f}" r="100" fill="#ffffff" stroke="{C_GLASS}" stroke-width="1.6"/>')
    o.append(f'<rect x="160" y="{level_y:.1f}" width="200" height="{cy + 110 - level_y:.1f}" fill="#3a3a3a" '
             f'clip-path="url(#flaskclip)"/>')
    o.append(f'<ellipse cx="240" cy="{cy + 92:.1f}" rx="14" ry="4" fill="#ffffff" stroke="{C_GLASS}"/>')
    o.append(_t(285, min(cy + 60, level_y + 30), "WPPO charge", 13, "#ffffff", "middle"))
    if s.time_min is not None and s.pot_fill > 0.02:
        for ax in (260, 290, 320):
            o.append(_line(ax, level_y - 4, ax, level_y - 44, C_VAP, 2.5, marker="aO"))
    o.append(_rect(244, y_col_bot, 32, 12, "#ffffff", C_GLASS, 1.6))                       # neck
    # side neck with the pot thermocouple, tip below the liquid line
    o.append(f'<polygon points="192,{cy - 62:.1f} 205,{cy - 72:.1f} 232,{cy - 40:.1f} 219,{cy - 30:.1f}" '
             f'fill="#ffffff" stroke="{C_GLASS}" stroke-width="1.6"/>')
    tip_y = min(level_y + 14, cy + 84)
    o.append(_line(196, cy - 66, 232, tip_y, C_TAG, 2))
    o.append(f'<circle cx="232" cy="{tip_y:.1f}" r="4" fill="#ffffff" stroke="{C_TAG}" stroke-width="1.4"/>')
    o.append(_t(20, cy - 80, "TT-101", 11, C_TAG, weight="bold") + _t(20, cy - 66, "Pot liquid T", 12))
    o.append(_t(20, cy - 50, _fmt(s.T_pot_c, "\u00b0C"), 14, C_TAG, weight="bold"))
    o.append(_line(140, cy - 62, 193, cy - 62, C_TAG, 1, "3 3"))
    # pot pressure tap
    o.append(_line(276, y_col_bot + 6, 330, y_col_bot + 6, C_TAG, 1, "3 3"))
    o.append(_sensor(338, y_col_bot + 6, "P"))
    o.append(_t(352, y_col_bot + 2, "PT-102", 11, C_TAG, weight="bold") + _t(352, y_col_bot + 16, "Pot pressure", 12))
    o.append(_t(352, y_col_bot + 32, _fmt(s.p_pot_mbar, "mbar"), 14, C_TAG, weight="bold"))

    # ---- column, insulation, packed bed ---------------------------------------------
    o.append(f'<rect x="{x0 - 17:.1f}" y="{y_col_top:.1f}" width="{Wc + 34:.1f}" height="{Hc:.1f}" '
             f'fill="url(#insul)" stroke="#a89f86"/>')
    o.append(_t(x0 - 24, y_col_top + 70, "Insulation (wrap / jacket)", 12, "#6b6247", "end"))
    o.append(_line(x0 - 17, y_col_top + 66, x0 - 11, y_col_top + 66, C_TAG, 1, "3 3"))
    o.append(_rect(x0, y_col_top, Wc, Hc, "#ffffff", C_GLASS, 1.6))
    o.append(_rect(x0 + 1, y_bed_top, Wc - 2, Hp, "#e9eef3", "none", 0))
    o.append(_rect(x0 + 1, y_bed_top, Wc - 2, Hp, "url(#packing)", "none", 0))
    sat = _clamp(s.saturation, 0.0, 1.0)
    if sat > 0:
        o.append(_rect(x0 + 1, y_bed_top, Wc - 2, Hp, C_LIQ, "none", 0, extra=f' fill-opacity="{0.10 + 0.45 * sat:.2f}"'))
    o.append(_line(x0 + 1, y_bed_top, x1 - 1, y_bed_top, "#c0392b", 2.5, "4 2"))
    o.append(_line(x0 + 1, y_bed_bot, x1 - 1, y_bed_bot, "#c0392b", 3))
    plug_y = max(y_bed_top + 6, y_head_top + 104)          # keep clear of the TT-102 readout
    o.append(_t(20, plug_y, "Retaining plug (top)", 12))
    o.append(_line(140, plug_y - 4, x0, y_bed_top, C_TAG, 1, "3 3"))
    o.append(_t(20, y_bed_bot + 5, "Bed support / constriction", 12))
    o.append(_line(168, y_bed_bot + 1, x0, y_bed_bot, C_TAG, 1, "3 3"))
    mid = (y_bed_top + y_bed_bot) / 2
    pack_txt = (f"stainless wool, {s.packing_size_mm:.2f} mm wire" if wool
                else f"glass beads, {s.packing_size_mm:g} mm")
    o.append(_t(20, mid - 14, "Packed bed", 12, weight="bold"))
    o.append(_t(20, mid, pack_txt, 12))
    o.append(_t(20, mid + 14, f"H_pack {s.H_pack_m:.2f} m, D {s.D_mm:.0f} mm", 12))
    if sat > 0:
        o.append(_t(20, mid + 28, f"liquid fills {sat:.0%} of voids", 12, C_LIQ))
    # vapour up the column, reflux return line
    o.append(_line(Xc + 8, y_col_bot - 2, Xc + 8, y_bed_bot + 4, C_VAP, 2.5, marker="aO"))
    o.append(_line(Xc + 8, y_bed_top - 3, Xc + 8, y_col_top + 8, C_VAP, 2.5, marker="aO"))
    if s.flooded:
        o.append(_rect(x0 - 20, y_col_top - 4, Wc + 40, Hc + 8, "none", "#c0392b", 3, extra=' stroke-dasharray="6 3"'))

    # ---- head: condenser, thermocouple, splitter, pressure tap ---------------------
    o.append(_rect(hx0, y_head_top, hw, head_h, "#ffffff", C_GLASS, 1.6))
    o.append(_path(f"M{hx0 + 8:.1f},{y_head_top + 22:.1f} q6,-9 12,0 t12,0 t12,0 t12,0 t12,0 t8,0", C_CW, 2.5))
    o.append(_t(20, y_head_top + 4, "Head condenser", 12, weight="bold"))
    o.append(_line(115, y_head_top, hx0 + 8, y_head_top + 18, C_TAG, 1, "3 3"))
    o.append(_path(f"M{hx1:.1f},72 L620,72", C_CW, 2.5, marker="aC"))
    o.append(_path(f"M620,100 L{hx1:.1f},100", C_CW, 2.5, marker="aC"))
    o.append(_t(440, 66, "coolant out", 12, C_CW) + _t(440, 94, "coolant in", 12, C_CW))
    tcx = Xc - 16
    o.append(_line(tcx, 40, tcx, y_head_top + head_h - 10, C_TAG, 2))
    o.append(f'<circle cx="{tcx:.1f}" cy="{y_head_top + head_h - 10:.1f}" r="4" fill="#ffffff" stroke="{C_TAG}" stroke-width="1.4"/>')
    o.append(_t(20, y_head_top + 52, "TT-102", 11, C_TAG, weight="bold") + _t(20, y_head_top + 66, "Head vapour T", 12))
    o.append(_t(20, y_head_top + 82, _fmt(s.T_head_c, "\u00b0C"), 14, C_TAG, weight="bold"))
    o.append(_line(100, y_head_top + 54, tcx - 6, y_head_top + head_h - 10, C_TAG, 1, "3 3"))
    # head pressure
    yp = y_head_top + head_h - 4
    o.append(_line(hx1, yp, 470, yp, C_TAG, 1, "3 3"))
    o.append(_sensor(478, yp, "P"))
    o.append(_t(492, yp - 4, "PT-101", 11, C_TAG, weight="bold") + _t(492, yp + 10, "Top pressure, P_top", 12))
    o.append(_t(492, yp + 26, _fmt(s.p_top_mbar, "mbar"), 14, C_TAG, weight="bold"))
    # splitter and the two liquid lines
    sx, sy = hx1 + 26, y_head_top + head_h + 14
    o.append(_path(f"M{hx1:.1f},{y_col_top:.1f} L{sx:.1f},{y_col_top:.1f} L{sx:.1f},{sy - 10:.1f}", C_LIQ, 2.5))
    o.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="10" fill="#ffffff" stroke="{C_TAG}" stroke-width="1.4"/>')
    o.append(_line(sx - 7, sy + 7, sx + 7, sy - 7, C_TAG, 1.6))
    ry = y_col_top + 76
    o.append(_path(f"M{sx:.1f},{sy + 10:.1f} L{sx:.1f},{ry:.1f} L{x1 + 4:.1f},{ry:.1f}", C_LIQ, 2.5, "5 3", "aB"))
    o.append(_t(sx - 36, ry + 18, "reflux return", 12, C_LIQ))
    o.append(_path(f"M{sx + 10:.1f},{sy:.1f} L410,{sy:.1f} L410,178", C_LIQ, 2.5, marker="aB"))
    o.append(_t(sx + 14, sy - 6, "take-off", 12, C_LIQ))
    rtxt = "R = \u2014" if s.R is None else f"R = {s.R:.2f}"
    o.append(_t(422, sy + 14, "RC-101", 11, C_TAG, weight="bold") + _t(422, sy + 27, "Reflux splitter", 12))
    o.append(_t(422, sy + 40, rtxt, 14, C_TAG, weight="bold"))

    # ---- product cooler, coolant bath ---------------------------------------------
    o.append('<g transform="rotate(14 400 200)">'
             + _rect(390, 181, 210, 38, "#ffffff", C_GLASS, 1.6, rx=6)
             + _path("M400,200 q10,-16 20,0 t20,0 t20,0 t20,0 t20,0 t20,0 t20,0 t20,0 t20,0", C_CW, 2.5) + "</g>")
    o.append(_t(400, 272, "Product cooler (inclined)", 12, weight="bold"))
    o.append(_rect(620, 55, 140, 60, "#ffffff", C_GLASS, 1.2))
    o.append(_t(690, 80, "Circulating bath", 12, anchor="middle") + _t(690, 96, "(T-controlled, feeds both", 12, anchor="middle")
             + _t(690, 110, "condensers)", 11, anchor="middle"))
    o.append(_path("M740,115 L740,262 L608,262", C_CW, 2.5, marker="aC"))
    o.append(_t(640, 280, "coolant IN (lower end)", 12, C_CW))
    o.append(_path("M540,212 L540,190 L660,190 L660,115", C_CW, 2.5, marker="aC"))
    o.append(_t(666, 150, "coolant OUT", 12, C_CW) + _t(666, 164, "(upper end)", 12, C_CW))

    # ---- receivers -------------------------------------------------------------------
    r = min(38.0, 150.0 / n_cut)
    gap = 8.0
    total_w = n_cut * (2 * r + gap) - gap
    xs = [615.0 - total_w / 2 + r + i * (2 * r + gap) for i in range(n_cut)]
    adapter_x0, adapter_x1 = xs[0] - r - 10, xs[-1] + r + 10
    o.append(_path("M595,258 L595,322", C_LIQ, 2.5, marker="aB"))
    o.append(_rect(adapter_x0, 322, adapter_x1 - adapter_x0, 22, "#ffffff", C_GLASS, 1.6))
    o.append(_t((adapter_x0 + adapter_x1) / 2, 338, 'Multi-flask receiver ("cow" adapter)', 12, anchor="middle"))
    masses = list(s.cut_masses_g[:n_cut]) + [0.0] * (n_cut - len(s.cut_masses_g))
    ref = s.fill_ref_g if s.fill_ref_g else (max(masses) if max(masses) > 0 else 1.0)
    fy = 410.0 if r > 30 else 400.0
    for i, cx in enumerate(xs):
        act = s.active_cut == i
        o.append(_line(cx, 344, cx, fy - r, C_GLASS, 1.6))
        o.append(f'<circle cx="{cx - 0:.1f}" cy="358" r="6" fill="#ffffff" stroke="{C_GLASS}"/>')
        frac = _clamp(masses[i] / ref, 0.0, 0.92)
        lvl = fy + r - 2 * r * frac
        o.append(f'<clipPath id="fl{i}"><circle cx="{cx:.1f}" cy="{fy:.1f}" r="{r - 1:.1f}"/></clipPath>')
        o.append(f'<circle cx="{cx:.1f}" cy="{fy:.1f}" r="{r:.1f}" fill="#ffffff" stroke="{C_LIQ if act else C_GLASS}" '
                 f'stroke-width="{3 if act else 1.6}"/>')
        if frac > 0:
            o.append(f'<rect x="{cx - r:.1f}" y="{lvl:.1f}" width="{2 * r:.1f}" height="{fy + r + 2 - lvl:.1f}" '
                     f'fill="#a5c8ea" clip-path="url(#fl{i})"/>')
        lab = s.cut_labels[i] if i < len(s.cut_labels) else f"Cut {i + 1}"
        o.append(_t(cx, fy + r + 16, lab, 11, C_LIQ if act else C_TEXT, "middle", "bold" if act else "normal"))
        o.append(_t(cx, fy + r + 30, f"{masses[i]:.0f} g", 12, C_TAG, "middle", "bold"))
    o.append(_t(612, 298, "isolation valves (swap cuts", 12) + _t(612, 311, "without breaking vacuum)", 12))

    # ---- vacuum train ------------------------------------------------------------------
    vx = max(790.0, adapter_x1 + 30)
    o.append(_path(f"M{adapter_x1:.1f},333 L{vx:.1f},333 L{vx:.1f},520 L{vx + 30:.1f},520", C_VAC, 2.5, "7 4", "aG"))
    o.append(_rect(vx + 30, 490, 110, 60, "#ffffff", C_GLASS, 1.2))
    o.append(_t(vx + 85, 515, "Cold trap", 12, anchor="middle") + _t(vx + 85, 531, "(light ends)", 12, anchor="middle"))
    o.append(_path(f"M{vx + 140:.1f},520 L{vx + 170:.1f},520", C_VAC, 2.5, "7 4", "aG"))
    o.append(_rect(vx + 170, 495, 100 if vx + 270 <= 1090 else 1090 - vx - 170, 50, "#ffffff", C_GLASS, 1.2))
    o.append(_t(vx + 215, 516, "Vacuum", 12, anchor="middle") + _t(vx + 215, 532, "pump", 12, anchor="middle"))
    o.append(f'<circle cx="{vx:.1f}" cy="440" r="11" fill="#ffffff" stroke="{C_GLASS}"/>')
    o.append(_t(vx, 444, "V", 11, anchor="middle"))
    o.append(_t(vx + 18, 436, "Bleed / vent valve", 12) + _t(vx + 18, 450, "(vent when at atm.)", 12))

    # ---- status: flooding gauge and operating labels -----------------------------------
    gx, gy, gw = 790.0, 20.0, 280.0
    o.append(_t(gx, gy + 10, "Flooding margin", 12, weight="bold"))
    for a, b, col in ((0, 70, "#7bc67b"), (70, 85, "#f0c75e"), (85, 120, "#e07b6b")):
        o.append(_rect(gx + gw * a / 120, gy + 16, gw * (b - a) / 120, 12, col, "none", 0))
    pf = None if s.percent_flood is None else s.percent_flood
    if s.flooded:
        o.append(_rect(gx, gy + 34, gw, 22, "#c0392b", "none", 0, rx=3))
        o.append(_t(gx + gw / 2, gy + 50, "FLOODED - no stable operating point", 12, "#ffffff", "middle", "bold"))
    elif pf is not None:
        mx = gx + gw * _clamp(pf, 0, 120) / 120
        o.append(f'<polygon points="{mx - 6:.1f},{gy + 14:.1f} {mx + 6:.1f},{gy + 14:.1f} {mx:.1f},{gy + 24:.1f}" '
                 f'fill="{C_TEXT}"/>')
        o.append(_t(gx, gy + 46, f"{pf:.0f} % of flooding", 12))
    if s.time_min is not None:
        o.append(_t(gx + gw, gy + 74, f"t = {s.time_min:.0f} min", 12, C_TAG, "end", "bold"))
    if s.light_ends:                                      # warning under the gauge, clear of the head
        ty = gy + 74
        o.append(f'<polygon points="{gx:.1f},{ty + 10:.1f} {gx + 9:.1f},{ty - 6:.1f} {gx + 18:.1f},{ty + 10:.1f}" '
                 f'fill="#f0c75e" stroke="{C_TAG}"/>')
        o.append(_t(gx + 9, ty + 8, "!", 11, C_TAG, "middle", "bold"))
        o.append(_t(gx + 26, ty + 7, "head colder than coolant", 11, C_TAG))

    # ---- legend -----------------------------------------------------------------------
    ly = H_canvas - 110
    o.append(_rect(780, ly, 300, 100, "#fafafa", C_GLASS, 1.2))
    o.append(_t(790, ly + 18, "Legend", 12, weight="bold"))
    for row, (col, txt, dash) in enumerate(((C_VAP, "Vapour", None), (C_LIQ, "Liquid (reflux dashed / distillate)", None),
                                            (C_CW, "Coolant", None), (C_VAC, "Vacuum line", "7 4"))):
        yy = ly + 32 + 16 * row
        o.append(_line(790, yy, 830, yy, col, 2.5, dash) + _t(838, yy + 4, txt, 12))
    o.append(_line(940, ly + 32, 980, ly + 32, C_TAG, 1, "3 3") + _t(988, ly + 36, "Signal", 12))
    o.append(_t(20, H_canvas - 12, "Schematic, not to scale except column height, bed height and diameter.", 11, "#555555", italic=True))
    o.append("</svg>")
    return "".join(o)


# --------------------------------------------------------------------------
# Building a state from engine results
# --------------------------------------------------------------------------

def _cut_label(c) -> str:
    return f"{c.T_lo_atm_c:.0f}\u2013{c.T_hi_atm_c:.0f} \u00b0C"


def state_from_design(d, **overrides) -> SchematicState:
    """The designed column at rest (``fdc.design_mode.DesignResult``)."""
    g = d.geom
    base = dict(D_mm=d.D_mm, H_pack_m=d.H_pack_m, H_col_m=d.H_col_m, packing_kind=g.kind,
                packing_size_mm=g.size_m * 1000.0, R=d.R, q_heater_w=d.q_heater_w, p_top_mbar=d.p_top_mbar,
                percent_flood=d.percent_flood, saturation=0.0, pot_fill=0.45)
    if d.verification is not None:
        base["cut_labels"] = tuple(_cut_label(c) for c in d.verification.cuts)
        base["cut_masses_g"] = tuple(c.mass_kg * 1000.0 for c in d.verification.cuts)
    base.update(overrides)
    return SchematicState(**base)


def state_from_batch(res, geom, spec, step: int, percent_flood: float | None = None,
                     saturation: float = 0.0, pot_fill_start: float = 0.45) -> SchematicState:
    """The schematic at one step of a simulated batch (``fdc.batch_pot.BatchResult``).

    Cut masses are rebuilt from the history (each step's increment belongs to the
    cut being collected at that step), and the pot level follows the mass left
    in the pot.  Hold-up shading and the flooding margin are static values you
    pass in (they come from the rating-mode hydraulics).
    """
    i = int(_clamp(step, 0, len(res.t_min) - 1))
    inc = np.diff(res.distillate_kg, prepend=0.0)
    ci = res.cut_index
    n_cut = len(res.cuts)
    masses = tuple(float(inc[: i + 1][ci[: i + 1] == c].sum()) * 1000.0 for c in range(n_cut))
    m_pot = spec.m_feed_kg - res.holdup_kg - float(res.distillate_kg[i])
    pot_fill = pot_fill_start * _clamp(m_pot / spec.m_feed_kg, 0.0, 1.0)
    final = i == len(res.t_min) - 1 and res.stop_reason == "target range complete"
    return SchematicState(
        D_mm=geom.D_m * 1000.0, H_pack_m=geom.H_pack_m, H_col_m=geom.H_col_m, packing_kind=geom.kind,
        packing_size_mm=geom.size_m * 1000.0, R=spec.R, q_heater_w=spec.q_heater_w, p_top_mbar=spec.p_top_mbar,
        T_pot_c=float(res.T_pot_c[i]), T_head_c=float(res.T_head_c[i]), p_pot_mbar=float(res.p_pot_mbar[i]),
        time_min=float(res.t_min[i]), saturation=saturation, percent_flood=percent_flood,
        flooded=res.stop_reason == "flooded" and i == len(res.t_min) - 1, pot_fill=pot_fill,
        cut_labels=tuple(_cut_label(c) for c in res.cuts), cut_masses_g=masses,
        active_cut=None if final else int(ci[i]), light_ends=float(res.T_head_c[i]) < spec.T_condenser_c)


# --------------------------------------------------------------------------
# Demo:  python -m app.schematic   (writes a few example states to ./schematic_demo/)
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import warnings
    from pathlib import Path

    from fdc.batch_pot import BatchSpec, simulate_batch
    from fdc.packing import PackingSpec, column_geometry
    from fdc.properties.feed import DEFAULT_WPPO_TBP, TBPCurve, characterise_feed

    warnings.simplefilter("ignore")
    out = Path("schematic_demo")
    out.mkdir(exist_ok=True)

    feed = characterise_feed(TBPCurve.from_points(DEFAULT_WPPO_TBP), n_cuts=30, sg_feed=0.80)
    geom = column_geometry(PackingSpec("glass_beads", 5.0), D_mm=50.0, H_pack_m=0.60)
    spec = BatchSpec(m_feed_kg=1.0, p_top_mbar=100.0, R=5.0, q_heater_w=100.0, cut_edges_c=(150.0, 250.0))
    res = simulate_batch(feed, geom, spec)

    states = {
        "default": SchematicState(),
        "mid_run": state_from_batch(res, geom, spec, len(res.t_min) // 2, percent_flood=45.0, saturation=0.35),
        "end_run": state_from_batch(res, geom, spec, len(res.t_min) - 1, percent_flood=45.0, saturation=0.35),
        "tall_wide_wool": SchematicState(D_mm=100.0, H_pack_m=1.6, H_col_m=1.75, packing_kind="steel_wool",
                                         packing_size_mm=0.05, p_top_mbar=50.0, T_pot_c=88.0, T_head_c=61.0,
                                         percent_flood=62.0, saturation=0.2, cut_labels=("A", "B", "C", "D"),
                                         cut_masses_g=(40, 120, 80, 0), active_cut=3, time_min=95),
        "flooded": SchematicState(percent_flood=130.0, flooded=True, T_pot_c=120.0, T_head_c=70.0,
                                  saturation=0.8, time_min=30),
    }
    for name, st in states.items():
        (out / f"{name}.svg").write_text(render_schematic(st), encoding="utf-8")
    print(f"wrote {len(states)} example schematics to {out}/")
