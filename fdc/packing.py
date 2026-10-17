"""
fdc.packing
===========

Geometry of the packed section: void fraction, specific surface area, bulk
density, packing mass, free volume and overall column height.

Corr-map position
-----------------
Drivers : V07 (column diameter D), V08 (packing type and size),
          V09 (packing height H_pack)
Solves  : V14 (void fraction and bulk density), V15 (mass of packing),
          V21 (column height).  Mostly exact geometry ('E' in the corr map);
          the only correlation is the wall effect on the void fraction of beads.

Two packings are supported
--------------------------
* ``glass_beads`` : spheres of diameter d_p.  Void fraction rises when the
  column is narrow compared with the beads (wall effect); Dixon's (1988)
  correlation is used.  Rule of thumb: D/d_p of about 8-10 or more is needed
  to avoid wall channelling.
* ``steel_wool``  : wire of diameter d_w, compressible, so the void fraction is
  *set by how much wool you load*.  Give the void fraction, the bulk density,
  or (better) weigh the wool you pack and use ``void_fraction_from_mass``.

Conventions
-----------
* Public inputs: column diameter ``D_mm`` and packing size ``size_mm`` in mm,
  heights in m.  Everything inside and in the results is SI (m, kg, m2/m3).
* ``d_equiv_m`` is the sphere-equivalent diameter d = 6 (1 - eps) / a, the
  quantity Ergun-type pressure-drop and hold-up correlations in
  ``hydraulics.py`` / ``holdup.py`` expect (for beads it equals the bead
  diameter, for wire it is 1.5 x the wire diameter).

Main references
---------------
* Dixon (1988), "Wall and particle-shape effects on heat transfer in packed
  beds", Chem. Eng. Commun. 71, 217-237 (void fraction vs D/d_p).
* Perry's Chemical Engineers' Handbook, packed-bed section (area, hydraulic
  diameter relations).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Literal

# --------------------------------------------------------------------------
# Constants and defaults
# --------------------------------------------------------------------------

# Solid densities [kg/m3].  Soda-lime glass; borosilicate is nearer 2230, so
# override ``material_density`` if your beads are borosilicate.
DENSITY_GLASS = 2500.0
DENSITY_STEEL = 7850.0          # carbon / low-alloy steel; stainless is ~7900-8000

# Void fraction of randomly poured, wall-free monodisperse spheres.
EPS_BEADS_INF = 0.40

# ASSUMPTION: default solid fraction of a lightly packed steel-wool plug
# (void fraction 0.96).  Real values depend entirely on how it is packed.
EPS_WOOL_DEFAULT = 0.96

# Wall-effect and bed-length guard rails (rules of thumb, not hard limits)
WALL_RATIO_MIN = 8.0            # D / d_p below this -> wall channelling likely
BED_LENGTH_MIN = 10.0           # H_pack / d_p below this -> end effects dominate

# ASSUMPTION: unpacked height above and below the bed [m].
H_TOP_DEFAULT = 0.10            # plug, reflux return point, thermowell zone
H_BOTTOM_DEFAULT = 0.05         # bed support and vapour entry from the pot neck


# --------------------------------------------------------------------------
# Packing specification
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PackingSpec:
    """What the packing is made of, before it goes into a particular column.

    kind             : ``"glass_beads"`` or ``"steel_wool"``.
    size_mm          : bead diameter, or wire diameter for steel wool [mm].
    material_density : solid density [kg/m3]; defaults to glass / steel.
    void_fraction    : optional measured void fraction, overriding the model.
    bulk_density     : optional packed bulk density [kg/m3]; the void fraction
                       follows as 1 - bulk_density / material_density.
    Give at most one of ``void_fraction`` and ``bulk_density``.
    """

    kind: Literal["glass_beads", "steel_wool"]
    size_mm: float
    material_density: float | None = None
    void_fraction: float | None = None
    bulk_density: float | None = None

    def __post_init__(self):
        if self.kind not in ("glass_beads", "steel_wool"):
            raise ValueError("kind must be 'glass_beads' or 'steel_wool'.")
        if self.size_mm <= 0:
            raise ValueError("Packing size must be positive.")
        if self.void_fraction is not None and self.bulk_density is not None:
            raise ValueError("Give either void_fraction or bulk_density, not both.")
        if self.void_fraction is not None and not 0.0 < self.void_fraction < 1.0:
            raise ValueError("void_fraction must lie between 0 and 1.")

    @property
    def rho_solid(self) -> float:
        if self.material_density is not None:
            return self.material_density
        return DENSITY_GLASS if self.kind == "glass_beads" else DENSITY_STEEL


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def bead_void_fraction(D_mm: float, d_p_mm: float, eps_inf: float = EPS_BEADS_INF) -> float:
    """Average void fraction of a random bed of spheres in a cylinder.

    Dixon (1988):  eps = eps_inf + 0.05 (d_p/D) + 0.412 (d_p/D)^2, valid for
    d_p/D <= 0.5.  The wall makes the bed looser, so a narrow column has a
    higher void fraction than the wall-free value ``eps_inf``.
    """
    r = d_p_mm / D_mm
    if r > 0.5:
        raise ValueError("Beads larger than half the column diameter: correlation invalid.")
    return eps_inf + 0.05 * r + 0.412 * r ** 2


def void_fraction_from_mass(m_pack_kg: float, material_density: float,
                            D_mm: float, H_pack_m: float) -> float:
    """Void fraction from the mass of packing you actually loaded.

    eps = 1 - m / (rho_solid * V_bed).  This is the practical route for steel
    wool: weigh it, push it into the column, measure the bed height.
    """
    V_bed = math.pi * (D_mm / 1000.0) ** 2 / 4.0 * H_pack_m
    eps = 1.0 - m_pack_kg / (material_density * V_bed)
    if not 0.0 < eps < 1.0:
        raise ValueError("Loaded mass is inconsistent with the bed volume (void fraction "
                         f"would be {eps:.3f}).")
    return eps


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ColumnGeometry:
    """Geometry of the packed column for given packing, diameter and height."""

    # inputs echoed back
    kind: str
    D_m: float             # inner diameter [m]
    H_pack_m: float        # packed height [m]
    size_m: float          # bead diameter / wire diameter [m]
    # packing properties in this column  (V14)
    eps: float             # void fraction [-]
    a_m2_m3: float         # specific surface area of packing [m2/m3 of bed]
    d_equiv_m: float       # sphere-equivalent diameter 6(1-eps)/a [m]
    rho_bulk: float        # packed bulk density [kg/m3]
    wall_ratio: float      # D / d_p (nan for steel wool)
    # bed and column quantities
    area_m2: float         # empty-column cross-section [m2]
    V_bed_m3: float        # packed volume [m3]
    V_free_m3: float       # void volume inside the bed [m3]
    m_pack_kg: float       # mass of packing (V15)
    H_col_m: float         # overall column height (V21)
    l_over_d: float        # H_pack / D [-]


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------

def column_geometry(
    spec: PackingSpec,
    D_mm: float,
    H_pack_m: float,
    h_top_m: float = H_TOP_DEFAULT,
    h_bottom_m: float = H_BOTTOM_DEFAULT,
) -> ColumnGeometry:
    """Void fraction, area, bulk density, packing mass and column height.

    Parameters
    ----------
    spec       : PackingSpec of the packing.
    D_mm       : column inner diameter [mm].
    H_pack_m   : height of the packed section [m].
    h_top_m, h_bottom_m : ASSUMPTION - unpacked allowances above and below the
                 bed [m]; ``H_col = H_pack + h_top + h_bottom``.
    """
    if D_mm <= 0 or H_pack_m <= 0:
        raise ValueError("Diameter and packed height must be positive.")
    if h_top_m < 0 or h_bottom_m < 0:
        raise ValueError("Allowances cannot be negative.")

    D = D_mm / 1000.0
    size = spec.size_mm / 1000.0
    rho_s = spec.rho_solid

    # ---- void fraction ------------------------------------------------------
    if spec.void_fraction is not None:
        eps = spec.void_fraction                         # measured: trust it
    elif spec.bulk_density is not None:
        eps = 1.0 - spec.bulk_density / rho_s            # from loaded density
        if not 0.0 < eps < 1.0:
            raise ValueError("bulk_density is not below the solid density.")
    elif spec.kind == "glass_beads":
        eps = bead_void_fraction(D_mm, spec.size_mm)     # Dixon wall-effect model
    else:
        eps = EPS_WOOL_DEFAULT                           # assumption, see top of file

    # ---- specific surface area and equivalent diameter ----------------------
    # spheres: a = 6(1-eps)/d_p ; long cylinders (wire): a = 4(1-eps)/d_w
    if spec.kind == "glass_beads":
        a = 6.0 * (1.0 - eps) / size
        wall_ratio = D / size
    else:
        a = 4.0 * (1.0 - eps) / size
        wall_ratio = float("nan")                        # compressible plug: not meaningful
    d_equiv = 6.0 * (1.0 - eps) / a

    # ---- bed quantities (exact geometry) ------------------------------------
    area = math.pi * D ** 2 / 4.0
    V_bed = area * H_pack_m
    rho_bulk = (1.0 - eps) * rho_s
    m_pack = rho_bulk * V_bed

    # ---- guard rails ----------------------------------------------------------
    if spec.kind == "glass_beads":
        if wall_ratio < WALL_RATIO_MIN:
            warnings.warn(f"D/d_p = {wall_ratio:.1f} is below ~{WALL_RATIO_MIN:.0f}: wall "
                          "channelling will degrade separation.", stacklevel=2)
        if not 0.33 <= eps <= 0.60:
            warnings.warn(f"Void fraction {eps:.2f} is unusual for a bed of spheres.", stacklevel=2)
    if H_pack_m / size < BED_LENGTH_MIN:
        warnings.warn(f"Bed is only {H_pack_m / size:.0f} packing diameters long; end "
                      "effects will dominate.", stacklevel=2)

    return ColumnGeometry(
        kind=spec.kind, D_m=D, H_pack_m=H_pack_m, size_m=size,
        eps=eps, a_m2_m3=a, d_equiv_m=d_equiv, rho_bulk=rho_bulk, wall_ratio=wall_ratio,
        area_m2=area, V_bed_m3=V_bed, V_free_m3=eps * V_bed, m_pack_kg=m_pack,
        H_col_m=H_pack_m + h_top_m + h_bottom_m, l_over_d=H_pack_m / D,
    )


# --------------------------------------------------------------------------
# Demo / smoke test:  python -m fdc.packing
# --------------------------------------------------------------------------

if __name__ == "__main__":
    beads = PackingSpec("glass_beads", size_mm=3.0)
    wool = PackingSpec("steel_wool", size_mm=0.05)          # 50 micron wire

    for label, spec, D in (("glass beads 3 mm", beads, 30.0), ("steel wool 0.05 mm wire", wool, 30.0)):
        g = column_geometry(spec, D_mm=D, H_pack_m=0.30)
        print(f"{label}, D = {D:.0f} mm, H_pack = 0.30 m")
        print(f"  void fraction {g.eps:.3f} | area {g.a_m2_m3:.0f} m2/m3 | d_equiv {g.d_equiv_m * 1000:.3f} mm "
              f"| bulk density {g.rho_bulk:.0f} kg/m3")
        print(f"  bed volume {g.V_bed_m3 * 1e6:.0f} mL | free volume {g.V_free_m3 * 1e6:.0f} mL "
              f"| packing mass {g.m_pack_kg * 1000:.1f} g | column height {g.H_col_m:.2f} m "
              f"| L/D {g.l_over_d:.1f}\n")
