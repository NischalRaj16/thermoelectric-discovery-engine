"""
composition.py — multi-dopant composition encoding for the Thermoelectric Discovery Engine.

Replaces the two-slot parser in evaluate_uploaded_datasets.parse_composition_to_krict().

Why this module exists
----------------------
The original encoder reduced any formula to four numbers (Sn_ratio, anion_ratio and two
dopant slots) and clipped both ratios into [0.5, 1.0]. Three consequences were measured
on the shipped 870-point experimental set:

  * Sn0.96Na0.01Ag0.03S encoded to Na(0.01) + S(1.00): the Ag content - the variable the
    Science 2020 benchmark is about - was discarded, because the host sulfur itself
    consumed the second slot.
  * 147 distinct formulas produced only 111 distinct input vectors. Eight
    (SnTe)x(CuSbTe2)y alloys collapsed onto a single vector and therefore onto a single
    prediction, although their measured ZT differs.
  * 55.7% of rows had their true Sn fraction clipped upward to 0.5.

The 15 base boosters are frozen at 62 inputs and their training set is not in the
repository, so they cannot be retrained. This module therefore does two things:

  BASE PROJECTION  - produces the legacy row the frozen base models require, while
                     recording what had to be thrown away (clipping, dropped dopants).
  TL FEATURES      - produces a full multi-dopant vector for the transfer-learning
                     layer, which IS retrainable. That layer is what resolves the
                     collisions the base representation cannot express.

Nothing here silently discards chemistry: whatever cannot be carried into the base
representation is reported in `ParsedComposition.base_lost`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from elements import ELEMENTS
except ImportError:  # pragma: no cover - resolved via sys.path in the pipeline
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "1_predicting_new_compositions"))
    from elements import ELEMENTS

# Elements treated as chalcogen / halogen anion-sublattice species.
ANION_SPECIES = ("S", "Se", "Te", "O", "Cl", "Br", "I")
HOST_CATION = "Sn"
HOST_ANION_DEFAULT = "Se"

# Dedicated slots that feature2.construct_feature() recognises on the base side.
NA_Z, CL_Z = 11, 17

# How many explicit dopant slots the transfer-learning feature vector carries.
# Beyond this the remainder is summarised by the aggregate descriptors, so no dopant
# is ever silently dropped from the TL representation.
TL_DOPANT_SLOTS = 6

_TOKEN = re.compile(r"([A-Z][a-z]?)\s*([0-9]*\.?[0-9]*)")
_GROUP = re.compile(r"\(([^()]*)\)\s*([0-9]*\.?[0-9]*)")


# --------------------------------------------------------------------------- parsing
def _accumulate(fragment: str, multiplier: float, into: Dict[str, float]) -> None:
    for sym, amt in _TOKEN.findall(fragment):
        if not sym:
            continue
        try:
            ELEMENTS[sym]
        except Exception:
            continue
        into[sym] = into.get(sym, 0.0) + (float(amt) if amt else 1.0) * multiplier


def parse_formula_to_counts(formula: str) -> Dict[str, float]:
    """Expand a formula, including one level of parenthesised groups, to atom counts.

    Handles the forms that actually occur in the benchmark sets, e.g.
    'Na0.01(Sn0.95Te0.03)0.99S', '(SnTe)0.9(CuSbTe2)0.1', '(SnTe)10Sb2Te3'.
    """
    text = str(formula).replace(" ", "").replace("[", "(").replace("]", ")")
    counts: Dict[str, float] = {}

    remainder = text
    for group, mult in _GROUP.findall(text):
        m = float(mult) if mult else 1.0
        _accumulate(group, m, counts)
    remainder = _GROUP.sub("", text)
    _accumulate(remainder, 1.0, counts)
    return {k: v for k, v in counts.items() if v > 0}


@dataclass
class Dopant:
    symbol: str
    z: int
    fraction: float          # fraction of its own sublattice
    site: str                # 'cation' | 'anion'

    @property
    def is_anion_site(self) -> int:
        return 1 if self.site == "anion" else 0


@dataclass
class ParsedComposition:
    formula: str
    counts: Dict[str, float]
    cation_frac: Dict[str, float]
    anion_frac: Dict[str, float]
    host_anion: str
    sn_fraction: float                      # true cation-sublattice Sn fraction
    host_anion_fraction: float              # true anion-sublattice host fraction
    dopants: List[Dopant] = field(default_factory=list)
    base_lost: List[str] = field(default_factory=list)
    ok: bool = True
    reason: str = ""

    # ----------------------------------------------------------------- convenience
    @property
    def cation_dopants(self) -> List[Dopant]:
        return [d for d in self.dopants if d.site == "cation"]

    @property
    def anion_dopants(self) -> List[Dopant]:
        return [d for d in self.dopants if d.site == "anion"]

    @property
    def n_dopants(self) -> int:
        return len(self.dopants)


def parse_composition(formula: str) -> ParsedComposition:
    """Parse a formula into cation / anion sublattices with an unlimited dopant list."""
    counts = parse_formula_to_counts(formula)
    if not counts:
        return ParsedComposition(formula, {}, {}, {}, HOST_ANION_DEFAULT, 0.0, 0.0,
                                 ok=False, reason="no recognisable elements")

    cations = {e: a for e, a in counts.items() if e not in ANION_SPECIES or e == HOST_CATION}
    anions = {e: a for e, a in counts.items() if e in ANION_SPECIES and e != HOST_CATION}

    tc, ta = sum(cations.values()), sum(anions.values())
    if tc <= 0 or ta <= 0:
        return ParsedComposition(formula, counts, {}, {}, HOST_ANION_DEFAULT, 0.0, 0.0,
                                 ok=False, reason="missing a cation or anion sublattice")

    cation_frac = {e: a / tc for e, a in cations.items()}
    anion_frac = {e: a / ta for e, a in anions.items()}

    # The host anion is the most abundant chalcogen/halogen, not a hard-coded Se.
    host_anion = max(anion_frac, key=anion_frac.get)

    dopants: List[Dopant] = []
    for e, f in cation_frac.items():
        if e == HOST_CATION or f <= 1e-6:
            continue
        dopants.append(Dopant(e, int(ELEMENTS[e].number), float(f), "cation"))
    for e, f in anion_frac.items():
        if e == host_anion or f <= 1e-6:
            continue
        dopants.append(Dopant(e, int(ELEMENTS[e].number), float(f), "anion"))
    dopants.sort(key=lambda d: d.fraction, reverse=True)

    return ParsedComposition(
        formula=str(formula),
        counts=counts,
        cation_frac=cation_frac,
        anion_frac=anion_frac,
        host_anion=host_anion,
        sn_fraction=float(cation_frac.get(HOST_CATION, 0.0)),
        host_anion_fraction=float(anion_frac[host_anion]),
        dopants=dopants,
    )


# ------------------------------------------------------------------ base projection
def to_base_row(p: ParsedComposition, temperature: float, direction: int = 0,
                numrow: int = 0, row_id: str = "C0000") -> Optional[list]:
    """Project onto the legacy row the frozen 62-feature base models require.

    feature2.construct_feature() offers a dedicated Na slot, a dedicated Cl slot and a
    single generic dopant slot. Everything else has to be dropped, so what is dropped
    is recorded on `p.base_lost` and surfaces as an out-of-domain flag at prediction
    time rather than disappearing.
    """
    if not p.ok:
        return None

    p.base_lost = []

    special = [d for d in p.dopants if d.z in (NA_Z, CL_Z)]
    generic_pool = [d for d in p.dopants if d.z not in (NA_Z, CL_Z)]

    # The host anion is itself a substitution on the Se sublattice as far as the base
    # DFT table is concerned, so a non-Se host occupies the generic slot - that is how
    # the frozen models are told the anion is S or Te.
    if p.host_anion != HOST_ANION_DEFAULT:
        # Cap the substitution ratio for the base projection at the supercell DFT ratio (0.03125)
        # to prevent unphysical 32x linear extrapolation in feature2.construct_feature().
        # The true 100% composition is preserved and carried in TL features.
        capped_ratio = min(float(p.host_anion_fraction), 0.03125)
        host_sub = Dopant(p.host_anion, int(ELEMENTS[p.host_anion].number),
                          capped_ratio, "anion")
        generic_pool.insert(0, host_sub)
        if float(p.host_anion_fraction) > 0.03125:
            p.base_lost.append(f"host anion substitution {p.host_anion} capped {p.host_anion_fraction:.3f}->0.031 for base DFT table")

    d1_num = d1_ratio = d2_num = d2_ratio = 0.0
    if special:
        d1_num, d1_ratio = special[0].z, special[0].fraction
        if len(special) > 1:
            p.base_lost.append(f"{special[1].symbol}({special[1].fraction:.3f})")
        if generic_pool:
            d2_num, d2_ratio = generic_pool[0].z, generic_pool[0].fraction
            for d in generic_pool[1:]:
                p.base_lost.append(f"{d.symbol}({d.fraction:.3f})")
    else:
        if generic_pool:
            d1_num, d1_ratio = generic_pool[0].z, generic_pool[0].fraction
        if len(generic_pool) > 1:
            d2_num, d2_ratio = generic_pool[1].z, generic_pool[1].fraction
        for d in generic_pool[2:]:
            p.base_lost.append(f"{d.symbol}({d.fraction:.3f})")

    sn_clipped = float(np.clip(p.sn_fraction, 0.5, 1.0))
    an_clipped = float(np.clip(p.host_anion_fraction, 0.5, 1.0))
    if abs(sn_clipped - p.sn_fraction) > 1e-6:
        p.base_lost.append(f"Sn fraction clipped {p.sn_fraction:.3f}->{sn_clipped:.3f}")
    if abs(an_clipped - p.host_anion_fraction) > 1e-6:
        p.base_lost.append(f"anion fraction clipped {p.host_anion_fraction:.3f}->{an_clipped:.3f}")

    row = [numrow, row_id, sn_clipped, an_clipped,
           int(d1_num), round(float(d1_ratio), 4),
           int(d2_num), round(float(d2_ratio), 4),
           0, 0, int(direction), float(temperature)]

    # category mirrors util.define_category: 1 for Na-doped, 2 for Cl-doped, else 0.
    category = 1 if int(d1_num) == NA_Z else (2 if int(d1_num) == CL_Z else 0)
    row.append(category)
    return row


BASE_ROW_COLUMNS = [
    "numrow", "ID", "Sn_ratio", "Se_ratio", "dopant1_num", "dopant1_ratio",
    "dopant2_num", "dopant2_ratio", "dopant3_num", "dopant3_ratio",
    "direction", "Temp", "category",
]


# -------------------------------------------------------------------- TL features
def _prop(sym: str, attr: str, default: float = 0.0) -> float:
    try:
        v = getattr(ELEMENTS[sym], attr)
        return float(v) if v is not None else default
    except Exception:
        return default


TL_FEATURE_NAMES: List[str] = (
    ["sn_fraction_true", "host_anion_Z", "host_anion_fraction_true",
     "n_cation_dopants", "n_anion_dopants", "n_dopants_total",
     "cation_dopant_load", "anion_dopant_load",
     "dopant_mean_Z", "dopant_mean_mass", "dopant_mass_variance",
     "dopant_mean_eneg", "dopant_eneg_contrast",
     "dopant_mean_covrad", "dopant_covrad_mismatch",
     "dopant_mean_valence_period", "host_mass_contrast",
     "direction_val", "is_single_crystal", "is_b_axis", "is_c_axis", "is_a_axis"]
    + [f"slot{i}_{k}" for i in range(1, TL_DOPANT_SLOTS + 1)
       for k in ("Z", "frac", "anion_site", "mass_ratio", "eneg_diff")]
)


def to_tl_features(p: ParsedComposition, direction: int = 0) -> np.ndarray:
    """Full multi-dopant descriptor for the retrainable transfer-learning layer.

    Explicit slots carry the TL_DOPANT_SLOTS largest dopants; the aggregate descriptors
    are computed over ALL dopants, so a ninth dopant still changes the vector.
    `direction` encodes crystal measurement direction (0=poly, 1=a, 2=b, 3=c).
    """
    if not p.ok:
        return np.zeros(len(TL_FEATURE_NAMES), dtype=float)

    host_an = p.host_anion
    host_an_mass = _prop(host_an, "mass", 1.0)
    host_an_eneg = _prop(host_an, "eleneg")
    sn_mass = _prop(HOST_CATION, "mass", 1.0)
    sn_covrad = _prop(HOST_CATION, "covrad", 1.0)
    sn_eneg = _prop(HOST_CATION, "eleneg")

    cat_d = p.cation_dopants
    an_d = p.anion_dopants
    alld = p.dopants

    w = np.array([d.fraction for d in alld], dtype=float)
    wsum = float(w.sum()) if w.size else 0.0

    def wavg(vals: List[float]) -> float:
        if not vals or wsum <= 0:
            return 0.0
        return float(np.dot(w, np.array(vals, dtype=float)) / wsum)

    masses = [_prop(d.symbol, "mass") for d in alld]
    enegs = [_prop(d.symbol, "eleneg") for d in alld]
    covs = [_prop(d.symbol, "covrad") for d in alld]
    periods = [_prop(d.symbol, "period") for d in alld]

    mean_mass = wavg(masses)
    mass_var = 0.0
    if alld and wsum > 0:
        mass_var = float(np.dot(w, (np.array(masses) - mean_mass) ** 2) / wsum)

    # Mass contrast against the sublattice each dopant actually substitutes - this is
    # the quantity that drives mass-contrast phonon scattering.
    host_contrast = 0.0
    if alld and wsum > 0:
        ref = np.array([host_an_mass if d.site == "anion" else sn_mass for d in alld])
        host_contrast = float(np.dot(w, np.abs(np.array(masses) - ref) / np.maximum(ref, 1e-6)) / wsum)

    eneg_contrast = 0.0
    if alld and wsum > 0:
        ref_e = np.array([host_an_eneg if d.site == "anion" else sn_eneg for d in alld])
        eneg_contrast = float(np.dot(w, np.abs(np.array(enegs) - ref_e)) / wsum)

    covrad_mismatch = 0.0
    if alld and wsum > 0:
        covrad_mismatch = float(np.dot(w, np.abs(np.array(covs) - sn_covrad) / max(sn_covrad, 1e-6)) / wsum)

    is_sc = 1.0 if direction in (1, 2, 3) else 0.0
    is_b = 1.0 if direction == 2 else 0.0
    is_c = 1.0 if direction == 3 else 0.0
    is_a = 1.0 if direction == 1 else 0.0

    feats: List[float] = [
        p.sn_fraction,
        float(ELEMENTS[host_an].number),
        p.host_anion_fraction,
        float(len(cat_d)),
        float(len(an_d)),
        float(len(alld)),
        float(sum(d.fraction for d in cat_d)),
        float(sum(d.fraction for d in an_d)),
        wavg([float(d.z) for d in alld]),
        mean_mass,
        mass_var,
        wavg(enegs),
        eneg_contrast,
        wavg(covs),
        covrad_mismatch,
        wavg(periods),
        host_contrast,
        float(direction),
        is_sc,
        is_b,
        is_c,
        is_a,
    ]

    for i in range(TL_DOPANT_SLOTS):
        if i < len(alld):
            d = alld[i]
            ref_mass = host_an_mass if d.site == "anion" else sn_mass
            ref_eneg = host_an_eneg if d.site == "anion" else sn_eneg
            feats += [
                float(d.z),
                float(d.fraction),
                float(d.is_anion_site),
                float(_prop(d.symbol, "mass") / max(ref_mass, 1e-6)),
                float(_prop(d.symbol, "eleneg") - ref_eneg),
            ]
        else:
            feats += [0.0, 0.0, 0.0, 0.0, 0.0]

    return np.asarray(feats, dtype=float)


# ------------------------------------------------------------- legacy compatibility
def parse_composition_to_krict(formula_str: str) -> Optional[dict]:
    """Drop-in replacement for the original two-slot parser.

    Same keys and same clipping as before, so existing callers keep working, but the
    slots are now filled by the site-aware logic in to_base_row() rather than by a
    plain sort that let the host anion crowd out a real dopant.
    """
    p = parse_composition(formula_str)
    row = to_base_row(p, temperature=300.0)
    if row is None:
        return None
    return {
        "Sn_ratio": row[2],
        "Se_ratio": row[3],
        "dopant1_num": row[4],
        "dopant1_ratio": row[5],
        "dopant2_num": row[6],
        "dopant2_ratio": row[7],
    }


def describe(formula: str) -> str:
    """One-line human-readable account of how a formula is encoded."""
    p = parse_composition(formula)
    if not p.ok:
        return f"{formula}: UNPARSEABLE ({p.reason})"
    to_base_row(p, 300.0)
    dop = ", ".join(f"{d.symbol}{d.fraction:.3f}[{d.site[:3]}]" for d in p.dopants) or "none"
    lost = f"  | base layer drops: {', '.join(p.base_lost)}" if p.base_lost else ""
    return (f"{formula}: host Sn{p.sn_fraction:.3f}/{p.host_anion}{p.host_anion_fraction:.3f} "
            f"| {p.n_dopants} dopants: {dop}{lost}")


if __name__ == "__main__":
    for f in ["SnSe", "SnS", "Sn0.96Na0.01Ag0.03S", "SnS0.91Se0.09", "SnSe0.25Te0.75",
              "Cu2ZnSnSe4", "(SnTe)0.9(CuSbTe2)0.1", "Cu5SnSbSe7",
              "Na0.01(Sn0.95Te0.03)0.99S"]:
        print(describe(f))
