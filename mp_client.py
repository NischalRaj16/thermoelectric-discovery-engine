"""
mp_client.py — Materials Project API Client for SnSe_MLcopy.

Connects to the Materials Project API (v2024+) using mp-api client.
Retrieves:
  - Thermodynamic phase stability (Energy above hull: energy_above_hull / Delta E_hull)
  - Formation energy per atom (formation_energy_per_atom)
  - Decomposition / Stable phase flag (is_stable)
  - Electronic bandgap (band_gap)
  - Spacegroup symmetry (symmetry.symbol, crystal_system)
  - Density and unit cell volume
  - Export crystal structures (CIF / POSCAR)
"""

import os
import json
import re
from typing import Dict, Any, Optional, List

# mp-api is optional: without it the pipeline still runs and simply reports that
# stability could not be checked, rather than failing to import.
try:
    from mp_api.client import MPRester
    MP_AVAILABLE = True
except ImportError:  # pragma: no cover
    MPRester = None
    MP_AVAILABLE = False

# The API key must NOT live in a tracked source file. Set the MP_API_KEY environment
# variable, or drop the key into a local .mp_api_key file (add it to .gitignore).
def _read_api_key() -> str:
    key = os.environ.get("MP_API_KEY", "").strip()
    if key:
        return key
    keyfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mp_api_key")
    if os.path.exists(keyfile):
        try:
            with open(keyfile, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            return ""
    return ""


DEFAULT_API_KEY = _read_api_key()
CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mp_cache.json")


def _load_cache() -> Dict[str, Any]:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(cache: Dict[str, Any]):
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass


def get_client(api_key: Optional[str] = None) -> MPRester:
    """Return an authenticated MPRester instance."""
    key = api_key or DEFAULT_API_KEY
    return MPRester(api_key=key)


def get_mp_stability(formula: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Query Materials Project for thermodynamic stability and electronic properties of a formula.
    Checks local cache first; if not present, queries MP and caches result.
    """
    if not MP_AVAILABLE or not DEFAULT_API_KEY:
        return {"material_id": "N/A", "energy_above_hull": None, "band_gap": None,
                "symmetry": "N/A", "status": "MP lookup unavailable (no mp-api or no MP_API_KEY)"}

    clean_formula = re.sub(r"[0-9\.\(\)]", "", formula).strip()
    cache = _load_cache()
    
    # Check cache for exact or base formula
    for query_key in [formula, clean_formula]:
        if query_key in cache:
            return cache[query_key]

    result = {
        "formula": formula,
        "material_id": "N/A",
        "energy_above_hull": None,
        "is_stable": None,
        "formation_energy_per_atom": None,
        "band_gap": None,
        "symmetry": "Unknown",
        "density": None,
        "status": "Not found in MP"
    }

    try:
        with get_client(api_key) as mpr:
            # First search by exact formula, then by reduced formula
            docs = mpr.materials.summary.search(
                formula=clean_formula,
                fields=[
                    "material_id",
                    "formula_pretty",
                    "symmetry",
                    "band_gap",
                    "formation_energy_per_atom",
                    "energy_above_hull",
                    "density",
                    "is_stable"
                ]
            )

            if docs:
                # Sort by energy_above_hull (ground state first)
                docs = sorted(docs, key=lambda x: getattr(x, "energy_above_hull", 999.0))
                best = docs[0]
                sym_str = f"{best.symmetry.symbol} ({best.symmetry.crystal_system})" if best.symmetry else "Unknown"

                result = {
                    "formula": formula,
                    "matched_mp_formula": best.formula_pretty,
                    "material_id": str(best.material_id),
                    "energy_above_hull": round(float(best.energy_above_hull), 4) if best.energy_above_hull is not None else None,
                    "is_stable": bool(best.is_stable),
                    "formation_energy_per_atom": round(float(best.formation_energy_per_atom), 4) if best.formation_energy_per_atom is not None else None,
                    "band_gap": round(float(best.band_gap), 3) if best.band_gap is not None else None,
                    "symmetry": sym_str,
                    "density": round(float(best.density), 3) if best.density is not None else None,
                    "status": "Stable Ground State" if best.is_stable else f"Metastable (Hull +{best.energy_above_hull:.3f} eV/atom)"
                }

                cache[formula] = result
                cache[clean_formula] = result
                _save_cache(cache)
    except Exception as e:
        result["status"] = f"MP Query Error: {str(e)[:60]}"

    return result


def search_materials(
    formula: Optional[str] = None,
    chemsys: Optional[str] = None,
    stable_only: bool = False,
    limit: int = 10,
    api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Search Materials Project materials by formula or chemical system.
    """
    results = []
    try:
        with get_client(api_key) as mpr:
            kwargs = {
                "fields": [
                    "material_id",
                    "formula_pretty",
                    "symmetry",
                    "band_gap",
                    "formation_energy_per_atom",
                    "energy_above_hull",
                    "density",
                    "is_stable"
                ]
            }
            if formula:
                kwargs["formula"] = formula
            if chemsys:
                kwargs["chemsys"] = chemsys
            if stable_only:
                kwargs["is_stable"] = True

            docs = mpr.materials.summary.search(**kwargs)
            docs = sorted(docs, key=lambda x: getattr(x, "energy_above_hull", 999.0))

            for d in docs[:limit]:
                sym_str = f"{d.symmetry.symbol} ({d.symmetry.crystal_system})" if d.symmetry else "N/A"
                results.append({
                    "material_id": str(d.material_id),
                    "formula": d.formula_pretty,
                    "symmetry": sym_str,
                    "energy_above_hull": round(float(d.energy_above_hull), 4) if d.energy_above_hull is not None else None,
                    "formation_energy_per_atom": round(float(d.formation_energy_per_atom), 4) if d.formation_energy_per_atom is not None else None,
                    "band_gap": round(float(d.band_gap), 3) if d.band_gap is not None else None,
                    "density": round(float(d.density), 3) if d.density is not None else None,
                    "is_stable": bool(d.is_stable)
                })
    except Exception as e:
        print(f"Error querying Materials Project: {e}")

    return results


def export_structure(material_id: str, out_cif: Optional[str] = None, out_poscar: Optional[str] = None, api_key: Optional[str] = None):
    """Retrieve crystal structure from MP and write to CIF or POSCAR."""
    with get_client(api_key) as mpr:
        docs = mpr.materials.summary.search(material_ids=[material_id], fields=["material_id", "formula_pretty", "structure"])
        if not docs:
            print(f"Material ID {material_id} not found in Materials Project.")
            return None
        doc = docs[0]
        structure = doc.structure

        cif_path = out_cif or f"{doc.formula_pretty}_{material_id}.cif"
        structure.to(filename=cif_path, fmt="cif")
        print(f"✓ Exported CIF: {cif_path}")

        if out_poscar:
            structure.to(filename=out_poscar, fmt="poscar")
            print(f"✓ Exported POSCAR: {out_poscar}")

        return structure


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Materials Project API for SnSe_MLcopy")
    parser.add_argument("--formula", default="SnSe", help="Chemical formula to check")
    parser.add_argument("--chemsys", help="Chemical system to query (e.g. Sn-Se, Sn-S)")
    parser.add_argument("--stable-only", action="store_true", help="Filter for thermodynamically stable materials only")
    args = parser.parse_args()

    print(f"\nQuerying Materials Project for: {args.formula or args.chemsys}...")
    res = get_mp_stability(args.formula)
    print(json.dumps(res, indent=2))
