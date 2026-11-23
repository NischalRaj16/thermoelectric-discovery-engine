"""
screen_sns_advanced.py — Comprehensive High-Throughput Screening for SnS Host Materials.

Workflow:
  - Host Lattice: SnS (Orthorhombic Pnma, anion = S, cation = Sn)
  - Dopant Elements: Na, Ag, Bi, Pb, Cd, Zn, Sb, In, Se, Te, Ge, Cu
  - Doping Ratios: 1.0%, 2.0%, 3.0%, 4.0%, 5.0%
  - Sn Vacancy Ratios: 0.0%, 0.5%, 1.0%, 2.0%
  - Na Co-Doping: Single-doped and Na 1.0% co-doped systems
  - Temperature Grid: 100 K to 1000 K (step: 50 K = 19 discrete temperature points)
  - Materials Project API (MP_API): Queries convex hull phase stability (Delta E_hull)
  - Experimental Transfer Learning: Calibrates predictions using uploaded datasets (ESTM, UCSB, Science SnS crystals)
  - Exports:
      1. outputdata_search/sns_screening_leaderboard_100K_to_1000K.csv
      2. outputdata_search/sns_full_temperature_sweeps_100K_to_1000K.csv
      3. outputdata_search/plot_sns_champions_100K_to_1000K.png
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PREDICT_DIR = os.path.join(HERE, "1_predicting_new_compositions")
sys.path.insert(0, HERE)
sys.path.insert(0, PREDICT_DIR)

from elements import ELEMENTS
from evaluate_uploaded_datasets import ModelEvaluator
from transfer_learning import SnSeTransferLearner
from composition import parse_composition, to_base_row, to_tl_features, BASE_ROW_COLUMNS
from mp_client import get_mp_stability


def run_sns_advanced_screening():
    out_dir = os.path.join(HERE, "outputdata_search")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print("      COMPREHENSIVE HIGH-THROUGHPUT THERMOELECTRIC SCREENING: SnS HOST")
    print("=" * 80)
    print("  • Host Material       : SnS (Orthorhombic Pnma, S host lattice)")
    print("  • Temperature Sweep   : 100 K to 1000 K (19 discrete points)")
    print("  • Dopant Spectrum     : Na, Ag, Bi, Pb, Cd, Zn, Sb, In, Se, Te, Ge, Cu")
    print("  • Doping Concentr.    : 1.0%, 2.0%, 3.0%, 4.0%, 5.0%")
    print("  • Sn Vacancy Defects  : 0.0%, 0.5%, 1.0%, 2.0%")
    print("  • Thermodynamic Check : Materials Project API (ΔE_hull in eV/atom)")
    print("  • Physics Engine      : Multi-Valley DFT + Experimental Transfer Learning")
    print("-" * 80)

    # 1. Initialize Engine
    print("Initializing Transfer Learning & Materials Project Engine...")
    learner = SnSeTransferLearner()
    learner.load_transfer_models()
    print("✓ Model engines loaded.\n")

    # 2. Define Parameter Space
    dopants = ["Na", "Ag", "Bi", "Pb", "Cd", "Zn", "Sb", "In", "Se", "Te", "Ge", "Cu"]
    ratios = [1.0, 2.0, 3.0, 4.0, 5.0]
    vacancies = [0.0, 0.5, 1.0, 2.0]
    na_modes = [False, True]  # False = single doped, True = co-doped with 1% Na
    temps = list(range(100, 1001, 50))  # 100, 150, ..., 1000 K

    # Materials Project stability cache for unique chemical systems
    mp_cache = {}
    def get_cached_mp(dop_elem):
        if dop_elem not in mp_cache:
            try:
                mp_info = get_mp_stability(f"Sn{dop_elem}S")
                mp_cache[dop_elem] = mp_info
            except Exception:
                mp_cache[dop_elem] = {"material_id": "N/A", "energy_above_hull": None, "status": "Unknown", "band_gap": None}
        return mp_cache[dop_elem]

    # Pre-cache MP data
    print("Querying Materials Project API for chemical systems...")
    for d in dopants:
        get_cached_mp(d)
    print("✓ Materials Project API checks complete.")

    # 3. Build Batch Prediction Table
    comp_inputs = []
    comp_metadata = []
    parsed_rows = []
    numrow = 0
    system_id = 0

    print("Generating composition configurations and temperature grids...")
    for dop in dopants:
        dop_num = ELEMENTS[dop].number
        for r in ratios:
            for vac in vacancies:
                for with_na in na_modes:
                    dop_frac = r / 100.0
                    vac_frac = vac / 100.0
                    na_frac = 0.01 if with_na else 0.0

                    # Stoichiometric normalization
                    sn_frac_in_cation = max(0.0, 1.0 - dop_frac - vac_frac)
                    sn_effective = (1.0 - na_frac) * sn_frac_in_cation
                    dop_effective = (1.0 - na_frac) * dop_frac

                    if with_na:
                        formula_str = f"Na0.01(Sn{sn_frac_in_cation:.3f}{dop}{dop_frac:.3f})0.99S"
                        config_name = f"SnS + 1%Na + {r}%{dop} (vac {vac}%)"
                    else:
                        formula_str = f"(Sn{sn_frac_in_cation:.3f}{dop}{dop_frac:.3f})S"
                        config_name = f"SnS + {r}%{dop} (vac {vac}%)"

                    # Build the base row FROM the formula so the host anion is
                    # actually encoded. Previously Se_ratio was hard-coded to 1.0 and
                    # sulfur was never passed, so every "SnS" candidate reached the
                    # frozen models as SnSe.
                    parsed_sys = parse_composition(formula_str)
                    for T in temps:
                        row = to_base_row(parsed_sys, float(T), direction=0,
                                          numrow=numrow, row_id=f"SYS{system_id:04d}")
                        if row is None:
                            continue
                        comp_inputs.append(row)
                        parsed_rows.append(parse_composition(formula_str))
                        comp_metadata.append({
                            "sys_id": f"SYS{system_id:04d}",
                            "formula": formula_str,
                            "config": config_name,
                            "dopant": dop,
                            "ratio": r,
                            "vacancy": vac,
                            "has_na": with_na,
                            "T": T,
                        })
                        numrow += 1

                    system_id += 1

    cols_header = list(BASE_ROW_COLUMNS)
    df_comp = pd.DataFrame(comp_inputs, columns=cols_header).set_index("numrow")
    print(f"Generated {len(df_comp)} calculation points across {system_id} unique chemical systems.")

    # 4. Run High-Throughput Predictions
    print("Executing 5-fold ensemble predictions with Transfer Learning domain adaptation...", flush=True)
    start_time = time.time()
    # Pass the formulas: previously Se_ratio was hard-coded to 1.0 and sulfur was
    # never encoded, so every "SnS" candidate was evaluated by the model as SnSe.
    tl_extra = np.vstack([to_tl_features(p) for p in parsed_rows])
    preds = learner._predict_core(df_comp, tl_extra, parsed_rows)
    elapsed = time.time() - start_time
    print(f"✓ Inference complete in {elapsed:.1f}s.")

    # Merge metadata with predictions
    df_all = pd.DataFrame(comp_metadata)
    df_all["sigma (S/cm)"] = preds["sigma"]
    df_all["Seebeck (µV/K)"] = preds["seebeck"]
    df_all["kappa (W/m·K)"] = preds["kappa"]
    df_all["PF (µW/cm·K²)"] = preds["PF"]
    df_all["ZT"] = preds["ZT"]
    df_all["ZT_uncertainty"] = preds["ZT_sd"]
    df_all["consistency_ratio"] = preds["consistency_ratio"]
    df_all["in_domain"] = preds["in_domain"]
    df_all["flags"] = ["; ".join(f) for f in preds["domain_flags"]]
    if "Base_ZT" in preds:
        df_all["Base_DFT_ZT"] = preds["Base_ZT"]

    # Export Full Temperature Sweeps
    sweep_path = os.path.join(out_dir, "sns_full_temperature_sweeps_100K_to_1000K.csv")
    df_all.to_csv(sweep_path, index=False)
    print(f"✓ Exported complete 100K–1000K temperature sweeps to:\n  {sweep_path}")

    # 5. Build Comprehensive Ranked Leaderboard
    leaderboard = []
    for sys_id, grp in df_all.groupby("sys_id"):
        # Rank on the best IN-DOMAIN point. Half of the previous leaderboard peaked
        # at exactly 1000 K, the edge of the sweep, where there is no experimental
        # support at all; those peaks are no longer allowed to win a ranking.
        pool = grp[grp["in_domain"]] if grp["in_domain"].any() else grp
        best_row = pool.loc[pool["ZT"].idxmax()]
        dop = best_row["dopant"]
        mp = mp_cache.get(dop, {})

        leaderboard.append({
            "Composition": best_row["formula"],
            "Configuration": best_row["config"],
            "Dopant": dop,
            "Doping_Ratio (%)": best_row["ratio"],
            "Sn_Vacancy (%)": best_row["vacancy"],
            "Na_Co-doped": "Yes (1% Na)" if best_row["has_na"] else "No",
            "Peak_ZT": float(best_row["ZT"]),
            "ZT_uncertainty": float(best_row["ZT_uncertainty"]),
            "In_Domain": bool(best_row["in_domain"]),
            "Flags": best_row["flags"],
            "T_opt (K)": int(best_row["T"]),
            "PF @ T_opt (µW/cm·K²)": float(best_row["PF (µW/cm·K²)"]),
            "κ @ T_opt (W/m·K)": float(best_row["kappa (W/m·K)"]),
            "σ @ T_opt (S/cm)": float(best_row["sigma (S/cm)"]),
            "S @ T_opt (µV/K)": float(best_row["Seebeck (µV/K)"]),
            "MP_Status": mp.get("status", "N/A"),
            "MP_Hull (eV/atom)": mp.get("energy_above_hull", None),
            "MP_Bandgap (eV)": mp.get("band_gap", None)
        })

    df_lead = pd.DataFrame(leaderboard).sort_values("Peak_ZT", ascending=False).reset_index(drop=True)
    df_lead.index = df_lead.index + 1
    df_lead.index.name = "Rank"

    lead_path = os.path.join(out_dir, "sns_screening_leaderboard_100K_to_1000K.csv")
    df_lead.to_csv(lead_path)
    print(f"✓ Exported ranked champions leaderboard to:\n  {lead_path}")

    # 6. Temperature Regime Breakdown
    print("\n" + "=" * 80)
    print("                     TOP CHAMPIONS BY TEMPERATURE REGIME")
    print("=" * 80)

    # Low-T: 100 K - 400 K
    low_t = df_all[df_all["T"] <= 400]
    best_low = low_t.loc[low_t.groupby("sys_id")["ZT"].idxmax()].sort_values("ZT", ascending=False).iloc[0]
    print(f"\n[1] LOW-TEMPERATURE REGIME (100 K – 400 K):")
    print(f"    • Champion Material  : {best_low['formula']}")
    print(f"    • Configuration      : {best_low['config']}")
    print(f"    • Maximum ZT         : {best_low['ZT']:.4f} at {best_low['T']} K")
    print(f"    • Power Factor       : {best_low['PF (µW/cm·K²)']:.2f} µW/(cm·K²)")
    print(f"    • Thermal Cond (κ)   : {best_low['kappa (W/m·K)']:.3f} W/(m·K)")

    # Mid-T: 450 K - 700 K
    mid_t = df_all[(df_all["T"] >= 450) & (df_all["T"] <= 700)]
    best_mid = mid_t.loc[mid_t.groupby("sys_id")["ZT"].idxmax()].sort_values("ZT", ascending=False).iloc[0]
    print(f"\n[2] MID-TEMPERATURE REGIME (450 K – 700 K):")
    print(f"    • Champion Material  : {best_mid['formula']}")
    print(f"    • Configuration      : {best_mid['config']}")
    print(f"    • Maximum ZT         : {best_mid['ZT']:.4f} at {best_mid['T']} K")
    print(f"    • Power Factor       : {best_mid['PF (µW/cm·K²)']:.2f} µW/(cm·K²)")
    print(f"    • Thermal Cond (κ)   : {best_mid['kappa (W/m·K)']:.3f} W/(m·K)")

    # High-T: 750 K - 1000 K
    high_t = df_all[df_all["T"] >= 750]
    best_high = high_t.loc[high_t.groupby("sys_id")["ZT"].idxmax()].sort_values("ZT", ascending=False).iloc[0]
    print(f"\n[3] HIGH-TEMPERATURE REGIME (750 K – 1000 K):")
    print(f"    • Champion Material  : {best_high['formula']}")
    print(f"    • Configuration      : {best_high['config']}")
    print(f"    • Maximum ZT         : {best_high['ZT']:.4f} at {best_high['T']} K")
    print(f"    • Power Factor       : {best_high['PF (µW/cm·K²)']:.2f} µW/(cm·K²)")
    print(f"    • Thermal Cond (κ)   : {best_high['kappa (W/m·K)']:.3f} W/(m·K)")

    # 7. Print Top 15 Overall Leaderboard
    print("\n" + "=" * 80)
    print("                    TOP 15 OVERALL CHAMPIONS ON SnS")
    print("=" * 80)
    display_cols = ["Composition", "Dopant", "Na_Co-doped", "Peak_ZT", "T_opt (K)", "PF @ T_opt (µW/cm·K²)", "κ @ T_opt (W/m·K)", "MP_Status"]
    print(df_lead[display_cols].head(15).to_string())
    print("=" * 80)

    # 8. Generate 4-Panel Publication Plot for Top Champions
    plot_path = os.path.join(out_dir, "plot_sns_champions_100K_to_1000K.png")
    plot_champions(df_all, df_lead.head(5), plot_path)
    print(f"\n✓ Generated high-resolution 4-panel publication plot at:\n  {plot_path}\n")

    return df_lead, df_all


def plot_champions(df_all, top_df, out_path):
    """Save publication-quality 4-panel curve comparison for top champions."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300)
    colors = ["#0d9488", "#2563eb", "#d97706", "#e11d48", "#7c3aed"]

    for idx, (_, row) in enumerate(top_df.iterrows()):
        formula = row["Composition"]
        sub = df_all[df_all["formula"] == formula].sort_values("T")
        t = sub["T"]
        c = colors[idx % len(colors)]
        lbl = f"{row['Dopant']} {row['Doping_Ratio (%)']}% (ZT={row['Peak_ZT']:.2f})"

        # 1. ZT Curve
        axes[0, 0].plot(t, sub["ZT"], color=c, lw=2.2, marker="o", ms=3.5, label=lbl)

        # 2. Power Factor
        axes[0, 1].plot(t, sub["PF (µW/cm·K²)"], color=c, lw=2.0, marker="s", ms=3.5, label=lbl)

        # 3. Electrical Conductivity
        axes[1, 0].plot(t, sub["sigma (S/cm)"], color=c, lw=2.0, marker="^", ms=3.5, label=lbl)

        # 4. Thermal Conductivity
        axes[1, 1].plot(t, sub["kappa (W/m·K)"], color=c, lw=2.0, marker="d", ms=3.5, label=lbl)

    # Format Axes
    axes[0, 0].set_title("Figure of Merit (ZT) vs Temperature (100 K – 1000 K)", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Temperature (K)", fontsize=10)
    axes[0, 0].set_ylabel("ZT", fontsize=10)
    axes[0, 0].grid(True, ls=":", alpha=0.6)
    axes[0, 0].legend(fontsize=8, loc="upper left")

    axes[0, 1].set_title("Power Factor (PF = S²·σ) vs Temperature", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Temperature (K)", fontsize=10)
    axes[0, 1].set_ylabel("PF [µW/(cm·K²)]", fontsize=10)
    axes[0, 1].grid(True, ls=":", alpha=0.6)
    axes[0, 1].legend(fontsize=8, loc="upper left")

    axes[1, 0].set_title("Electrical Conductivity (σ) vs Temperature", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Temperature (K)", fontsize=10)
    axes[1, 0].set_ylabel("Conductivity σ (S/cm)", fontsize=10)
    axes[1, 0].grid(True, ls=":", alpha=0.6)
    axes[1, 0].legend(fontsize=8, loc="upper left")

    axes[1, 1].set_title("Thermal Conductivity (κ) vs Temperature", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Temperature (K)", fontsize=10)
    axes[1, 1].set_ylabel("κ [W/(m·K)]", fontsize=10)
    axes[1, 1].grid(True, ls=":", alpha=0.6)
    axes[1, 1].legend(fontsize=8, loc="upper right")

    fig.suptitle("SnS High-Throughput Thermoelectric Screening: Top Champions (100 K – 1000 K)", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path)
    plt.close(fig)


if __name__ == "__main__":
    run_sns_advanced_screening()
