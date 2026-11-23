"""
run_snse_ml_interactive.py — Unified Interactive Discovery & Prediction Pipeline for SnSe_MLcopy.

Features:
  1. Base KRICT DFT Physics Models + Transfer Learning Domain Adaptation
  2. Materials Project API (MP_API) Integration for Thermodynamic Stability (Delta E_hull, Spacegroup, Bandgap)
  3. Temperature sweeps from 100 K to 1000 K with Champion and Optimal Temperature (T_opt) Identification
  4. Interactive terminal menu or command-line screening mode
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from evaluate_uploaded_datasets import parse_composition_to_krict
from transfer_learning import SnSeTransferLearner
from mp_client import get_mp_stability, search_materials, export_structure


def predict_single_composition(formula: str, learner: SnSeTransferLearner, use_tl: bool = True, query_mp: bool = True):
    """Predict transport properties and ZT for a single composition from 100 K to 1000 K."""
    kdict = parse_composition_to_krict(formula)
    if kdict is None:
        print(f"Error: Unable to parse formula '{formula}' into Sn-chalcogenide representation.")
        return None

    # MP API query
    mp_info = None
    if query_mp:
        print(f"\n[MP_API] Querying Materials Project for '{formula}'...")
        mp_info = get_mp_stability(formula)
        print(f"  • Materials Project ID   : {mp_info.get('material_id', 'N/A')}")
        print(f"  • Spacegroup / Symmetry  : {mp_info.get('symmetry', 'N/A')}")
        print(f"  • Thermodynamic Status   : {mp_info.get('status', 'N/A')}")
        print(f"  • Energy Above Hull      : {mp_info.get('energy_above_hull', 'N/A')} eV/atom")
        print(f"  • MP Bandgap             : {mp_info.get('band_gap', 'N/A')} eV")

    # Generate temperature sweep from 100 K to 1000 K
    temps = list(range(100, 1001, 50))
    rows = []
    cols_header = [
        "numrow", "ID", "Sn_ratio", "Se_ratio", "dopant1_num", "dopant1_ratio",
        "dopant2_num", "dopant2_ratio", "dopant3_num", "dopant3_ratio",
        "direction", "Temp", "category"
    ]

    for idx, T in enumerate(temps):
        rows.append([
            idx, f"P{idx:03d}",
            kdict["Sn_ratio"], kdict["Se_ratio"],
            kdict["dopant1_num"], kdict["dopant1_ratio"],
            kdict["dopant2_num"], kdict["dopant2_ratio"],
            0, 0, 0, T, 1  # polycrystal, p-type
        ])

    df_comp = pd.DataFrame(rows, columns=cols_header).set_index("numrow")

    if use_tl:
        preds = learner.predict_transfer(df_comp)
        model_name = "Transfer-Learned (KRICT DFT + Experimental)"
    else:
        preds = learner.base_evaluator.predict_batch(df_comp)
        model_name = "Base KRICT DFT Ensemble"

    df_res = pd.DataFrame({
        "Temp (K)": temps,
        "sigma (S/cm)": preds["sigma"],
        "Seebeck (µV/K)": preds["seebeck"],
        "kappa (W/m·K)": preds["kappa"],
        "PF (µW/cm·K²)": preds["PF"],
        "ZT": preds["ZT"]
    })

    if "Base_ZT" in preds and use_tl:
        df_res["Base_KRICT_ZT"] = preds["Base_ZT"]

    best_idx = df_res["ZT"].idxmax()
    champ_row = df_res.loc[best_idx]

    print("\n" + "=" * 78)
    print(f"  PREDICTION RESULTS FOR: {formula}")
    print(f"  Model Engine: {model_name}")
    print("=" * 78)
    print(f"  • Maximum ZT (Peak ZT)   : {champ_row['ZT']:.4f}")
    print(f"  • Optimal Temp (T_opt)   : {int(champ_row['Temp (K)'])} K")
    print(f"  • Power Factor @ T_opt   : {champ_row['PF (µW/cm·K²)']:.2f} µW/(cm·K²)")
    print(f"  • Thermal Cond @ T_opt   : {champ_row['kappa (W/m·K)']:.3f} W/(m·K)")
    print(f"  • Conductivity @ T_opt   : {champ_row['sigma (S/cm)']:.2f} S/cm")
    print(f"  • Seebeck Coeff @ T_opt  : {champ_row['Seebeck (µV/K)']:.2f} µV/K")
    print("-" * 78)
    print("  Temperature Sweep Profile (100 K to 1000 K):")
    print(df_res.to_string(index=False))
    print("=" * 78)

    return df_res, champ_row


def interactive_menu():
    learner = SnSeTransferLearner()
    learner.load_transfer_models()

    while True:
        print("\n" + "=" * 70)
        print("   SnSe_MLcopy: DFT Multi-Valley + Transfer Learning + MP_API")
        print("=" * 70)
        print("  1. Predict Composition (100 K – 1000 K, find Peak ZT & T_opt)")
        print("  2. Screen Uploaded Datasets (ESTM, UCSB, Science SnS Crystals)")
        print("  3. Query Materials Project API (Phase Stability, Bandgap, Spacegroup)")
        print("  4. Re-train Transfer Learning Models on Experimental Datasets")
        print("  q. Quit")
        print("-" * 70)

        choice = input("Enter selection [1-4 or q]: ").strip().lower()
        if choice in ["q", "quit", "exit"]:
            print("Exiting SnSe_MLcopy. Goodbye!")
            break

        if choice == "1":
            formula = input("Enter chemical composition (e.g. Sn0.96Na0.01Ag0.03S, Sn0.98In0.02Se): ").strip()
            if not formula:
                continue
            use_tl_in = input("Enable Transfer Learning enhancement? [Y/n]: ").strip().lower()
            use_tl = use_tl_in != "n"
            predict_single_composition(formula, learner, use_tl=use_tl, query_mp=True)

        elif choice == "2":
            print("\nRunning complete evaluation and 100 K – 1000 K sweep across uploaded datasets...")
            from evaluate_uploaded_datasets import main as run_eval
            run_eval()

        elif choice == "3":
            formula = input("Enter formula to query in Materials Project (e.g. SnSe, SnS, Bi2Te3): ").strip()
            if formula:
                res = get_mp_stability(formula)
                print("\n[Materials Project Response]:")
                for k, v in res.items():
                    print(f"  • {k:<25}: {v}")

        elif choice == "4":
            print("\nRe-training transfer learning models...")
            df_comp, df_data, x_tl, parsed_all = learner.prepare_experimental_dataset()
            learner.train_transfer_learning_models(df_comp, df_data, x_tl, parsed_all)


def main():
    parser = argparse.ArgumentParser(description="SnSe_MLcopy Unified Discovery Pipeline")
    parser.add_argument("-i", "--interactive", action="store_true", help="Launch interactive CLI menu")
    parser.add_argument("--formula", help="Single chemical composition to predict (e.g. Sn0.96Na0.01Ag0.03S)")
    parser.add_argument("--no-tl", action="store_true", help="Disable transfer learning (use base KRICT models only)")
    parser.add_argument("--no-mp", action="store_true", help="Disable Materials Project API query")
    parser.add_argument("--mp-search", help="Search Materials Project by formula or chemsys")
    args = parser.parse_args()

    learner = SnSeTransferLearner()
    learner.load_transfer_models()

    if args.formula:
        predict_single_composition(args.formula, learner, use_tl=not args.no_tl, query_mp=not args.no_mp)
    elif args.mp_search:
        res = get_mp_stability(args.mp_search)
        print("\n[Materials Project API Results]:")
        for k, v in res.items():
            print(f"  • {k:<25}: {v}")
    elif args.interactive or len(sys.argv) == 1:
        interactive_menu()


if __name__ == "__main__":
    main()
