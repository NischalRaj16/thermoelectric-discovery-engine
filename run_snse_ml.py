"""
run_snse_ml.py — Full Execution Pipeline for KRICT SnSe Machine Learning Framework.

Implements the complete predictive workflow from the study:
"Data-Driven Enhancement of ZT in SnSe-Based Thermoelectric Systems" (JACS 2022, 144, 13748–13763)

Features:
  1. Systematic screening and temperature sweeps (300 K to 820 K)
  2. DFT multi-valley electronic feature construction (VBM/CBM effective masses, bandgaps) + Mendeleev elemental descriptors
  3. 5-Fold Ensemble XGBoost predictions for:
     - Electrical Conductivity (σ in S/cm)
     - Seebeck Coefficient (S in µV/K)
     - Thermal Conductivity (κ in W/m·K)
  4. Calculation of Power Factor (PF) and Dimensionless Figure of Merit (ZT = S²·σ·T / κ · 1e-10)
  5. Leaderboard ranking and automatic CSV export
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb

HERE = os.path.dirname(os.path.abspath(__file__))
PREDICT_DIR = os.path.join(HERE, "1_predicting_new_compositions")
if PREDICT_DIR not in sys.path:
    sys.path.insert(0, PREDICT_DIR)

from elements import ELEMENTS
from feature2 import construct_feature
from myfunction import my_range
from util import cut_feature, calculate_ZT, define_category

MEASURESET = ["electrical_conductivity", "seebeck_coeff", "thermal_conductivity"]
DEFAULT_DOPANTS = ["Zn", "Ge", "Pb", "Cd", "Y", "As", "Ag", "Sb", "Bi", "S", "Te"]

# Full feature names
COLUMN = [
    'id', 'direction', 'temp', 'temp**1.5', '1/temp',
    'vbm_e', 'vbm_e2', 'vbm_e3',
    'cbm_e', 'cbm_e2', 'cbm_e3', 'magnetism', 'deep_level',
    'vbm_org_1', 'vbm_org_2', 'vbm_org_3', 'cbm_org_1', 'cbm_org_2', 'cbm_org_3',
    'vbm_m', 'vbm_m2', 'vbm_m3', 'cbm_m', 'cbm_m2', 'cbm_m3',
    'vbm_e_sn', 'vbm_e2_sn', 'vbm_e3_sn', 'cbm_e_sn', 'cbm_e2_sn', 'cbm_e3_sn',
    'vbm_e_se', 'vbm_e2_se', 'vbm_e3_se', 'cbm_e_se', 'cbm_e2_se', 'cbm_e3_se',
    'vbm_m_sn', 'vbm_m2_sn', 'vbm_m3_sn', 'cbm_m_sn', 'cbm_m2_sn', 'cbm_m3_sn',
    'vbm_m_se', 'vbm_m2_se', 'vbm_m3_se', 'cbm_m_se', 'cbm_m2_se', 'cbm_m3_se',
    'bandgap', 'bandgap_sn', 'bandgap_se', 'bandgap_rate',
    'ion', 'covalent', 'vdw', 'atomic_radius', 'pauling_en', 'en_allen', 'polarizability', 'melting_point',
    'boiling_point', 'density', 'specific_heat', 'thermal_conductivity',
    'Sn_ratio', 'Se_ratio', 'Na_ratio', 'Cl_ratio', 'dopant_num', 'dopant_ratio'
]

# Features to extract for models
FEATURE_CUT_SUBSET = [
    'direction', 'temp',
    'magnetism', 'deep_level',
    'vbm_org_1', 'vbm_org_2', 'vbm_org_3', 'cbm_org_1', 'cbm_org_2', 'cbm_org_3',
    'vbm_m', 'vbm_m2', 'vbm_m3', 'cbm_m', 'cbm_m2', 'cbm_m3',
    'vbm_e_sn', 'vbm_e2_sn', 'vbm_e3_sn', 'cbm_e_sn', 'cbm_e2_sn', 'cbm_e3_sn',
    'vbm_e_se', 'vbm_e2_se', 'vbm_e3_se', 'cbm_e_se', 'cbm_e2_se', 'cbm_e3_se',
    'vbm_m_sn', 'vbm_m2_sn', 'vbm_m3_sn', 'cbm_m_sn', 'cbm_m2_sn', 'cbm_m3_sn',
    'vbm_m_se', 'vbm_m2_se', 'vbm_m3_se', 'cbm_m_se', 'cbm_m2_se', 'cbm_m3_se',
    'bandgap', 'bandgap_sn', 'bandgap_se', 'bandgap_rate',
    'ion', 'covalent', 'vdw', 'atomic_radius', 'pauling_en', 'en_allen', 'polarizability', 'melting_point',
    'boiling_point', 'density', 'specific_heat', 'thermal_conductivity',
    'Sn_ratio', 'Se_ratio', 'Na_ratio', 'Cl_ratio', 'dopant_num', 'dopant_ratio'
]

FEATURES_FOR_CONSTRUCT = [
    'vbm_e', 'vbm_e2', 'vbm_e3',
    'cbm_e', 'cbm_e2', 'cbm_e3', 'magnetism', 'deep_level',
    'vbm_org_1', 'vbm_org_2', 'vbm_org_3', 'cbm_org_1', 'cbm_org_2', 'cbm_org_3',
    'vbm_m', 'vbm_m2', 'vbm_m3', 'cbm_m', 'cbm_m2', 'cbm_m3',
    'vbm_e_sn', 'vbm_e2_sn', 'vbm_e3_sn', 'cbm_e_sn', 'cbm_e2_sn', 'cbm_e3_sn',
    'vbm_e_se', 'vbm_e2_se', 'vbm_e3_se', 'cbm_e_se', 'cbm_e2_se', 'cbm_e3_se',
    'vbm_m_sn', 'vbm_m2_sn', 'vbm_m3_sn', 'cbm_m_sn', 'cbm_m2_sn', 'cbm_m3_sn',
    'vbm_m_se', 'vbm_m2_se', 'vbm_m3_se', 'cbm_m_se', 'cbm_m2_se', 'cbm_m3_se',
    'vbm_ordering', 'cbm_ordering', 'vbm_org_ordering', 'cbm_org_ordering',
    'vbm_sn_ordering', 'cbm_sn_ordering', 'vbm_se_ordering', 'cbm_se_ordering',
    'bandgap', 'bandgap_sn', 'bandgap_se', 'bandgap_rate',
    'ion', 'covalent', 'vdw', 'atomic_radius', 'pauling_en', 'en_allen', 'polarizability', 'melting_point',
    'boiling_point', 'density', 'specific_heat', 'thermal_conductivity'
]


class SnSeMLPipeline:
    def __init__(self, models_dir=PREDICT_DIR):
        self.models_dir = models_dir
        self.cal_df = pd.read_excel(os.path.join(models_dir, "Results_Summary_Final.xlsx"), index_col=0)
        self.calnum = self.cal_df.values
        self.boosters = {}
        self._load_models()

    def _load_models(self):
        """Load the 5-fold ensemble boosters for all 3 target properties."""
        print("  Loading 5-Fold Ensemble XGBoost models...")
        for measure in MEASURESET:
            self.boosters[measure] = []
            for fold in range(5):
                json_path = os.path.join(self.models_dir, f"{measure}_{fold}_seed_0.json")
                if not os.path.exists(json_path):
                    raise FileNotFoundError(f"Missing model JSON: {json_path}")
                booster = xgb.Booster()
                booster.load_model(json_path)
                self.boosters[measure].append(booster)
        print("  ✓ All 15 XGBoost models (3 targets × 5 folds) loaded successfully!\n")

    def run_screening(
        self,
        elements=None,
        alloy_ratios=None,
        start_temp=300,
        end_temp=820,
        step_temp=20,
        eval_with_na=True,
        eval_without_na=True,
        output_csv=None
    ):
        """
        Execute full screening pipeline across temperatures and dopants.
        """
        elements = elements or DEFAULT_DOPANTS
        temps = list(range(start_temp, end_temp + 1, step_temp))
        if alloy_ratios is None:
            # Default 1% to 3% doping
            ratios = [1.0] * len(elements)
        else:
            ratios = alloy_ratios

        print("=" * 78)
        print("     KRICT SnSe MACHINE LEARNING THERMOELECTRIC PIPELINE (JACS 2022)")
        print("=" * 78)
        print(f"  • Dopant Elements     : {', '.join(elements)}")
        print(f"  • Temperature Range   : {start_temp} K to {end_temp} K (step: {step_temp} K, {len(temps)} points)")
        print(f"  • Test Configurations : {'[Na co-doped (1%)] ' if eval_with_na else ''}{'[Single-doped] ' if eval_without_na else ''}")
        print("-" * 78)

        # 1. Generate compositions table
        rows = []
        comp_records = []
        numrow = 0
        ids = 0

        for el, r in zip(elements, ratios):
            el_num = ELEMENTS[el].number

            # Configuration A: With 1% Na (p-type optimization)
            if eval_with_na:
                formula_label = f"Na0.01(Sn{1.0 - (r/100.0):.2f}{el}{r/100.0:.2f})0.99Se"
                for T in temps:
                    comp = [
                        numrow,
                        f"TY{ids:04d}",
                        (1.0 - (r / 100.0)) * 0.99,
                        1.0,
                        el_num,
                        r / 100.0,
                        11,  # Na atomic number
                        0.01,
                        0,
                        0,
                        0,  # crystallographic direction 0
                        T
                    ]
                    define_category(comp)
                    rows.append(comp)
                    comp_records.append({
                        "id": f"TY{ids:04d}",
                        "formula": formula_label,
                        "dopant": el,
                        "ratio": r,
                        "has_Na": True,
                        "T": T
                    })
                    numrow += 1
                ids += 1

            # Configuration B: Without Na (single-doped baseline)
            if eval_without_na:
                formula_label = f"Sn{1.0 - (r/100.0):.2f}{el}{r/100.0:.2f}Se"
                for T in temps:
                    comp = [
                        numrow,
                        f"TY{ids:04d}",
                        1.0 - (r / 100.0),
                        1.0,
                        el_num,
                        r / 100.0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        T
                    ]
                    define_category(comp)
                    rows.append(comp)
                    comp_records.append({
                        "id": f"TY{ids:04d}",
                        "formula": formula_label,
                        "dopant": el,
                        "ratio": r,
                        "has_Na": False,
                        "T": T
                    })
                    numrow += 1
                ids += 1

        cols_header = [
            "numrow", "ID", "Sn_ratio", "Se_ratio", "dopant1_num", "dopant1_ratio",
            "dopant2_num", "dopant2_ratio", "dopant3_num", "dopant3_ratio",
            "direction", "Temp", "category"
        ]
        df_comp = pd.DataFrame(rows, columns=cols_header).set_index("numrow")

        print(f"  • Total Compositions Generated : {len(df_comp)} points across {ids} chemical systems")
        print("  • Constructing DFT electronic & Mendeleev features...", end=" ", flush=True)

        # 2. Featurize
        # Temporarily switch cwd to PREDICT_DIR because feature2 loads mendeleev.csv locally
        orig_cwd = os.getcwd()
        os.chdir(self.models_dir)
        try:
            feats_raw = construct_feature(df_comp, self.calnum, FEATURES_FOR_CONSTRUCT, False)
        finally:
            os.chdir(orig_cwd)

        print("done.")
        print("  • Running 5-fold ensemble predictions for σ(T), S(T), and κ(T)...", end=" ", flush=True)

        # 3. Model Predictions
        n_test = df_comp.shape[0]
        preds = np.zeros((3, n_test))

        for idx, measure in enumerate(MEASURESET):
            # Cut features to the exact 57 input dimensions
            test_x = cut_feature(COLUMN, FEATURE_CUT_SUBSET, feats_raw)
            dmat = xgb.DMatrix(test_x)
            fold_preds = np.zeros(n_test)
            for booster in self.boosters[measure]:
                fold_preds += booster.predict(dmat)
            preds[idx, :] = fold_preds / len(self.boosters[measure])

        print("done.")

        # 4. Compute ZT and Power Factor
        # predicted[0] = el_cond (S/cm)
        # predicted[1] = seebeck (µV/K)
        # predicted[2] = thermal (W/m·K)
        temps_arr = df_comp["Temp"].values
        sigma = preds[0, :]
        seebeck = preds[1, :]
        kappa = np.clip(preds[2, :], 0.05, None)  # prevent division by zero

        # PF in µW/(cm·K²) = S² [µV/K] * σ [S/cm] * 1e-12 V² * 1e6 µW/W = S²·σ / 1e6
        pf_micro = (seebeck ** 2) * sigma * 1e-6

        # ZT = S²·σ·T / κ · 1e-10
        zt = (seebeck ** 2) * sigma * temps_arr / kappa * 1e-10

        res_df = pd.DataFrame(comp_records)
        res_df["sigma (S/cm)"] = sigma.round(2)
        res_df["Seebeck (µV/K)"] = seebeck.round(2)
        res_df["kappa (W/m·K)"] = kappa.round(3)
        res_df["PF (µW/cm·K²)"] = pf_micro.round(2)
        res_df["ZT"] = zt.round(4)

        # 5. Aggregate summary per chemical composition system
        summary_records = []
        for (formula, has_na, dop), grp in res_df.groupby(["formula", "has_Na", "dopant"]):
            best_idx = grp["ZT"].idxmax()
            best_row = grp.loc[best_idx]
            summary_records.append({
                "Composition": formula,
                "Dopant": dop,
                "Na_Co-doped": "Yes (1% Na)" if has_na else "No (pure doped)",
                "Peak_ZT": float(best_row["ZT"]),
                "T_opt (K)": int(best_row["T"]),
                "PF @ T_opt (µW/cm·K²)": float(best_row["PF (µW/cm·K²)"]),
                "κ @ T_opt (W/m·K)": float(best_row["kappa (W/m·K)"]),
                "σ @ T_opt (S/cm)": float(best_row["sigma (S/cm)"]),
                "S @ T_opt (µV/K)": float(best_row["Seebeck (µV/K)"])
            })

        summary_df = pd.DataFrame(summary_records).sort_values("Peak_ZT", ascending=False).reset_index(drop=True)
        summary_df.index = summary_df.index + 1
        summary_df.index.name = "Rank"

        # 6. Announce Champion
        champ = summary_df.iloc[0]
        champ_formula = champ["Composition"]
        champ_sweep = res_df[res_df["formula"] == champ_formula]

        print("\n" + "=" * 78)
        print("  🏆 CHAMPION SnSe COMPOSITION IDENTIFIED (HIGHEST ZT)")
        print("=" * 78)
        print(f"  • Composition            : {champ_formula}")
        print(f"  • Maximum ZT (Peak ZT)   : {champ['Peak_ZT']:.3f}")
        print(f"  • Optimal Temp (T_opt)   : {champ['T_opt (K)']} K")
        print(f"  • Power Factor @ T_opt   : {champ['PF @ T_opt (µW/cm·K²)']:.2f} µW/(cm·K²)")
        print(f"  • Thermal Cond @ T_opt   : {champ['κ @ T_opt (W/m·K)']:.3f} W/(m·K)")
        print(f"  • Conductivity @ T_opt   : {champ['σ @ T_opt (S/cm)']:.2f} S/cm")
        print(f"  • Seebeck Coeff @ T_opt  : {champ['S @ T_opt (µV/K)']:.2f} µV/K")
        print("-" * 78)
        print("  Champion Temperature Sweep Profile (Selected Temperatures):")
        disp_sweep = champ_sweep[["T", "sigma (S/cm)", "Seebeck (µV/K)", "kappa (W/m·K)", "PF (µW/cm·K²)", "ZT"]]
        print("  " + disp_sweep[disp_sweep["T"].isin([300, 400, 500, 600, 700, 780, 800, 820])].to_string(index=False).replace("\n", "\n  "))
        print("=" * 78)

        # 7. Print Ranked Leaderboard
        print("\n  Top Ranked SnSe Compositions by Peak ZT:")
        print("-" * 78)
        show_cols = ["Composition", "Na_Co-doped", "Peak_ZT", "T_opt (K)", "PF @ T_opt (µW/cm·K²)", "κ @ T_opt (W/m·K)"]
        print(summary_df[show_cols].to_string())
        print("-" * 78)

        # 8. Save CSV results
        out_csv = output_csv or os.path.join(HERE, "outputdata_search", "snse_ml_pipeline_results.csv")
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        res_df.to_csv(out_csv, index=False)
        summary_csv = out_csv.replace(".csv", "_summary.csv")
        summary_df.to_csv(summary_csv)
        print(f"\n  ✓ Saved complete temperature sweeps to: {out_csv}")
        print(f"  ✓ Saved ranked leaderboard summary to:  {summary_csv}\n")

        return summary_df, res_df


def main():
    parser = argparse.ArgumentParser(description="Run SnSe_ML Full Pipeline")
    parser.add_argument("--dopants", nargs="+", default=DEFAULT_DOPANTS, help="List of dopants (e.g. Zn Ge Pb Cd Y As)")
    parser.add_argument("--ratios", nargs="+", type=float, default=None, help="Doping ratios in % (e.g. 1 2 3)")
    parser.add_argument("--start-temp", type=int, default=300, help="Start temperature in K")
    parser.add_argument("--end-temp", type=int, default=820, help="End temperature in K")
    parser.add_argument("--step-temp", type=int, default=20, help="Step temperature in K")
    args = parser.parse_args()

    pipeline = SnSeMLPipeline()
    pipeline.run_screening(
        elements=args.dopants,
        alloy_ratios=args.ratios,
        start_temp=args.start_temp,
        end_temp=args.end_temp,
        step_temp=args.step_temp
    )


if __name__ == "__main__":
    main()
