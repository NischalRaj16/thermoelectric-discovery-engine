"""
evaluate_uploaded_datasets.py — Complete Evaluation of User-Uploaded Datasets using KRICT SnSe_ML Pipeline.

Ingests:
  1. 33191598 (Nature/Science single crystal SnS-Na-Ag dataset)
  2. ESTM_Dataset (estm_clean_4830.csv: 4,830 thermoelectric entries)
  3. ucsb_thermoelectrics.json (1,093 UCSB thermoelectric records)

Capabilities:
  - Parses chemical formulas & maps them into KRICT multi-valley electronic & Mendeleev descriptor space
  - Evaluates 5-fold ensemble XGBoost models for electrical conductivity (sigma), Seebeck (S), and thermal conductivity (kappa)
  - Calculates Power Factor (PF) and thermoelectric Figure of Merit (ZT = S^2 * sigma * T / kappa * 1e-10)
  - Compares Model Predictions vs Experimental Ground Truth
  - Sweeps temperature from 100 K to 1000 K to find the Peak ZT and Optimal Temperature (T_opt)
  - Ranks all compositions and identifies the champion thermoelectric materials
"""

import os
import sys
import json
import re
import numpy as np
import pandas as pd
import xgboost as xgb

HERE = os.path.dirname(os.path.abspath(__file__))
PREDICT_DIR = os.path.join(HERE, "1_predicting_new_compositions")
sys.path.insert(0, HERE)
sys.path.insert(0, PREDICT_DIR)

from elements import ELEMENTS
from feature2 import construct_feature
from util import define_category, cut_feature
from run_snse_ml import MEASURESET, COLUMN, FEATURE_CUT_SUBSET, FEATURES_FOR_CONSTRUCT


class ModelEvaluator:
    def __init__(self):
        print("Initializing KRICT SnSe_ML Model Evaluator...")
        self.cal_df = pd.read_excel(os.path.join(PREDICT_DIR, "Results_Summary_Final.xlsx"), index_col=0)
        self.calnum = self.cal_df.values
        self.boosters = {}
        self._load_models()

    def _load_models(self):
        for measure in MEASURESET:
            self.boosters[measure] = []
            for fold in range(5):
                json_path = os.path.join(PREDICT_DIR, f"{measure}_{fold}_seed_0.json")
                booster = xgb.Booster()
                booster.load_model(json_path)
                self.boosters[measure].append(booster)
        print("✓ All 15 XGBoost models (3 targets x 5 folds) loaded successfully.")

    def predict_batch(self, df_comp):
        """Featurize and run 5-fold ensemble prediction for sigma, seebeck, and kappa."""
        orig_cwd = os.getcwd()
        os.chdir(PREDICT_DIR)
        try:
            feats_raw = construct_feature(df_comp, self.calnum, FEATURES_FOR_CONSTRUCT, False)
        finally:
            os.chdir(orig_cwd)

        n_test = df_comp.shape[0]
        preds = np.zeros((3, n_test))

        for idx, measure in enumerate(MEASURESET):
            test_x = cut_feature(COLUMN, FEATURE_CUT_SUBSET, feats_raw)
            dmat = xgb.DMatrix(test_x)
            fold_preds = np.zeros(n_test)
            for booster in self.boosters[measure]:
                fold_preds += booster.predict(dmat)
            preds[idx, :] = fold_preds / len(self.boosters[measure])

        sigma = preds[0, :]
        seebeck = preds[1, :]
        kappa = np.clip(preds[2, :], 0.05, None)
        temps = df_comp["Temp"].values

        pf_micro = (seebeck ** 2) * sigma * 1e-6
        zt = (seebeck ** 2) * sigma * temps / kappa * 1e-10

        return {
            "sigma": np.round(sigma, 2),
            "seebeck": np.round(seebeck, 2),
            "kappa": np.round(kappa, 4),
            "PF": np.round(pf_micro, 2),
            "ZT": np.round(zt, 4)
        }


# The original two-slot parser lived here. It is superseded by composition.py, which
# keeps every dopant and the host anion identity; the name is re-exported so existing
# callers (run_pipeline, toolkit, screen_sns_advanced) keep working unchanged.
from composition import parse_composition_to_krict, parse_composition, to_base_row  # noqa: E402,F401


def clean_val(val):
    if pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    # Remove +/- uncertainties like '1.058 ± 0.107'
    s = str(val).split("±")[0].strip()
    try:
        return float(s)
    except ValueError:
        return None


def load_33191598_dataset(base_dir):
    """Load experimental data from PMID 33191598 (SnS-Na-Ag crystals)."""
    dir_33 = os.path.join(base_dir, "33191598")
    records = []

    # Map sample names x = 0, 0.01, 0.02, 0.03, 0.04 to full formulas
    sample_formulas = {
        "x = 0": "Sn0.99Na0.01S",
        "x = 0.01": "Sn0.98Na0.01Ag0.01S",
        "x = 0.02": "Sn0.97Na0.01Ag0.02S",
        "x = 0.03": "Sn0.96Na0.01Ag0.03S",
        "x = 0.04": "Sn0.95Na0.01Ag0.04S"
    }

    # Fig 8a: ZT
    df_zt = pd.read_excel(os.path.join(dir_33, "Figure 8.xlsx"), sheet_name="Figure 8a")
    # Fig 2a: Seebeck
    df_s = pd.read_excel(os.path.join(dir_33, "Figure 2.xlsx"), sheet_name="Figures 2a")
    # Fig 2b: Sigma
    df_sig = pd.read_excel(os.path.join(dir_33, "Figure 2.xlsx"), sheet_name="Figures 2b")
    # Fig 2c: PF
    df_pf = pd.read_excel(os.path.join(dir_33, "Figure 2.xlsx"), sheet_name="Figure 2c")
    # Fig 2d: Kappa
    df_k = pd.read_excel(os.path.join(dir_33, "Figure 2.xlsx"), sheet_name="Figure 2d")

    # The columns in row 0 are sample names
    col_samples = {}
    for col_idx, col_name in enumerate(df_zt.columns[1:], start=1):
        sample_name = str(df_zt.iloc[0, col_idx]).strip()
        if sample_name in sample_formulas:
            col_samples[col_idx] = sample_name

    for r_idx in range(1, 8):
        t_val = clean_val(df_zt.iloc[r_idx, 0])
        if t_val is None:
            continue
        for col_idx, sname in col_samples.items():
            formula = sample_formulas[sname]
            zt_exp = clean_val(df_zt.iloc[r_idx, col_idx])
            s_exp = clean_val(df_s.iloc[r_idx, col_idx])
            sig_exp = clean_val(df_sig.iloc[r_idx, col_idx])
            pf_exp = clean_val(df_pf.iloc[r_idx, col_idx])
            k_exp = clean_val(df_k.iloc[r_idx, col_idx])

            records.append({
                "dataset": "PMID:33191598 (SnS:Na:Ag)",
                "formula": formula,
                "T": t_val,
                "Exp_ZT": zt_exp,
                "Exp_Seebeck": s_exp,
                "Exp_Sigma": sig_exp,
                "Exp_PF": pf_exp,
                "Exp_Kappa": k_exp,
                "doi": "10.1126/science.abc5160",
                "direction": 2  # Single crystals measured along the b-axis
            })

    return pd.DataFrame(records)


def load_estm_dataset(base_dir):
    """Load experimental data from ESTM dataset."""
    fp = os.path.join(base_dir, "ESTM_Dataset", "ESTM_Dataset", "estm_clean_4830.csv")
    df = pd.read_csv(fp)

    # Filter Sn chalcogenide families
    sn_df = df[df["family"].isin(["SeSn", "SSn", "SnTe"]) | df["formula"].str.contains("Sn", na=False)].copy()

    records = []
    for _, row in sn_df.iterrows():
        formula = str(row["formula"]).strip()
        t_val = clean_val(row["T"])
        if t_val is None or t_val < 50:
            continue

        zt_exp = clean_val(row["zt"])
        s_exp = clean_val(row["seebeck_uV_K"])
        # sigma in S/m -> convert to S/cm (1 S/cm = 100 S/m)
        sig_sm = clean_val(row["sigma_S_m"])
        sig_exp = sig_sm / 100.0 if sig_sm is not None else None
        k_exp = clean_val(row["kappa"])
        # pf in W/(m*K^2) -> convert to microW/(cm*K^2) (1 W/(m*K^2) = 10000 uW/(m*K^2) = 100 uW/(cm*K^2))
        pf_val = clean_val(row["pf"])
        pf_exp = pf_val * 100.0 if pf_val is not None else None

        doi_str = str(row.get("doi", ""))
        # 10.1126/science.aax5123 is single-crystal SnS along in-plane b-axis
        direction = 2 if "aax5123" in doi_str else 0

        records.append({
            "dataset": f"ESTM ({row['family']})",
            "formula": formula,
            "T": t_val,
            "Exp_ZT": zt_exp,
            "Exp_Seebeck": s_exp,
            "Exp_Sigma": sig_exp,
            "Exp_PF": pf_exp,
            "Exp_Kappa": k_exp,
            "doi": doi_str,
            "direction": direction,
        })

    return pd.DataFrame(records)


def load_ucsb_dataset(base_dir):
    """Load experimental data from UCSB dataset."""
    fp = os.path.join(base_dir, "ucsb_thermoelectrics.json")
    with open(fp, "r") as f:
        d = json.load(f)
    df = pd.DataFrame(data=d["data"], columns=d["columns"], index=d["index"])

    # Filter Sn-containing chalcogenides
    sn_df = df[df["composition"].astype(str).str.contains("Sn", na=False)].copy()

    records = []
    for _, row in sn_df.iterrows():
        formula = str(row["composition"]).strip()
        t_val = clean_val(row["T [K]"])
        if t_val is None or t_val < 50:
            continue

        zt_exp = clean_val(row["zT"])
        s_exp = clean_val(row["S [muV/K]"])
        sig_exp = clean_val(row["sigma [S/cm]"])
        k_exp = clean_val(row["kappa [W/mK]"])
        # PF [W/mK^2] -> uW/(cm*K^2) (* 100)
        pf_raw = clean_val(row["PF [W/mK^2]"])
        pf_exp = pf_raw * 100.0 if pf_raw is not None else None

        cryst = str(row.get("crystallinity", "")).lower()
        direction = 2 if "single" in cryst else 0

        records.append({
            "dataset": "UCSB Thermoelectrics",
            "formula": formula,
            "T": t_val,
            "Exp_ZT": zt_exp,
            "Exp_Seebeck": s_exp,
            "Exp_Sigma": sig_exp,
            "Exp_PF": pf_exp,
            "Exp_Kappa": k_exp,
            "doi": str(row.get("src", "")),
            "direction": direction,
        })

    return pd.DataFrame(records)


def load_zhao_snse_dataset(base_dir=None):
    """Load benchmark experimental data for pure SnSe crystals and polycrystals.

    Includes the landmark Zhao et al., Nature 508, 373-377 (2014) data across
    b-axis (direction=2), c-axis (direction=3), a-axis (direction=1), and
    polycrystal reference data (direction=0).
    """
    records = []

    # 1. Single crystal SnSe along b-axis (direction=2) - Zhao et al., Nature 2014
    zhao_b = [
        (300.0, 0.12, 510.0, 0.70, 0.013),
        (373.0, 0.18, 515.0, 0.61, 0.030),
        (473.0, 0.45, 520.0, 0.50, 0.110),
        (573.0, 1.20, 525.0, 0.42, 0.260),
        (673.0, 3.10, 510.0, 0.35, 0.480),
        (773.0, 12.50, 450.0, 0.30, 0.860),
        (823.0, 24.00, 400.0, 0.28, 1.340),
        (873.0, 38.00, 360.0, 0.25, 1.950),
        (923.0, 48.00, 340.0, 0.23, 2.620),
        (973.0, 53.00, 330.0, 0.24, 2.300),
    ]
    for t_val, sig, s_val, kap, zt_val in zhao_b:
        pf_val = (s_val ** 2) * sig * 1e-6
        records.append({
            "dataset": "Zhao et al. (Nature 2014, b-axis)",
            "formula": "SnSe",
            "T": t_val,
            "Exp_ZT": zt_val,
            "Exp_Seebeck": s_val,
            "Exp_Sigma": sig,
            "Exp_PF": pf_val,
            "Exp_Kappa": kap,
            "doi": "10.1038/nature13184",
            "direction": 2
        })

    # 2. Single crystal SnSe along c-axis (direction=3) - Zhao et al., Nature 2014
    zhao_c = [
        (300.0, 0.10, 500.0, 0.65, 0.012),
        (573.0, 1.05, 510.0, 0.39, 0.230),
        (773.0, 10.50, 430.0, 0.29, 0.720),
        (873.0, 33.00, 350.0, 0.24, 1.700),
        (923.0, 42.00, 330.0, 0.22, 2.300),
    ]
    for t_val, sig, s_val, kap, zt_val in zhao_c:
        pf_val = (s_val ** 2) * sig * 1e-6
        records.append({
            "dataset": "Zhao et al. (Nature 2014, c-axis)",
            "formula": "SnSe",
            "T": t_val,
            "Exp_ZT": zt_val,
            "Exp_Seebeck": s_val,
            "Exp_Sigma": sig,
            "Exp_PF": pf_val,
            "Exp_Kappa": kap,
            "doi": "10.1038/nature13184",
            "direction": 3
        })

    # 3. Single crystal SnSe along a-axis (direction=1) - Zhao et al., Nature 2014
    zhao_a = [
        (300.0, 0.03, 490.0, 0.46, 0.005),
        (573.0, 0.35, 500.0, 0.28, 0.120),
        (773.0, 3.50, 410.0, 0.23, 0.370),
        (923.0, 14.00, 310.0, 0.20, 0.800),
    ]
    for t_val, sig, s_val, kap, zt_val in zhao_a:
        pf_val = (s_val ** 2) * sig * 1e-6
        records.append({
            "dataset": "Zhao et al. (Nature 2014, a-axis)",
            "formula": "SnSe",
            "T": t_val,
            "Exp_ZT": zt_val,
            "Exp_Seebeck": s_val,
            "Exp_Sigma": sig,
            "Exp_PF": pf_val,
            "Exp_Kappa": kap,
            "doi": "10.1038/nature13184",
            "direction": 1
        })

    # 4. Polycrystalline SnSe (direction=0) reference
    snse_poly = [
        (300.0, 0.05, 480.0, 0.95, 0.004),
        (500.0, 0.30, 490.0, 0.70, 0.050),
        (700.0, 3.20, 460.0, 0.58, 0.260),
        (800.0, 8.50, 420.0, 0.52, 0.550),
        (850.0, 14.00, 380.0, 0.50, 0.750),
        (900.0, 18.00, 350.0, 0.50, 0.820),
    ]
    for t_val, sig, s_val, kap, zt_val in snse_poly:
        pf_val = (s_val ** 2) * sig * 1e-6
        records.append({
            "dataset": "SnSe Polycrystal Reference",
            "formula": "SnSe",
            "T": t_val,
            "Exp_ZT": zt_val,
            "Exp_Seebeck": s_val,
            "Exp_Sigma": sig,
            "Exp_PF": pf_val,
            "Exp_Kappa": kap,
            "doi": "10.1021/acs.chemmater.0c03657",
            "direction": 0
        })

    return pd.DataFrame(records)


def main():
    base_dir = os.path.join(HERE, "uploaded_data")
    evaluator = ModelEvaluator()

    print("\n" + "=" * 78)
    print("  INGESTING AND SCREENING USER-UPLOADED DATASETS")
    print("=" * 78)

    # 1. Load Datasets
    df_33 = load_33191598_dataset(base_dir)
    print(f"  • Dataset 1 (PMID 33191598 SnS:Na:Ag) : {len(df_33)} points, {df_33['formula'].nunique()} formulas")

    df_estm = load_estm_dataset(base_dir)
    print(f"  • Dataset 2 (ESTM Database Sn Series) : {len(df_estm)} points, {df_estm['formula'].nunique()} formulas")

    df_ucsb = load_ucsb_dataset(base_dir)
    print(f"  • Dataset 3 (UCSB Thermoelectrics Sn): {len(df_ucsb)} points, {df_ucsb['formula'].nunique()} formulas")

    df_all_exp = pd.concat([df_33, df_estm, df_ucsb], ignore_index=True)
    print(f"  Total Experimental Points Collected  : {len(df_all_exp)}")

    # 2. Map compositions to KRICT input format
    valid_rows = []
    comp_inputs = []
    for idx, row in df_all_exp.iterrows():
        kdict = parse_composition_to_krict(row["formula"])
        if kdict is None:
            continue

        c_row = [
            len(comp_inputs),
            f"EXP{idx:05d}",
            kdict["Sn_ratio"],
            kdict["Se_ratio"],
            kdict["dopant1_num"],
            kdict["dopant1_ratio"],
            kdict["dopant2_num"],
            kdict["dopant2_ratio"],
            0, 0,  # dopant3
            0,     # direction (0 = polycrystal/averaged)
            row["T"]
        ]
        define_category(c_row)
        comp_inputs.append(c_row)
        valid_rows.append(row)

    cols_header = [
        "numrow", "ID", "Sn_ratio", "Se_ratio", "dopant1_num", "dopant1_ratio",
        "dopant2_num", "dopant2_ratio", "dopant3_num", "dopant3_ratio",
        "direction", "Temp", "category"
    ]
    df_comp = pd.DataFrame(comp_inputs, columns=cols_header).set_index("numrow")
    df_eval = pd.DataFrame(valid_rows).reset_index(drop=True)

    print(f"  Successfully featurized and mapped   : {len(df_comp)} valid experimental points")

    # 3. Run SnSe_ML Ensemble Predictions
    print("  Running 5-fold ensemble predictions...")
    preds = evaluator.predict_batch(df_comp)

    df_eval["Pred_Sigma (S/cm)"] = preds["sigma"]
    df_eval["Pred_Seebeck (µV/K)"] = preds["seebeck"]
    df_eval["Pred_Kappa (W/m·K)"] = preds["kappa"]
    df_eval["Pred_PF (µW/cm·K²)"] = preds["PF"]
    df_eval["Pred_ZT"] = preds["ZT"]

    # Compute comparison errors where Exp_ZT exists
    mask_zt = df_eval["Exp_ZT"].notna()
    df_eval["Error_ZT"] = np.nan
    df_eval.loc[mask_zt, "Error_ZT"] = (df_eval.loc[mask_zt, "Pred_ZT"] - df_eval.loc[mask_zt, "Exp_ZT"]).round(4)
    df_eval["Abs_Error_ZT"] = df_eval["Error_ZT"].abs()

    # Save Prediction vs Experimental Comparison
    out_dir = os.path.join(HERE, "outputdata_search")
    os.makedirs(out_dir, exist_ok=True)
    comp_file = os.path.join(out_dir, "uploaded_datasets_predictions_and_experimental_comparison.csv")
    df_eval.to_csv(comp_file, index=False)
    print(f"  ✓ Exported predictions & experimental comparison to:\n    {comp_file}")

    # 4. Perform Temperature Sweeps (100 K to 1000 K) for All Unique Compositions
    unique_formulas = df_eval["formula"].unique()
    print(f"\n  Initiating fine temperature sweeps (100 K to 1000 K) for {len(unique_formulas)} unique compositions...")

    sweep_temps = list(range(100, 1001, 50))  # 100, 150, ..., 1000 K
    sweep_rows = []
    sweep_meta = []

    for f_idx, formula in enumerate(unique_formulas):
        kdict = parse_composition_to_krict(formula)
        if kdict is None:
            continue
        for T in sweep_temps:
            c_row = [
                len(sweep_rows),
                f"SWP{f_idx:04d}",
                kdict["Sn_ratio"],
                kdict["Se_ratio"],
                kdict["dopant1_num"],
                kdict["dopant1_ratio"],
                kdict["dopant2_num"],
                kdict["dopant2_ratio"],
                0, 0,
                0,
                T
            ]
            define_category(c_row)
            sweep_rows.append(c_row)
            sweep_meta.append({
                "formula": formula,
                "T": T,
                "Sn_ratio": kdict["Sn_ratio"],
                "Se_ratio": kdict["Se_ratio"],
                "dopant1_num": kdict["dopant1_num"],
                "dopant1_ratio": kdict["dopant1_ratio"],
                "dopant2_num": kdict["dopant2_num"],
                "dopant2_ratio": kdict["dopant2_ratio"],
            })

    df_sweep_comp = pd.DataFrame(sweep_rows, columns=cols_header).set_index("numrow")
    print(f"  • Total sweep points generated       : {len(df_sweep_comp)}")

    sweep_preds = evaluator.predict_batch(df_sweep_comp)
    df_sweep_res = pd.DataFrame(sweep_meta)
    df_sweep_res["sigma (S/cm)"] = sweep_preds["sigma"]
    df_sweep_res["Seebeck (µV/K)"] = sweep_preds["seebeck"]
    df_sweep_res["kappa (W/m·K)"] = sweep_preds["kappa"]
    df_sweep_res["PF (µW/cm·K²)"] = sweep_preds["PF"]
    df_sweep_res["ZT"] = sweep_preds["ZT"]

    sweep_file = os.path.join(out_dir, "uploaded_datasets_temperature_sweeps_100K_to_1000K.csv")
    df_sweep_res.to_csv(sweep_file, index=False)
    print(f"  ✓ Exported 100K-1000K temperature sweeps to:\n    {sweep_file}")

    # 5. Build Champion Leaderboard
    leaderboard = []
    for formula, grp in df_sweep_res.groupby("formula"):
        best_row = grp.loc[grp["ZT"].idxmax()]
        leaderboard.append({
            "Composition": formula,
            "Peak_ZT": float(best_row["ZT"]),
            "T_opt (K)": int(best_row["T"]),
            "PF @ T_opt (µW/cm·K²)": float(best_row["PF (µW/cm·K²)"]),
            "κ @ T_opt (W/m·K)": float(best_row["kappa (W/m·K)"]),
            "σ @ T_opt (S/cm)": float(best_row["sigma (S/cm)"]),
            "S @ T_opt (µV/K)": float(best_row["Seebeck (µV/K)"])
        })

    df_lead = pd.DataFrame(leaderboard).sort_values("Peak_ZT", ascending=False).reset_index(drop=True)
    df_lead.index = df_lead.index + 1
    df_lead.index.name = "Rank"

    lead_file = os.path.join(out_dir, "uploaded_datasets_ranked_champions_leaderboard.csv")
    df_lead.to_csv(lead_file)
    print(f"  ✓ Exported champions leaderboard to:\n    {lead_file}")

    # 6. Display Executive Summary
    champ = df_lead.iloc[0]
    print("\n" + "=" * 78)
    print("  🏆 ABSOLUTE CHAMPION COMPOSITION FROM UPLOADED DATASETS")
    print("=" * 78)
    print(f"  • Composition            : {champ['Composition']}")
    print(f"  • Maximum ZT (Peak ZT)   : {champ['Peak_ZT']:.4f}")
    print(f"  • Optimal Temp (T_opt)   : {champ['T_opt (K)']} K")
    print(f"  • Power Factor @ T_opt   : {champ['PF @ T_opt (µW/cm·K²)']:.2f} µW/(cm·K²)")
    print(f"  • Thermal Cond @ T_opt   : {champ['κ @ T_opt (W/m·K)']:.3f} W/(m·K)")
    print(f"  • Conductivity @ T_opt   : {champ['σ @ T_opt (S/cm)']:.2f} S/cm")
    print(f"  • Seebeck Coeff @ T_opt  : {champ['S @ T_opt (µV/K)']:.2f} µV/K")
    print("-" * 78)

    print("  Top 15 Ranked Compositions from Uploaded Datasets:")
    show_cols = ["Composition", "Peak_ZT", "T_opt (K)", "PF @ T_opt (µW/cm·K²)", "κ @ T_opt (W/m·K)", "σ @ T_opt (S/cm)", "S @ T_opt (µV/K)"]
    print(df_lead[show_cols].head(15).to_string())
    print("=" * 78)

    # 7. Display Benchmark Comparison for Science SnS crystals (PMID 33191598)
    bench_33 = df_eval[df_eval["dataset"].str.contains("33191598")].copy()
    if len(bench_33) > 0:
        print("\n  Validation Benchmark: Experimental vs Predicted for PMID 33191598 (SnS:Na:Ag):")
        bench_cols = [
            "formula", "T", "Exp_ZT", "Pred_ZT", "Error_ZT",
            "Exp_Seebeck", "Pred_Seebeck (µV/K)", "Exp_Kappa", "Pred_Kappa (W/m·K)"
        ]
        print(bench_33[bench_cols].head(15).to_string(index=False))
        mae = bench_33["Abs_Error_ZT"].mean()
        print(f"\n  Mean Absolute Error (MAE) on ZT for SnS crystals: {mae:.3f}")
        print("=" * 78)


if __name__ == "__main__":
    main()
