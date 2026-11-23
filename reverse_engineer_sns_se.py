"""
reverse_engineer_sns_se.py — Reverse Engineering and Detailed Calculation for SnS0.91Se0.09.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PREDICT_DIR = os.path.join(HERE, "1_predicting_new_compositions")
sys.path.insert(0, HERE)
sys.path.insert(0, PREDICT_DIR)

from evaluate_uploaded_datasets import parse_composition_to_krict, ModelEvaluator
from transfer_learning import SnSeTransferLearner
from mp_client import get_mp_stability, search_materials
from feature2 import construct_feature
from util import cut_feature
from run_snse_ml import FEATURES_FOR_CONSTRUCT, COLUMN, FEATURE_CUT_SUBSET

formula = "SnS0.91Se0.09"

print("=" * 78)
print(f"  REVERSE ENGINEERING & THERMOELECTRIC CALCULATION: {formula}")
print("=" * 78)

# 1. Chemical Formula Parsing & Site Stoichiometry
print("\n[STEP 1] Chemical Site Mapping & Normalization:")
kdict = parse_composition_to_krict(formula)
print(f"  • Cation Site (Sn)        : {kdict['Sn_ratio'] * 100:.1f}% Sn (fraction = {kdict['Sn_ratio']})")
print(f"  • Anion Site Main (S/Se)  : {kdict['Se_ratio'] * 100:.1f}% Main Chalcogen (fraction = {kdict['Se_ratio']})")
print(f"  • Anion Substituent/Dopant: Element Z={kdict['dopant1_num']} at {kdict['dopant1_ratio'] * 100:.1f}% (fraction = {kdict['dopant1_ratio']})")
print(f"  • Secondary Dopant        : Z={kdict['dopant2_num']} ({kdict['dopant2_ratio']})")

# 2. Materials Project API Data
print("\n[STEP 2] Materials Project Database (MP_API) Query:")
mp_sns = get_mp_stability("SnS")
mp_snse = get_mp_stability("SnSe")
print(f"  • SnS Host End-Member : ID {mp_sns['material_id']} | Sym: {mp_sns['symmetry']} | Hull: {mp_sns['energy_above_hull']} eV | Eg: {mp_sns['band_gap']} eV | {mp_sns['status']}")
print(f"  • SnSe Alloy End-Member: ID {mp_snse['material_id']} | Sym: {mp_snse['symmetry']} | Hull: {mp_snse['energy_above_hull']} eV | Eg: {mp_snse['band_gap']} eV | {mp_snse['status']}")
print("  • Crystal Phase: Both SnS and SnSe crystallize in the Orthorhombic Pnma crystal structure.")
print("    Because S and Se share the same valence (2-) and crystal symmetry, they form a continuous isostructural solid solution.")

# 3. Check Experimental ground truth in ESTM dataset
estm_p = os.path.join(HERE, "uploaded_data", "ESTM_Dataset", "ESTM_Dataset", "estm_clean_4830.csv")
if os.path.exists(estm_p):
    df_estm = pd.read_csv(estm_p)
    exp_matches = df_estm[df_estm["formula"] == formula]
    print(f"\n[STEP 3] Experimental Ground Truth from Uploaded ESTM Dataset ({len(exp_matches)} points found):")
    if len(exp_matches) > 0:
        for _, row in exp_matches.iterrows():
            print(f"  • T = {row['T']} K | Exp ZT = {row['zt']:.4f} | Exp Seebeck = {row['seebeck_uV_K']} µV/K | Exp Kappa = {row['kappa']} W/m·K | Exp Sigma = {row['sigma_S_m']/100.0:.1f} S/cm")

# 4. Feature Extraction Details
print("\n[STEP 4] DFT Electronic Structure & Mendeleev Feature Extraction:")
evaluator = ModelEvaluator()
cal_df = evaluator.cal_df
s_num = 16
se_num = 34
sn_num = 50

print(f"  • DFT Multi-Valley Parameters from Results_Summary_Final.xlsx:")
print(f"    - S  (Z=16): VBM1_E={cal_df.loc[s_num, 'VBM1_E']:.4f}, VBM1_M={cal_df.loc[s_num, 'VBM1_M']:.4f}, CBM1_E={cal_df.loc[s_num, 'CBM1_E']:.4f}, Bandgap={cal_df.loc[s_num, 'Band gap']:.4f} eV")
print(f"    - Se (Z=34): VBM1_E={cal_df.loc[se_num, 'VBM1_E']:.4f}, VBM1_M={cal_df.loc[se_num, 'VBM1_M']:.4f}, CBM1_E={cal_df.loc[se_num, 'CBM1_E']:.4f}, Bandgap={cal_df.loc[se_num, 'Band gap']:.4f} eV")
print(f"    - Sn (Z=50): VBM1_E={cal_df.loc[sn_num, 'VBM1_E']:.4f}, VBM1_M={cal_df.loc[sn_num, 'VBM1_M']:.4f}, CBM1_E={cal_df.loc[sn_num, 'CBM1_E']:.4f}, Bandgap={cal_df.loc[sn_num, 'Band gap']:.4f} eV")

# 5. Calculations at 300 K, 500 K, 700 K, 900 K
print("\n[STEP 5] 5-Fold Ensemble XGBoost Prediction & Transfer Learning Domain Adaptation:")

target_temps = [300, 500, 700, 900]
rows = []
cols_header = [
    "numrow", "ID", "Sn_ratio", "Se_ratio", "dopant1_num", "dopant1_ratio",
    "dopant2_num", "dopant2_ratio", "dopant3_num", "dopant3_ratio",
    "direction", "Temp", "category"
]

for idx, T in enumerate(target_temps):
    rows.append([
        idx, f"T{idx}",
        kdict["Sn_ratio"], kdict["Se_ratio"],
        kdict["dopant1_num"], kdict["dopant1_ratio"],
        kdict["dopant2_num"], kdict["dopant2_ratio"],
        0, 0, 0, T, 1  # polycrystal / p-type
    ])

df_comp = pd.DataFrame(rows, columns=cols_header).set_index("numrow")

learner = SnSeTransferLearner()
learner.load_transfer_models()

base_preds = learner.base_evaluator.predict_batch(df_comp)
tl_preds = learner.predict_transfer(df_comp)

results_table = []
for i, T in enumerate(target_temps):
    sigma_b = base_preds["sigma"][i]
    seebeck_b = base_preds["seebeck"][i]
    kappa_b = base_preds["kappa"][i]
    pf_b = base_preds["PF"][i]
    zt_b = base_preds["ZT"][i]

    sigma_tl = tl_preds["sigma"][i]
    seebeck_tl = tl_preds["seebeck"][i]
    kappa_tl = tl_preds["kappa"][i]
    pf_tl = tl_preds["PF"][i]
    zt_tl = tl_preds["ZT"][i]

    results_table.append({
        "Temperature": f"{T} K",
        "sigma (S/cm)": sigma_tl,
        "Seebeck (µV/K)": seebeck_tl,
        "kappa (W/m·K)": kappa_tl,
        "PF (µW/cm·K²)": pf_tl,
        "Base KRICT ZT": zt_b,
        "Transfer-Learned ZT": zt_tl
    })

df_res = pd.DataFrame(results_table)
print("\n" + "=" * 78)
print(f"  FINAL CALCULATED RESULTS FOR {formula} AT 300K, 500K, 700K, 900K")
print("=" * 78)
print(df_res.to_string(index=False))
print("=" * 78)

# Explanations of how PF and ZT are calculated
print("\n[STEP 6] Exact Mathematical Calculations:")
for i, T in enumerate(target_temps):
    s = tl_preds["seebeck"][i]
    sig = tl_preds["sigma"][i]
    k = tl_preds["kappa"][i]
    pf = (s ** 2) * sig * 1e-6
    zt_val = (s ** 2) * sig * T / k * 1e-10
    print(f"\n  At T = {T} K:")
    print(f"    • Power Factor PF = S² · σ = ({s:.2f} µV/K)² · ({sig:.2f} S/cm) · 10⁻⁶ = {pf:.2f} µW/(cm·K²)")
    print(f"    • Raw Formulated ZT = (S² · σ · T) / κ · 10⁻¹⁰ = ({s:.2f}² · {sig:.2f} · {T}) / {k:.4f} · 10⁻¹⁰ = {zt_val:.4f}")
    print(f"    • Transfer-Learned ZT = {tl_preds['ZT'][i]:.4f}  (Base KRICT ZT: {base_preds['ZT'][i]:.4f})")
