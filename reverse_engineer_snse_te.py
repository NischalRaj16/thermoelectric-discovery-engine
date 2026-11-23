"""
reverse_engineer_snse_te.py — Reverse Engineering and Detailed Calculation of SnSe0.25Te0.75.
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

formula = "SnSe0.25Te0.75"

print("=" * 78)
print(f"  REVERSE ENGINEERING & THERMOELECTRIC CALCULATION: {formula}")
print("=" * 78)

# 1. Chemical Formula Parsing & Site Stoichiometry
print("\n[STEP 1] Chemical Site Mapping & Normalization:")
kdict = parse_composition_to_krict(formula)
print(f"  • Cation Site (Sn)        : {kdict['Sn_ratio'] * 100:.1f}% Sn (fraction = {kdict['Sn_ratio']})")
print(f"  • Anion Site Main (Se)    : {kdict['Se_ratio'] * 100:.1f}% Se (fraction = {kdict['Se_ratio']})")
print(f"  • Anion Substituent/Dopant: Element Z={kdict['dopant1_num']} (Te) at {kdict['dopant1_ratio'] * 100:.1f}% (fraction = {kdict['dopant1_ratio']})")
print(f"  • Secondary Dopant        : Z={kdict['dopant2_num']} ({kdict['dopant2_ratio']})")

# 2. Materials Project API Data
print("\n[STEP 2] Materials Project Database (MP_API) Query:")
mp_snse = get_mp_stability("SnSe")
mp_snte = get_mp_stability("SnTe")
print(f"  • SnSe End-Member  : ID {mp_snse['material_id']} | Sym: {mp_snse['symmetry']} | Hull: {mp_snse['energy_above_hull']} eV | Eg: {mp_snse['band_gap']} eV | {mp_snse['status']}")
print(f"  • SnTe End-Member  : ID {mp_snte['material_id']} | Sym: {mp_snte['symmetry']} | Hull: {mp_snte['energy_above_hull']} eV | Eg: {mp_snte['band_gap']} eV | {mp_snte['status']}")

# Check crystal system crossover:
print("  • Note on Crystal Structure: SnSe crystallizes in Orthorhombic Pnma, while SnTe crystallizes in Rock-Salt Fm-3m.")
print("    In SnSe0.25Te0.75 (75% Te), the system is predominantly in the cubic rock-salt / distorted rock-salt phase space with reduced lattice thermal conductivity from mass-fluctuation scattering.")

# 3. Feature Extraction Details
print("\n[STEP 3] DFT Electronic Structure & Mendeleev Feature Extraction:")
evaluator = ModelEvaluator()
cal_df = evaluator.cal_df
te_num = 52
se_num = 34
sn_num = 50

print(f"  • DFT VBM/CBM Parameters from Results_Summary_Final.xlsx:")
print(f"    - Te (Z=52): VBM1_E={cal_df.loc[te_num, 'VBM1_E']:.4f}, VBM1_M={cal_df.loc[te_num, 'VBM1_M']:.4f}, CBM1_E={cal_df.loc[te_num, 'CBM1_E']:.4f}, Bandgap={cal_df.loc[te_num, 'Band gap']:.4f} eV")
print(f"    - Se (Z=34): VBM1_E={cal_df.loc[se_num, 'VBM1_E']:.4f}, VBM1_M={cal_df.loc[se_num, 'VBM1_M']:.4f}, CBM1_E={cal_df.loc[se_num, 'CBM1_E']:.4f}, Bandgap={cal_df.loc[se_num, 'Band gap']:.4f} eV")
print(f"    - Sn (Z=50): VBM1_E={cal_df.loc[sn_num, 'VBM1_E']:.4f}, VBM1_M={cal_df.loc[sn_num, 'VBM1_M']:.4f}, CBM1_E={cal_df.loc[sn_num, 'CBM1_E']:.4f}, Bandgap={cal_df.loc[sn_num, 'Band gap']:.4f} eV")

# 4. Predictions at 300 K, 500 K, 700 K, 900 K
print("\n[STEP 4] 5-Fold Ensemble XGBoost Prediction & Transfer Learning Domain Adaptation:")

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
print("  FINAL CALCULATED RESULTS FOR SnSe0.25Te0.75 AT 300K, 500K, 700K, 900K")
print("=" * 78)
print(df_res.to_string(index=False))
print("=" * 78)

# Explanations of how PF and ZT are calculated
print("\n[STEP 5] Exact Formulas Used:")
for i, T in enumerate(target_temps):
    s = tl_preds["seebeck"][i]
    sig = tl_preds["sigma"][i]
    k = tl_preds["kappa"][i]
    pf = (s ** 2) * sig * 1e-6
    zt = (s ** 2) * sig * T / k * 1e-10
    print(f"\n  At T = {T} K:")
    print(f"    • Power Factor PF = S² · σ = ({s:.2f} µV/K)² · ({sig:.2f} S/cm) · 10⁻⁶ = {pf:.2f} µW/(cm·K²)")
    print(f"    • Figure of Merit ZT = (S² · σ · T) / κ · 10⁻¹⁰ = ({s:.2f}² · {sig:.2f} · {T}) / {k:.4f} · 10⁻¹⁰ = {zt:.4f}")
    print(f"    • Calibrated Transfer Learning ZT = {tl_preds['ZT'][i]:.4f}")
