"""
check_snse_te.py — Predict SnSe0.25Te0.75 using the KRICT SnSe_ML pipeline.
Evaluates both:
  1. Pure alloy: SnSe0.25Te0.75
  2. Na co-doped alloy: Na0.01Sn0.99Se0.25Te0.75 (p-type band convergence)
across temperatures from 300 K to 820 K.
"""

import os
import sys
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
from util import cut_feature, define_category

MEASURESET = ["electrical_conductivity", "seebeck_coeff", "thermal_conductivity"]

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

def evaluate_snse_te():
    cal_df = pd.read_excel(os.path.join(PREDICT_DIR, "Results_Summary_Final.xlsx"), index_col=0)
    calnum = cal_df.values

    # Load 5-fold models
    boosters = {}
    for measure in MEASURESET:
        boosters[measure] = []
        for fold in range(5):
            json_path = os.path.join(PREDICT_DIR, f"{measure}_{fold}_seed_0.json")
            b = xgb.Booster()
            b.load_model(json_path)
            boosters[measure].append(b)

    temps = list(range(300, 821, 20))
    rows = []
    metadata = []
    numrow = 0

    # 1. SnSe0.25Te0.75 (Pure alloy)
    for T in temps:
        # Se_ratio = 0.25, dopant1 = Te (52), dopant1_ratio = 0.75
        comp = [numrow, "TY_Te_pure", 1.0, 0.25, 52, 0.75, 0, 0.0, 0, 0, 0, T]
        define_category(comp)
        rows.append(comp)
        metadata.append({"formula": "SnSe0.25Te0.75", "T": T, "co-doped": "None"})
        numrow += 1

    # 2. Na0.01Sn0.99Se0.25Te0.75 (p-type optimization with 1% Na)
    for T in temps:
        # Sn_ratio = 0.99, Se_ratio = 0.25, dopant1 = Na (11, 0.01), dopant2 = Te (52, 0.75)
        comp = [numrow, "TY_Te_Na", 0.99, 0.25, 11, 0.01, 52, 0.75, 0, 0, 0, T]
        define_category(comp)
        rows.append(comp)
        metadata.append({"formula": "Na0.01Sn0.99Se0.25Te0.75", "T": T, "co-doped": "1% Na"})
        numrow += 1

    cols_header = [
        "numrow", "ID", "Sn_ratio", "Se_ratio", "dopant1_num", "dopant1_ratio",
        "dopant2_num", "dopant2_ratio", "dopant3_num", "dopant3_ratio",
        "direction", "Temp", "category"
    ]
    df_comp = pd.DataFrame(rows, columns=cols_header).set_index("numrow")

    orig_cwd = os.getcwd()
    os.chdir(PREDICT_DIR)
    try:
        feats_raw = construct_feature(df_comp, calnum, FEATURES_FOR_CONSTRUCT, False)
    finally:
        os.chdir(orig_cwd)

    n_test = df_comp.shape[0]
    preds = np.zeros((3, n_test))
    for idx, measure in enumerate(MEASURESET):
        test_x = cut_feature(COLUMN, FEATURE_CUT_SUBSET, feats_raw)
        dmat = xgb.DMatrix(test_x)
        fold_preds = np.zeros(n_test)
        for booster in boosters[measure]:
            fold_preds += booster.predict(dmat)
        preds[idx, :] = fold_preds / len(boosters[measure])

    temps_arr = df_comp["Temp"].values
    sigma = preds[0, :]
    seebeck = preds[1, :]
    kappa = np.clip(preds[2, :], 0.05, None)

    pf_micro = (seebeck ** 2) * sigma * 1e-6
    zt = (seebeck ** 2) * sigma * temps_arr / kappa * 1e-10

    res = pd.DataFrame(metadata)
    res["sigma (S/cm)"] = sigma.round(2)
    res["Seebeck (µV/K)"] = seebeck.round(2)
    res["kappa (W/m·K)"] = kappa.round(3)
    res["PF (µW/cm·K²)"] = pf_micro.round(2)
    res["ZT"] = zt.round(4)

    return res

if __name__ == "__main__":
    res = evaluate_snse_te()
    
    print("\n" + "=" * 80)
    print("      KRICT SnSe_ML EVALUATION FOR SnSe0.25Te0.75 ALLOY")
    print("=" * 80)
    
    for f in ["SnSe0.25Te0.75", "Na0.01Sn0.99Se0.25Te0.75"]:
        sub = res[res["formula"] == f]
        best_row = sub.loc[sub["ZT"].idxmax()]
        print(f"\n--- {f} ---")
        print(f"  • Maximum ZT (Peak ZT) : {best_row['ZT']:.3f} at T = {int(best_row['T'])} K")
        print(f"  • Power Factor @ Peak  : {best_row['PF (µW/cm·K²)']:.2f} µW/(cm·K²)")
        print(f"  • Thermal Cond @ Peak  : {best_row['kappa (W/m·K)']:.3f} W/(m·K)")
        print(f"  • Conductivity @ Peak  : {best_row['sigma (S/cm)']:.2f} S/cm")
        print(f"  • Seebeck Coeff @ Peak : {best_row['Seebeck (µV/K)']:.2f} µV/K")
        print("\n  Temperature Sweep Profile:")
        disp_pts = sub[sub["T"].isin([300, 400, 500, 600, 700, 780, 800, 820])]
        print("  " + disp_pts[["T", "sigma (S/cm)", "Seebeck (µV/K)", "kappa (W/m·K)", "PF (µW/cm·K²)", "ZT"]].to_string(index=False).replace("\n", "\n  "))

    print("\n" + "=" * 80)
