"""
toolkit.py — Programmatic & Terminal Thermoelectric ZT Prediction Toolkit for SnSe.
Implements the exact methodology and computational workflow of the TExplorer toolkit:
  - Input: Dopant element, doping ratio (%), Sn vacancy (%), and host lattice
  - Temperature sweep: 300 K to 820 K at 20 K intervals (27 discrete points)
  - First-principles DFT multi-valley descriptors (VBM/CBM masses, valley offsets)
  - 5-fold ensemble XGBoost predictions for sigma(T), S(T), kappa(T), PF(T), and ZT(T)
  - Transfer learning domain calibration + Materials Project API phase stability (Delta E_hull)
  - Saves tabular CSVs and high-resolution matplotlib publication plots (NO BROWSER NEEDED)
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend, saves directly to image files without opening a browser
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PREDICT_DIR = os.path.join(HERE, "1_predicting_new_compositions")
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if PREDICT_DIR not in sys.path:
    sys.path.insert(0, PREDICT_DIR)

from elements import ELEMENTS
from composition import parse_composition, to_base_row, BASE_ROW_COLUMNS
from transfer_learning import SnSeTransferLearner
from mp_client import get_mp_stability


class SnSeToolkit:
    """
    Self-contained programmatic toolkit for predicting and screening thermoelectric
    performance of Sn-chalcogenides without any web browser or server dependency.
    """
    def __init__(self, use_transfer_learning=True):
        self.use_transfer_learning = use_transfer_learning
        self.learner = SnSeTransferLearner()
        self.learner.load_transfer_models()
        self.output_dir = os.path.join(HERE, "outputdata_search")
        os.makedirs(self.output_dir, exist_ok=True)

    def predict(
        self,
        dopant="Pb",
        ratio=3.0,
        vacancy=0.5,
        host="SnSe",
        na_ratio=1.0,
        direction="poly",
        start_temp=300,
        end_temp=820,
        step_temp=20,
        check_mp=True,
        save_plot=True,
        export_csv=True,
        verbose=True
    ):
        """
        Run the complete prediction pipeline for a single alloy configuration.
        """
        # 1. Validate inputs
        dopant_clean = dopant.strip().capitalize()
        try:
            dopant_num = ELEMENTS[dopant_clean].number
        except Exception:
            raise ValueError(f"Unknown element symbol '{dopant}'. Please provide a valid chemical symbol (e.g. Pb, Bi, Ag, Te, S).")

        dop_frac = float(ratio) / 100.0
        vac_frac = float(vacancy) / 100.0
        na_frac = float(na_ratio) / 100.0

        # Default is polycrystal (0). Every experimental row the transfer layer was
        # trained on had direction = 0, so any other value is an extrapolation and the
        # engine will flag it rather than silently answer.
        dir_map = {"poly": 0, "a": 1, "b": 2, "c": 3}
        direction_val = dir_map.get(str(direction).lower(), 0)

        # 2. Formulate stoichiometry
        sn_frac_in_cation = max(0.0, 1.0 - dop_frac - vac_frac)
        sn_effective = (1.0 - na_frac) * sn_frac_in_cation
        dop_effective = (1.0 - na_frac) * dop_frac

        anion_elem = "Se" if host == "SnSe" else ("S" if host == "SnS" else "Te")
        if na_frac > 0:
            formula_str = f"Na{na_frac*100:.1f}%-(Sn{sn_frac_in_cation:.3f}{dopant_clean}{dop_frac:.3f}){anion_elem}"
            formula_label = f"Na{na_frac:.2f}(Sn{sn_frac_in_cation:.3f}{dopant_clean}{dop_frac:.3f})0.99{anion_elem}"
        else:
            formula_str = f"(Sn{sn_frac_in_cation:.3f}{dopant_clean}{dop_frac:.3f}){anion_elem}"
            formula_label = formula_str

        if verbose:
            print("\n" + "=" * 78)
            print(f"  SnSe TOOLKIT: PREDICTING THERMOELECTRIC PERFORMANCE")
            print("=" * 78)
            print(f"  • Composition Target    : {formula_label}")
            print(f"  • Host Matrix           : {host} (Cation = Sn, Anion = {anion_elem})")
            print(f"  • Alloying Element      : {dopant_clean} (Z = {dopant_num}) at {ratio:.1f}%")
            print(f"  • Sn Vacancy Ratio      : {vacancy:.1f}% (v = {vac_frac:.4f})")
            print(f"  • p-Type Optimization   : {'1.0% Na Co-doped' if na_frac > 0 else 'No Na (Single Doped)'}")
            print(f"  • Crystallographic Axis : {direction} ({"polycrystal - matches transfer-learning support" if direction_val==0 else "single-crystal axis - EXTRAPOLATION, will be flagged"})")
            print(f"  • Temperature Range     : {start_temp} K to {end_temp} K (step: {step_temp} K)")

        # 3. Materials Project API Phase Stability Check
        mp_info = None
        if check_mp:
            mp_query = f"Sn{dopant_clean}{anion_elem}" if dopant_clean not in [anion_elem, "Sn"] else host
            mp_info = get_mp_stability(mp_query)
            if verbose:
                print(f"  • Materials Project ID  : {mp_info.get('material_id', 'N/A')}")
                print(f"  • Spacegroup / Lattice  : {mp_info.get('symmetry', 'N/A')}")
                print(f"  • Energy Above Hull     : {mp_info.get('energy_above_hull', 'N/A')} eV/atom")
                print(f"  • Thermodynamic Status  : {mp_info.get('status', 'N/A')}")
                print(f"  • Electronic Bandgap    : {mp_info.get('band_gap', 'N/A')} eV")

        # 4. Temperature grid and prediction through the transfer-learned final model
        temps = list(range(start_temp, end_temp + 1, step_temp))

        # A parseable formula so the multi-dopant encoder keeps Na AND the dopant AND
        # the host anion, instead of the old two-slot row which dropped one of them.
        if na_frac > 0:
            parse_formula = (f"Na{na_frac:.4f}Sn{sn_effective:.4f}"
                             f"{dopant_clean}{dop_effective:.4f}{anion_elem}")
        else:
            parse_formula = f"Sn{sn_effective:.4f}{dopant_clean}{dop_effective:.4f}{anion_elem}"

        parsed = parse_composition(parse_formula)
        if not parsed.ok:
            raise ValueError(f"could not encode composition '{parse_formula}': {parsed.reason}")

        if self.use_transfer_learning:
            preds = self.learner.predict([parsed] * len(temps), temps, direction=direction_val)
        else:
            rows = [to_base_row(parse_composition(parse_formula), float(T),
                                direction=direction_val, numrow=i, row_id=f"P{i:03d}")
                    for i, T in enumerate(temps)]
            df_comp = pd.DataFrame(rows, columns=BASE_ROW_COLUMNS).set_index("numrow")
            preds = self.learner.base.predict(df_comp)

        df_results = pd.DataFrame({
            "Temp (K)": temps,
            "sigma (S/cm)": preds["sigma"],
            "Seebeck (µV/K)": preds["seebeck"],
            "kappa (W/m·K)": preds["kappa"],
            "PF (µW/cm·K²)": preds["PF"],
            "ZT": preds["ZT"]
        })
        if self.use_transfer_learning:
            df_results["ZT_uncertainty"] = preds["ZT_sd"]
            df_results["consistency_ratio"] = preds["consistency_ratio"]
        if "Base_ZT" in preds and self.use_transfer_learning:
            df_results["Base_DFT_ZT"] = preds["Base_ZT"]

        # 6. Extract Champion Metrics
        best_idx = df_results["ZT"].idxmax()
        champ_row = df_results.loc[best_idx]
        max_zt = float(champ_row["ZT"])
        max_zt_temp = int(champ_row["Temp (K)"])
        max_pf = float(df_results["PF (µW/cm·K²)"].max())

        summary = {
            "formula": formula_label,
            "host": host,
            "dopant": dopant_clean,
            "dopant_ratio": ratio,
            "sn_vacancy": vacancy,
            "max_zt": max_zt,
            "max_zt_temp": max_zt_temp,
            "max_pf": max_pf,
            "mp_info": mp_info
        }

        if verbose:
            print("-" * 78)
            print(f"  🏆 PREDICTION RESULTS SUMMARY:")
            print(f"  • Maximum ZT (Peak ZT)    : {max_zt:.4f}")
            print(f"  • Optimal Operating Temp  : {max_zt_temp} K")
            print(f"  • Power Factor @ Opt Temp : {champ_row['PF (µW/cm·K²)']:.2f} µW/(cm·K²)")
            print(f"  • Thermal Cond @ Opt Temp : {champ_row['kappa (W/m·K)']:.3f} W/(m·K)")
            print(f"  • Conductivity @ Opt Temp : {champ_row['sigma (S/cm)']:.2f} S/cm")
            print(f"  • Seebeck Coeff @ Opt Temp: {champ_row['Seebeck (µV/K)']:.2f} µV/K")
            print("-" * 78)
            print("  Temperature Profile (Sample Points):")
            sample_df = df_results[df_results["Temp (K)"].isin([300, 400, 500, 600, 700, 740, 780, 820])]
            print(sample_df.to_string(index=False))
            print("=" * 78)

        # 7. Export CSV
        if export_csv:
            safe_formula = formula_label.replace("%", "pct").replace("(", "_").replace(")", "_")
            csv_path = os.path.join(self.output_dir, f"predict_zt_{safe_formula}_{int(time.time())}.csv")
            df_results.to_csv(csv_path, index=False)
            if verbose:
                print(f"  [✓] Saved CSV data table to: {csv_path}")

        # 8. Save Matplotlib Plot (NO BROWSER - Direct Image File)
        if save_plot:
            safe_formula = formula_label.replace("%", "pct").replace("(", "_").replace(")", "_")
            plot_path = os.path.join(self.output_dir, f"plot_zt_{safe_formula}.png")
            self._save_plot(df_results, summary, plot_path)
            if verbose:
                print(f"  [✓] Saved 4-panel publication plot to: {plot_path}\n")

        return df_results, summary

    def _save_plot(self, df, summary, out_path):
        """Generates a 4-panel publication-quality figure of merit curves directly to disk."""
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), dpi=300)
        t = df["Temp (K)"]

        # 1. ZT Curve
        ax = axes[0, 0]
        ax.plot(t, df["ZT"], color="#0d9488", lw=2.5, marker="o", ms=4, label="Transfer-Learned ZT")
        if "Base_DFT_ZT" in df.columns:
            ax.plot(t, df["Base_DFT_ZT"], color="#94a3b8", lw=1.8, ls="--", label="Base DFT ZT")
        ax.scatter([summary["max_zt_temp"]], [summary["max_zt"]], color="#e11d48", s=80, zorder=5, label=f"Peak ZT: {summary['max_zt']:.3f} @ {summary['max_zt_temp']}K")
        ax.set_title("Figure of Merit (ZT) vs Temperature", fontsize=11, fontweight="bold")
        ax.set_xlabel("Temperature (K)", fontsize=10)
        ax.set_ylabel("ZT", fontsize=10)
        ax.grid(True, ls=":", alpha=0.6)
        ax.legend(fontsize=8.5, loc="upper left")

        # 2. Power Factor Curve
        ax = axes[0, 1]
        ax.plot(t, df["PF (µW/cm·K²)"], color="#d97706", lw=2.5, marker="s", ms=4)
        ax.set_title("Power Factor (PF = S²·σ) vs Temperature", fontsize=11, fontweight="bold")
        ax.set_xlabel("Temperature (K)", fontsize=10)
        ax.set_ylabel("PF [µW/(cm·K²)]", fontsize=10)
        ax.grid(True, ls=":", alpha=0.6)

        # 3. Electrical Conductivity
        ax = axes[1, 0]
        ax.plot(t, df["sigma (S/cm)"], color="#0891b2", lw=2.5, marker="^", ms=4)
        ax.set_title("Electrical Conductivity (σ) vs Temperature", fontsize=11, fontweight="bold")
        ax.set_xlabel("Temperature (K)", fontsize=10)
        ax.set_ylabel("Conductivity σ (S/cm)", fontsize=10)
        ax.grid(True, ls=":", alpha=0.6)

        # 4. Seebeck Coefficient & Thermal Conductivity
        ax = axes[1, 1]
        ax.plot(t, df["Seebeck (µV/K)"], color="#2563eb", lw=2.2, marker="d", ms=4, label="Seebeck (µV/K)")
        ax.set_title("Seebeck Coefficient (S) & Thermal Cond (κ)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Temperature (K)", fontsize=10)
        ax.set_ylabel("Seebeck S (µV/K)", fontsize=10, color="#2563eb")
        ax.tick_params(axis="y", labelcolor="#2563eb")
        ax.grid(True, ls=":", alpha=0.6)

        ax2 = ax.twinx()
        ax2.plot(t, df["kappa (W/m·K)"], color="#e11d48", lw=2.2, marker="x", ms=4, label="κ (W/m·K)")
        ax2.set_ylabel("Thermal Conductivity κ (W/m·K)", fontsize=10, color="#e11d48")
        ax2.tick_params(axis="y", labelcolor="#e11d48")

        fig.suptitle(f"Thermoelectric Prediction Profile: {summary['formula']}", fontsize=13, fontweight="bold", y=0.98)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(out_path)
        plt.close(fig)

    def screen_dopants(
        self,
        dopants=None,
        ratios=None,
        vacancy=0.5,
        host="SnSe",
        na_ratio=1.0,
        direction="poly"
    ):
        """
        Screen multiple dopant elements and ratios, returning a ranked leaderboard of Max ZT.
        """
        default_dopants = ["Pb", "Bi", "Ag", "Te", "S", "Zn", "Cd", "Ge", "In", "Sb", "Cu", "Ti"]
        dop_list = dopants or default_dopants
        ratio_list = ratios or [1.0, 2.0, 3.0, 4.0]

        print("\n" + "=" * 78)
        print(f"  SCREENING {len(dop_list)} DOPANTS ACROSS RATIOS ON {host}")
        print("=" * 78)

        records = []
        for dop in dop_list:
            for r in ratio_list:
                try:
                    df, summ = self.predict(
                        dopant=dop,
                        ratio=r,
                        vacancy=vacancy,
                        host=host,
                        na_ratio=na_ratio,
                        direction=direction,
                        check_mp=False,
                        save_plot=False,
                        export_csv=False,
                        verbose=False
                    )
                    records.append({
                        "Dopant": dop,
                        "Ratio (%)": r,
                        "Vacancy (%)": vacancy,
                        "Peak_ZT": summ["max_zt"],
                        "T_opt (K)": summ["max_zt_temp"],
                        "Peak_PF": summ["max_pf"]
                    })
                except Exception as e:
                    print(f"  Warning: Skipping {dop} {r}%: {e}")

        df_lead = pd.DataFrame(records).sort_values("Peak_ZT", ascending=False).reset_index(drop=True)
        df_lead.index = df_lead.index + 1
        df_lead.index.name = "Rank"

        print("\n  Top 15 Ranked Doped Configurations by Peak ZT:")
        print("-" * 78)
        print(df_lead.head(15).to_string())
        print("=" * 78)

        out_csv = os.path.join(self.output_dir, f"screening_leaderboard_{host}_{int(time.time())}.csv")
        df_lead.to_csv(out_csv)
        print(f"  [✓] Exported full screening leaderboard to: {out_csv}\n")
        return df_lead


def main():
    parser = argparse.ArgumentParser(description="SnSe Programmatic Thermoelectric Toolkit (No Browser Needed)")
    parser.add_argument("--dopant", default="Pb", help="Alloying atom (e.g. Pb, Bi, Ag, Te, S)")
    parser.add_argument("--ratio", type=float, default=3.0, help="Alloying ratio in percent (e.g. 3.0 for 3%)")
    parser.add_argument("--vacancy", type=float, default=0.5, help="Sn vacancy in percent (e.g. 0.5 for 0.5%)")
    parser.add_argument("--host", default="SnSe", choices=["SnSe", "SnS", "SnTe"], help="Host crystal lattice")
    parser.add_argument("--direction", default="poly", choices=["poly", "a", "b", "c"], help="Crystallographic axis. Default poly: the transfer layer was trained only on polycrystal data, so a/b/c are flagged as extrapolation.")
    parser.add_argument("--no-na", action="store_true", help="Disable 1% Na p-type co-doping")
    parser.add_argument("--no-tl", action="store_true", help="Disable transfer learning (use raw DFT models)")
    parser.add_argument("--no-mp", action="store_true", help="Disable Materials Project stability check")
    parser.add_argument("--no-plot", action="store_true", help="Disable saving 4-panel matplotlib figure to disk")
    parser.add_argument("--screen", action="store_true", help="Run multi-dopant screening sweep")
    args = parser.parse_args()

    toolkit = SnSeToolkit(use_transfer_learning=not args.no_tl)

    if args.screen:
        toolkit.screen_dopants(host=args.host, vacancy=args.vacancy, na_ratio=0.0 if args.no_na else 1.0, direction=args.direction)
    else:
        toolkit.predict(
            dopant=args.dopant,
            ratio=args.ratio,
            vacancy=args.vacancy,
            host=args.host,
            na_ratio=0.0 if args.no_na else 1.0,
            direction=args.direction,
            check_mp=not args.no_mp,
            save_plot=not args.no_plot,
            export_csv=True,
            verbose=True
        )


if __name__ == "__main__":
    main()
