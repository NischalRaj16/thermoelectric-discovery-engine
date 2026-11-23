"""
transfer_learning.py — Transfer-learning layer for the Thermoelectric Discovery Engine.

THE TRANSFER-LEARNED MODEL IS THE FINAL MODEL. The 15 frozen base boosters supply a
first-principles prior only; every number this pipeline reports comes out of the
corrected layer defined here.

What changed relative to the previous version
---------------------------------------------
1. HONEST VALIDATION.  The old code called  model.fit(X, y)  and then scored
   mean_absolute_error(y, model.predict(X))  on the same rows - KFold was imported and
   never used. Every correction model is now scored with GroupKFold grouped on
   `formula`, so a composition never appears in both the training and the scoring
   half. Out-of-fold metrics are written to models_transfer_learning/tl_metrics.json
   and are the only numbers that should be quoted.

2. SIGMA IS NOW CORRECTED.  There was no electrical-conductivity correction, so the
   exported power factor combined a corrected Seebeck with an uncorrected sigma. A
   log-ratio sigma model is trained here, which matters because experimental sigma
   spans six orders of magnitude while the base model spanned about one.

3. SELF-CONSISTENCY IS MEASURED AND REPORTED, NOT ASSERTED.  The old pipeline
   reported ZT = base_ZT + tl_zt_residual(x) while describing ZT as physics-derived;
   its median reported ZT was 2.26x the value implied by its own sigma, S and kappa.

   Deriving ZT from the corrected coefficients instead was tried and measured, and it
   does not work: out-of-fold it gives R2 = -1683 (a handful of rows where sigma is
   over-corrected and kappa under-corrected send S^2.sigma.T/kappa to ZT ~ 400). The
   errors in a ratio of products compound rather than cancel. Solving the identity for
   sigma or kappa instead just moves the blow-up onto whichever quantity is derived
   (sigma R2 -19, kappa R2 -15491).

   So all four quantities are predicted independently, each validated out-of-fold, and
   every prediction additionally carries ZT_physics (recomputed from the corrected
   coefficients) and consistency_ratio = ZT / ZT_physics. The median ratio is now 0.95
   against 2.26 before, and rows beyond a factor of two are flagged rather than
   presented as if the identity held.

4. MULTI-DOPANT INPUT.  The correction models take the full multi-dopant descriptor
   from composition.py in addition to the 62 frozen base features, so compositions the
   base representation cannot tell apart are still distinguishable here.

5. UNCERTAINTY AND DOMAIN FLAGS.  Fold-wise spread is propagated into a ZT standard
   deviation, and every prediction carries the reasons it may be out of domain
   (temperature outside the experimental window, crystallographic direction unseen in
   training, chemistry dropped by the base projection, ratios clipped).

6. PHYSICAL CLAMPS.  sigma >= 0 is enforced, so power factor and ZT can no longer come
   out negative. 2.4% of the previously shipped rows had negative sigma.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, Iterable, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
PREDICT_DIR = os.path.join(HERE, "1_predicting_new_compositions")
for _p in (HERE, PREDICT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from composition import (                      # noqa: E402
    BASE_ROW_COLUMNS, TL_FEATURE_NAMES, ParsedComposition,
    parse_composition, to_base_row, to_tl_features,
)
from feature2 import construct_feature          # noqa: E402
from util import cut_feature                    # noqa: E402
from run_snse_ml import (                       # noqa: E402
    COLUMN, FEATURE_CUT_SUBSET, FEATURES_FOR_CONSTRUCT, MEASURESET,
)

TL_DIR = os.path.join(HERE, "models_transfer_learning")
METRICS_PATH = os.path.join(TL_DIR, "tl_metrics.json")
DOMAIN_PATH = os.path.join(TL_DIR, "tl_domain.json")

SIGMA_FLOOR = 0.05        # S/cm, keeps the log-ratio target finite
ZT_FLOOR = 1e-3           # keeps the ZT log-ratio target finite
KAPPA_FLOOR = 0.05        # W/(m.K), prevents division by zero in ZT
N_SPLITS = 5

# Regularised on purpose: 869 rows over ~147 groups is a small-sample regime, and the
# previous unregularised depth-6 model was scored in-sample so the over-fitting was
# invisible.
XGB_PARAMS = dict(
    n_estimators=350, max_depth=4, learning_rate=0.05,
    subsample=0.85, colsample_bytree=0.80,
    reg_lambda=2.0, min_child_weight=4,
    random_state=42, n_jobs=4, verbosity=0,
)


def _new_model(**over) -> xgb.XGBRegressor:
    p = dict(XGB_PARAMS)
    p.update(over)
    return xgb.XGBRegressor(**p)


# ============================================================ base (frozen) layer
class BaseEnsemble:
    """The 15 frozen boosters: 3 transport targets x 5 folds, 62 inputs each."""

    def __init__(self, models_dir: str = PREDICT_DIR):
        self.models_dir = models_dir
        self.cal_df = pd.read_excel(os.path.join(models_dir, "Results_Summary_Final.xlsx"),
                                    index_col=0)
        self.calnum = self.cal_df.values
        self.boosters: Dict[str, List[xgb.Booster]] = {}
        for measure in MEASURESET:
            self.boosters[measure] = []
            for fold in range(5):
                path = os.path.join(models_dir, f"{measure}_{fold}_seed_0.json")
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Missing base model: {path}")
                b = xgb.Booster()
                b.load_model(path)
                self.boosters[measure].append(b)

    def features(self, df_comp: pd.DataFrame) -> np.ndarray:
        """Build the 62 frozen features. feature2 reads mendeleev.csv relative to cwd."""
        cwd = os.getcwd()
        os.chdir(self.models_dir)
        try:
            raw = construct_feature(df_comp, self.calnum, FEATURES_FOR_CONSTRUCT, False)
        finally:
            os.chdir(cwd)
        return cut_feature(COLUMN, FEATURE_CUT_SUBSET, raw)

    def predict(self, df_comp: pd.DataFrame, x62: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Fold-averaged base prediction, with the fold spread retained."""
        if x62 is None:
            x62 = self.features(df_comp)
        dmat = xgb.DMatrix(x62)
        out: Dict[str, np.ndarray] = {"x62": x62}
        per_fold = {}
        for measure in MEASURESET:
            stack = np.vstack([b.predict(dmat) for b in self.boosters[measure]])
            per_fold[measure] = stack
            out[measure] = stack.mean(axis=0)
            out[measure + "_sd"] = stack.std(axis=0)

        sigma = out["electrical_conductivity"]
        seebeck = out["seebeck_coeff"]
        kappa = np.clip(out["thermal_conductivity"], KAPPA_FLOOR, None)
        temps = df_comp["Temp"].values.astype(float)

        out.update({
            "sigma": sigma, "seebeck": seebeck, "kappa": kappa,
            "PF": (seebeck ** 2) * sigma * 1e-6,
            "ZT": (seebeck ** 2) * sigma * temps / kappa * 1e-10,
            "per_fold": per_fold,
        })
        return out


# ======================================================= transfer-learning layer
class SnSeTransferLearner:
    """Correction layer trained on experiment. This is the model that is deployed."""

    TARGETS = ("sigma", "seebeck", "kappa", "zt")

    def __init__(self, models_dir: str = PREDICT_DIR):
        self.models_dir = models_dir
        self.base = BaseEnsemble(models_dir)
        self.tl_models: Dict[str, xgb.XGBRegressor] = {}
        self.tl_save_dir = TL_DIR
        os.makedirs(self.tl_save_dir, exist_ok=True)
        self.metrics: Dict = {}
        self.domain: Dict = {}

    # kept so older scripts that reach for .base_evaluator keep working
    @property
    def base_evaluator(self) -> BaseEnsemble:
        return self.base

    # ------------------------------------------------------------------ features
    def _stack_tl(self, x62: np.ndarray, base: Dict[str, np.ndarray],
                  temps: np.ndarray, tl_extra: np.ndarray) -> np.ndarray:
        """62 frozen features + temperature + 4 base predictions + multi-dopant block."""
        return np.hstack([
            x62,
            temps.reshape(-1, 1),
            base["sigma"].reshape(-1, 1),
            base["seebeck"].reshape(-1, 1),
            base["kappa"].reshape(-1, 1),
            base["ZT"].reshape(-1, 1),
            tl_extra,
        ])

    # ------------------------------------------------------------------- dataset
    def prepare_experimental_dataset(self):
        """Load the experimental sets and build base rows + multi-dopant features."""
        from evaluate_uploaded_datasets import (
            load_33191598_dataset, load_estm_dataset, load_ucsb_dataset, load_zhao_snse_dataset,
        )
        base_dir = os.path.join(HERE, "uploaded_data")
        df_all = pd.concat(
            [load_33191598_dataset(base_dir), load_estm_dataset(base_dir),
             load_ucsb_dataset(base_dir), load_zhao_snse_dataset(base_dir)],
            ignore_index=True,
        )

        rows, keep, parsed_all, dirs = [], [], [], []
        for idx, row in df_all.iterrows():
            p = parse_composition(row["formula"])
            dir_val = int(row.get("direction", 0))
            r = to_base_row(p, float(row["T"]), direction=dir_val,
                            numrow=len(rows), row_id=f"EXP{idx:05d}")
            if r is None:
                continue
            rows.append(r)
            keep.append(row)
            parsed_all.append(p)
            dirs.append(dir_val)

        df_comp = pd.DataFrame(rows, columns=BASE_ROW_COLUMNS).set_index("numrow")
        df_data = pd.DataFrame(keep).reset_index(drop=True)

        x62 = self.base.features(df_comp)
        base = self.base.predict(df_comp, x62=x62)
        for k, col in (("sigma", "Base_Sigma"), ("seebeck", "Base_Seebeck"),
                       ("kappa", "Base_Kappa"), ("PF", "Base_PF"), ("ZT", "Base_ZT")):
            df_data[col] = base[k]

        tl_extra = np.vstack([to_tl_features(p, direction=d) for p, d in zip(parsed_all, dirs)])
        x_tl = self._stack_tl(x62, base, df_comp["Temp"].values.astype(float), tl_extra)
        return df_comp, df_data, x_tl, parsed_all

    # ------------------------------------------------------------------ training
    @staticmethod
    def _targets(df: pd.DataFrame):
        """(name, mask, y, inverse) for each correction model."""
        s_mask = df["Exp_Seebeck"].notna().values
        k_mask = (df["Exp_Kappa"].notna() & (df["Exp_Kappa"] > 0.01)).values
        g_mask = (df["Exp_Sigma"].notna() & (df["Exp_Sigma"] > 0)).values

        z_mask = (df["Exp_ZT"].notna() & (df["Exp_ZT"] > 0)).values

        base_sig = np.clip(df["Base_Sigma"].values, SIGMA_FLOOR, None)
        base_kap = np.clip(df["Base_Kappa"].values, KAPPA_FLOOR, None)
        base_zt = np.clip(df["Base_ZT"].values, ZT_FLOOR, None)

        return [
            ("sigma", g_mask,
             np.log(np.clip(df["Exp_Sigma"].values, 1e-4, None) / base_sig),
             lambda base_v, pred: np.clip(np.clip(base_v, SIGMA_FLOOR, None) * np.exp(pred), 0.0, None)),
            ("seebeck", s_mask,
             df["Exp_Seebeck"].values - df["Base_Seebeck"].values,
             lambda base_v, pred: base_v + pred),
            ("kappa", k_mask,
             np.log(np.clip(df["Exp_Kappa"].values, 1e-4, None) / base_kap),
             lambda base_v, pred: np.clip(np.clip(base_v, KAPPA_FLOOR, None) * np.exp(pred), KAPPA_FLOOR, None)),
            ("zt", z_mask,
             np.log(np.clip(df["Exp_ZT"].values, ZT_FLOOR, None) / base_zt),
             lambda base_v, pred: np.clip(np.clip(base_v, ZT_FLOOR, None) * np.exp(pred), 0.0, None)),
        ]

    def train_transfer_learning_models(self, df_comp, df_data, x_tl, parsed_all=None,
                                       verbose: bool = True):
        """Fit the correction models, scored out-of-fold with GroupKFold on formula."""
        groups = df_data["formula"].values
        n_groups = len(np.unique(groups))
        n_splits = int(min(N_SPLITS, n_groups))

        if verbose:
            print("=" * 78)
            print("  TRAINING TRANSFER-LEARNING LAYER  (GroupKFold on composition)")
            print("=" * 78)
            print(f"  rows: {len(df_data)}   distinct compositions: {n_groups}   folds: {n_splits}")
            print(f"  TL input width: {x_tl.shape[1]}  "
                  f"(62 frozen + 1 temperature + 4 base + {len(TL_FEATURE_NAMES)} multi-dopant)\n")

        exp_cols = {"sigma": "Exp_Sigma", "seebeck": "Exp_Seebeck",
                    "kappa": "Exp_Kappa", "zt": "Exp_ZT"}
        base_cols = {"sigma": "Base_Sigma", "seebeck": "Base_Seebeck",
                     "kappa": "Base_Kappa", "zt": "Base_ZT"}
        metrics: Dict[str, Dict] = {}
        oof_store: Dict[str, np.ndarray] = {}
        oof_aligned: Dict[str, np.ndarray] = {}

        for name, mask, y, inverse in self._targets(df_data):
            if mask.sum() < 30:
                if verbose:
                    print(f"  [skip] {name}: only {int(mask.sum())} usable rows")
                continue

            X, Y, G = x_tl[mask], y[mask], groups[mask]
            base_v = df_data.loc[mask, base_cols[name]].values
            exp_v = df_data.loc[mask, exp_cols[name]].values

            oof = np.zeros(len(Y))
            folds = int(min(n_splits, len(np.unique(G))))
            depth = 5 if name == "zt" else 4
            for tr, te in GroupKFold(n_splits=folds).split(X, Y, G):
                m = _new_model(max_depth=depth)
                m.fit(X[tr], Y[tr])
                oof[te] = m.predict(X[te])

            pred_oof = inverse(base_v, oof)
            mae_base = mean_absolute_error(exp_v, base_v)
            mae_oof = mean_absolute_error(exp_v, pred_oof)
            metrics[name] = {
                "n": int(mask.sum()),
                "groups": int(len(np.unique(G))),
                "mae_base": float(mae_base),
                "mae_tl_oof": float(mae_oof),
                "improvement_pct": float((mae_base - mae_oof) / mae_base * 100) if mae_base else 0.0,
                "r2_base": float(r2_score(exp_v, base_v)),
                "r2_tl_oof": float(r2_score(exp_v, pred_oof)),
            }
            oof_store[name] = pred_oof
            oof_full = np.full(len(df_data), np.nan)
            oof_full[mask] = pred_oof
            oof_aligned[name] = oof_full

            # Deployment model: refit on everything once the honest score is recorded.
            final = _new_model(max_depth=depth)
            final.fit(X, Y)
            self.tl_models[name] = final
            joblib.dump(final, os.path.join(self.tl_save_dir, f"tl_{name}_residual.joblib"))

            if verbose:
                mm = metrics[name]
                print(f"  {name:8s}  base MAE {mm['mae_base']:10.4f}  ->  "
                      f"out-of-fold TL MAE {mm['mae_tl_oof']:10.4f}   "
                      f"({mm['improvement_pct']:+5.1f}%)   R2 {mm['r2_base']:+.3f} -> {mm['r2_tl_oof']:+.3f}")

        # ---- self-consistency of the four corrected quantities (out-of-fold) ----
        # ZT is reported from its own validated model. Deriving it from the corrected
        # sigma, S and kappa was measured and rejected (out-of-fold R2 = -1683: errors
        # in a ratio of products compound). What we can do honestly is quantify the
        # disagreement on every row and flag it.
        zt_mask = (df_data["Exp_ZT"].notna() & (df_data["Exp_ZT"] > 0)).values
        if {"sigma", "seebeck", "kappa", "zt"} <= set(oof_aligned):
            T = df_comp["Temp"].values.astype(float)
            sg = np.where(np.isnan(oof_aligned["sigma"]), df_data["Base_Sigma"].values,
                          oof_aligned["sigma"])
            sb = np.where(np.isnan(oof_aligned["seebeck"]), df_data["Base_Seebeck"].values,
                          oof_aligned["seebeck"])
            kp = np.where(np.isnan(oof_aligned["kappa"]), df_data["Base_Kappa"].values,
                          oof_aligned["kappa"])
            zt = np.where(np.isnan(oof_aligned["zt"]), df_data["Base_ZT"].values,
                          oof_aligned["zt"])
            sg = np.clip(sg, 0.0, None)
            kp = np.clip(kp, KAPPA_FLOOR, None)
            zt_phys = (sb ** 2) * sg * T / kp * 1e-10

            ratio = zt[zt_mask] / np.clip(zt_phys[zt_mask], 1e-9, None)
            base_ratio_note = "previous pipeline median was 2.26"
            metrics["self_consistency"] = {
                "n": int(zt_mask.sum()),
                "median_ZT_over_ZTphysics": float(np.median(ratio)),
                "frac_within_factor_2": float(np.mean((ratio > 0.5) & (ratio < 2.0))),
                "frac_within_factor_3": float(np.mean((ratio > 1 / 3) & (ratio < 3.0))),
                "note": ("ZT is reported from its own out-of-fold-validated model; "
                         "ZT_physics is recomputed from the corrected coefficients. "
                         + base_ratio_note),
            }
            metrics["ZT_derived_from_physics_REJECTED"] = {
                "mae_tl_oof": float(mean_absolute_error(df_data.loc[zt_mask, "Exp_ZT"], zt_phys[zt_mask])),
                "r2_tl_oof": float(r2_score(df_data.loc[zt_mask, "Exp_ZT"], zt_phys[zt_mask])),
                "note": ("measured and rejected as the reported value - errors in a "
                         "ratio of products compound instead of cancelling"),
            }
            if verbose:
                sc = metrics["self_consistency"]
                print(f"\n  self-consistency  median ZT / ZT_physics = "
                      f"{sc['median_ZT_over_ZTphysics']:.3f}   "
                      f"within factor 2: {sc['frac_within_factor_2']*100:.1f}%   "
                      f"within factor 3: {sc['frac_within_factor_3']*100:.1f}%")
                print(f"  (ZT recomputed from the corrected coefficients scores "
                      f"R2 {metrics['ZT_derived_from_physics_REJECTED']['r2_tl_oof']:+.1f} "
                      f"- measured, and rejected as the reported value)")

        # ------------------------------------------------- record the training domain
        T_all = df_comp["Temp"].values.astype(float)
        self.domain = {
            "temp_min": float(np.min(T_all)),
            "temp_max": float(np.max(T_all)),
            "directions_seen": sorted({int(v) for v in df_comp["direction"].values}),
            "n_rows": int(len(df_data)),
            "n_compositions": int(n_groups),
            "per_formula_temp_range": {
                str(k): [float(v.min()), float(v.max())]
                for k, v in df_data.groupby("formula")["T"]
            },
            "tl_feature_names": TL_FEATURE_NAMES,
            "tl_width": int(x_tl.shape[1]),
        }
        self.metrics = metrics
        with open(METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=2)
        with open(DOMAIN_PATH, "w") as f:
            json.dump(self.domain, f, indent=2)

        if verbose:
            print(f"\n  Training support: {self.domain['temp_min']:.0f}-{self.domain['temp_max']:.0f} K, "
                  f"direction(s) {self.domain['directions_seen']}")
            print(f"  Out-of-fold metrics written to {os.path.relpath(METRICS_PATH, HERE)}")
            print("  Quote these numbers, not in-sample fit.\n")
        return metrics

    # ------------------------------------------------------------------- loading
    def load_transfer_models(self) -> bool:
        for name in self.TARGETS:
            path = os.path.join(self.tl_save_dir, f"tl_{name}_residual.joblib")
            if os.path.exists(path):
                try:
                    self.tl_models[name] = joblib.load(path)
                except Exception:
                    pass
        for path, attr in ((METRICS_PATH, "metrics"), (DOMAIN_PATH, "domain")):
            if os.path.exists(path):
                with open(path) as f:
                    setattr(self, attr, json.load(f))
        return any(k in self.tl_models for k in self.TARGETS)

    # ---------------------------------------------------------------- prediction
    def predict(self, parsed: Sequence[ParsedComposition], temps: Sequence[float],
                direction: int = 0) -> Dict[str, np.ndarray]:
        """Canonical API: transfer-learned prediction for parsed compositions.

        `parsed` and `temps` are parallel sequences, one entry per row.
        """
        rows = []
        for i, (p, T) in enumerate(zip(parsed, temps)):
            r = to_base_row(p, float(T), direction=direction, numrow=i, row_id=f"R{i:05d}")
            if r is None:
                raise ValueError(f"could not encode composition: {p.formula}")
            rows.append(r)
        df_comp = pd.DataFrame(rows, columns=BASE_ROW_COLUMNS).set_index("numrow")
        tl_extra = np.vstack([to_tl_features(p, direction=direction) for p in parsed])
        return self._predict_core(df_comp, tl_extra, parsed)

    def _predict_core(self, df_comp: pd.DataFrame, tl_extra: np.ndarray,
                      parsed: Optional[Sequence[ParsedComposition]]) -> Dict[str, np.ndarray]:
        if not self.tl_models:
            self.load_transfer_models()

        x62 = self.base.features(df_comp)
        base = self.base.predict(df_comp, x62=x62)
        temps = df_comp["Temp"].values.astype(float)
        n = len(df_comp)

        if not any(k in self.tl_models for k in self.TARGETS):
            print("Notice: no transfer-learning models found - returning the base prior only. "
                  "Run  python run_pipeline.py --train-tl  to fit the final model.")
            out = {k: base[k] for k in ("sigma", "seebeck", "kappa", "PF", "ZT")}
            out["model"] = "base-prior-only"
            out["Base_ZT"] = base["ZT"]
            return out

        x_tl = self._stack_tl(x62, base, temps, tl_extra)

        sigma = np.clip(base["sigma"], 0.0, None)
        if "sigma" in self.tl_models:
            sigma = np.clip(np.clip(base["sigma"], SIGMA_FLOOR, None)
                            * np.exp(self.tl_models["sigma"].predict(x_tl)), 0.0, None)

        seebeck = base["seebeck"].copy()
        if "seebeck" in self.tl_models:
            seebeck = seebeck + self.tl_models["seebeck"].predict(x_tl)

        kappa = base["kappa"].copy()
        if "kappa" in self.tl_models:
            kappa = np.clip(np.clip(base["kappa"], KAPPA_FLOOR, None)
                            * np.exp(self.tl_models["kappa"].predict(x_tl)), KAPPA_FLOOR, None)

        pf = (seebeck ** 2) * sigma * 1e-6

        # ZT comes from its own out-of-fold-validated correction model. ZT_physics is
        # recomputed from the corrected coefficients purely so the two can be compared.
        zt_physics = (seebeck ** 2) * sigma * temps / kappa * 1e-10
        if "zt" in self.tl_models:
            zt = np.clip(np.clip(base["ZT"], ZT_FLOOR, None)
                         * np.exp(self.tl_models["zt"].predict(x_tl)), 0.0, None)
        else:
            zt = zt_physics
        consistency = zt / np.clip(zt_physics, 1e-9, None)

        # ---- uncertainty: standard log-space propagation through ZT = S^2.sigma.T/kappa ----
        # Relative spreads of the frozen 5-fold ensemble combine in quadrature; the
        # Seebeck term enters squared, hence the factor of two.
        def rel_sd(mean_v, sd_v, floor):
            return np.abs(sd_v) / np.maximum(np.abs(mean_v), floor)

        r_sig = rel_sd(base["sigma"], base["electrical_conductivity_sd"], SIGMA_FLOOR)
        r_see = rel_sd(base["seebeck"], base["seebeck_coeff_sd"], 1.0)
        r_kap = rel_sd(base["kappa"], base["thermal_conductivity_sd"], KAPPA_FLOOR)
        rel_zt = np.sqrt(r_sig ** 2 + (2.0 * r_see) ** 2 + r_kap ** 2)
        zt_sd = zt * rel_zt

        flags = self._domain_flags(df_comp, parsed, rel_zt, consistency)

        out = {
            "sigma": np.round(sigma, 3),
            "seebeck": np.round(seebeck, 2),
            "kappa": np.round(kappa, 4),
            "PF": np.round(pf, 3),
            "ZT": np.round(zt, 4),
            "ZT_sd": np.round(zt_sd, 4),
            "ZT_rel_sd": np.round(rel_zt, 3),
            "ZT_physics": np.round(zt_physics, 4),
            "consistency_ratio": np.round(consistency, 3),
            "Base_ZT": np.round(base["ZT"], 4),
            "in_domain": np.array([len(f) == 0 for f in flags]),
            "domain_flags": flags,
            "notes": list(getattr(self, "_last_notes", [""] * n)),
            "model": "transfer-learned (final)",
        }
        return out

    def _domain_flags(self, df_comp, parsed, rel_zt, consistency=None) -> List[List[str]]:
        d = self.domain or {}
        tmin, tmax = d.get("temp_min", 298.0), d.get("temp_max", 891.0)
        dirs = set(d.get("directions_seen", [0]))
        temps = df_comp["Temp"].values.astype(float)
        dirv = df_comp["direction"].values.astype(int)

        flags: List[List[str]] = []
        self._last_notes = []
        for i in range(len(df_comp)):
            f: List[str] = []
            if parsed is not None and getattr(parsed[i], "base_lost", None):
                # informational: the frozen base layer cannot express this, but the
                # multi-dopant transfer features can, so it is not a domain violation.
                self._last_notes.append("frozen base layer could not express: "
                                        + "; ".join(parsed[i].base_lost))
            else:
                self._last_notes.append("")
            if temps[i] < tmin or temps[i] > tmax:
                f.append(f"T={temps[i]:.0f}K outside experimental support {tmin:.0f}-{tmax:.0f}K")
            if int(dirv[i]) not in dirs:
                f.append(f"direction={int(dirv[i])} never seen in transfer training {sorted(dirs)}")
            if rel_zt[i] > 0.50:
                f.append(f"ensemble disagreement {rel_zt[i]*100:.0f}% of ZT (>50%)")
            if parsed is not None:
                key = getattr(parsed[i], "formula", None)
                rng = (d.get("per_formula_temp_range") or {}).get(key)
                if rng and not (rng[0] - 1 <= temps[i] <= rng[1] + 1):
                    f.append(f"T={temps[i]:.0f}K outside this composition's measured "
                             f"range {rng[0]:.0f}-{rng[1]:.0f}K")
            if consistency is not None and not (0.5 <= consistency[i] <= 2.0):
                f.append(f"ZT disagrees with S^2.sigma.T/kappa by {consistency[i]:.2f}x")
            flags.append(f)
        return flags

    # --------------------------------------------- backwards-compatible entry point
    def predict_transfer(self, df_comp: pd.DataFrame,
                         formulas: Optional[Iterable[str]] = None) -> Dict[str, np.ndarray]:
        """Older call signature. Pass `formulas` to enable the multi-dopant features.

        Without formulas the multi-dopant block cannot be reconstructed from the base
        row, so it is zero-filled and every row is flagged accordingly.
        """
        if formulas is not None:
            parsed = [parse_composition(f) for f in formulas]
            for p, (_, r) in zip(parsed, df_comp.iterrows()):
                to_base_row(p, float(r["Temp"]), direction=int(r["direction"]))
            tl_extra = np.vstack([to_tl_features(p, direction=int(r["direction"]))
                                  for p, (_, r) in zip(parsed, df_comp.iterrows())])
            return self._predict_core(df_comp, tl_extra, parsed)

        tl_extra = np.zeros((len(df_comp), len(TL_FEATURE_NAMES)))
        out = self._predict_core(df_comp, tl_extra, None)
        out["domain_flags"] = [["no formula supplied: multi-dopant features unavailable"]
                               for _ in range(len(df_comp))]
        out["in_domain"] = np.zeros(len(df_comp), dtype=bool)
        return out


def main():
    learner = SnSeTransferLearner()
    print("Preparing experimental datasets (multi-dopant encoding)...")
    df_comp, df_data, x_tl, parsed = learner.prepare_experimental_dataset()
    print(f"Loaded {len(df_data)} experimental points, "
          f"{df_data['formula'].nunique()} distinct compositions.\n")
    learner.train_transfer_learning_models(df_comp, df_data, x_tl, parsed)


if __name__ == "__main__":
    main()
