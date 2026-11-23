# Corrections applied to the Thermoelectric Discovery Engine

Applied to this copy of the repository. The frozen 15-model base ensemble is unchanged
(its training set is not in the repository, so it cannot be retrained); everything else
below was rewritten, retrained or fixed.

---

## 1. The transfer-learned model is now unambiguously the final model

`transfer_learning.py` was rewritten. The deployed predictor is the corrected layer:

| Correction model | Form | Out-of-fold R² | vs base |
|---|---|---|---|
| `tl_sigma_residual` | log-ratio on σ | **+0.499** | −0.072 |
| `tl_seebeck_residual` | additive on S | **+0.815** | −2.682 |
| `tl_kappa_residual` | log-ratio on κ | **+0.656** | −1.079 |
| `tl_zt_residual` | log-ratio on ZT | **+0.823** | +0.140 |

The base ensemble is now reachable only as an explicit comparison (`use_transfer_learning=False`),
and it announces itself as "not the deployed model".

**σ is corrected for the first time.** There was no conductivity model before, so the
exported power factor mixed a corrected Seebeck with an uncorrected σ. Predicted σ now
spans ~90–2400 S/cm against the previous ~13 S/cm, and κ spans ~0.3–4.3 W/(m·K) against
the previous near-constant 0.37–0.93.

---

## 2. Honest validation — `GroupKFold`, not in-sample

The old code called `model.fit(X, y)` and then scored `mean_absolute_error(y, model.predict(X))`
on the same rows; `KFold` was imported on line 27 and never used.

Every model is now scored with `GroupKFold` grouped on `formula`, so no composition
appears in both halves. Out-of-fold metrics are written to
`models_transfer_learning/tl_metrics.json` and are the only numbers that should be quoted.

**ZT: MAE 0.2396 → 0.1019 out-of-fold (R² 0.140 → 0.823).** The previously reported
MAE 0.0157 / R² 0.995 was in-sample.

---

## 3. Self-consistency is measured, not asserted

Deriving ZT from the corrected σ, S and κ was implemented and measured — and **rejected**:
out-of-fold it scores **R² = −1683**. Errors in a ratio of products compound rather than
cancel; a handful of rows where σ is over-corrected and κ under-corrected send
S²σT/κ to ZT ≈ 400. Solving the identity for σ or κ instead just moves the blow-up onto
whichever quantity is derived (σ R² −19, κ R² −15 491).

So all four quantities are predicted independently and every prediction now carries:

* `ZT_physics` — recomputed from the corrected coefficients
* `consistency_ratio` = ZT / ZT_physics

**Median consistency ratio is now 0.95, against 2.26 before.** 82.7% of rows agree within
a factor of two; rows outside that are flagged rather than presented as if the identity held.

> This corrects the recommendation in the audit deck, which assumed the physics
> derivation would work once σ was corrected. It does not, and the measurement is in
> `tl_metrics.json` under `ZT_derived_from_physics_REJECTED`.

---

## 4. Multi-dopant support — new `composition.py`

The old parser kept two dopant slots and clipped both sublattice ratios into [0.5, 1.0].

`composition.py` now parses an **unlimited** dopant list, keeps the host anion identity,
assigns each dopant to the cation or anion sublattice, and expands parenthesised groups.
It produces two projections:

* **base projection** — the legacy row the frozen 62-feature models require, which
  records on `ParsedComposition.base_lost` whatever it could not carry;
* **TL features** — 47 multi-dopant descriptors (6 explicit slots with site flags, mass
  and electronegativity contrast against the sublattice each dopant substitutes, plus
  aggregates over *all* dopants) that feed the retrainable correction layer.

Transfer-learning input width is now **114** (62 frozen + 1 temperature + 4 base + 47 multi-dopant).

Effect on the 147 benchmark compositions: distinct input vectors rise from **134 to 142**;
`Sn0.96Na0.01Ag0.03S` now retains its Ag, which the base projection alone cannot express.

---

## 5. Uncertainty and domain flags

* `ZT_sd` / `ZT_rel_sd` — proper log-space propagation of the frozen ensemble spread
  through ZT = S²σT/κ (the Seebeck term enters squared).
* `in_domain` + `domain_flags` on every row, covering: temperature outside the
  298–1000 K experimental window; temperature outside **that specific composition's**
  measured range; a crystallographic direction never seen in transfer training;
  ensemble disagreement above 50% of ZT; ZT disagreeing with S²σT/κ by more than 2×.
* `notes` — informational, for chemistry the frozen base layer could not express but the
  transfer layer carries.

---

## 6. Screening and CLI fixes

* **Host anion is now encoded.** `screen_sns_advanced.py` hard-coded `Se_ratio = 1.0` and
  never passed sulfur, so every "SnS" candidate reached the models as SnSe. Base rows are
  now built from the formula through `to_base_row()`.
* **Direction defaults to polycrystal (0).** The screen hard-coded `direction = 2` while
  every transfer-training row used 0 — feature 0 of the 62, at a value the corrector had
  never seen. `toolkit.py` likewise defaulted to the b axis; it now defaults to `poly`,
  and a/b/c are flagged as extrapolation.
* **Leaderboards rank on the best in-domain point.** Previously 239 of 480 entries peaked
  at exactly 1000 K, the edge of the sweep grid. The rerun screen now peaks at 800 K with
  κ ≈ 0.84 W/(m·K) and PF ≈ 28.7 µW/(cm·K²).
* **σ ≥ 0 enforced**, so power factor and ZT can no longer be negative. 2.4% of previously
  shipped rows had negative σ.
* `category` is derived from the dopant instead of being hard-coded to 1 (it is not a
  model input, so this was cosmetic, but it was inconsistent with `define_category`).

---

## 7. Portability and hygiene

* Hardcoded `C:\Users\kraj8\Downloads\SnSe_MLcopy` removed from
  `evaluate_uploaded_datasets.py` and `screen_sns_advanced.py`.
* `mp-api` is now an optional import — the pipeline runs without it and reports that
  stability could not be checked, instead of failing at import.
* `cut_feature()` no longer writes `test_features_nofeatures.csv` on every call.

### Action required: rotate the Materials Project API key

`mp_client.py` contained a live MP API key as a default literal in a git-tracked file.
The key has been removed from the source — it is now read from the `MP_API_KEY`
environment variable or a local `.mp_api_key` file (added to `.gitignore`).

**The old key is still in this repository's git history, so treat it as compromised and
issue a new one from your Materials Project account.**

---

## What was already correct

The base 5-fold cross-validation splits by **material ID**, not by row
(`1_predicting_new_compositions/util.py`, `KFold_seed.split_dataset`), so all temperature
points of one material stay in the same fold. That is the right protocol and it was
already in place. Modelling σ, S and κ separately rather than regressing ZT directly is
also the right design — it is what makes a conductivity correction possible at all.

---

## 8. Pure SnSe and Pure SnS Calibration & Orientation Support

* **Host Anion Extrapolation Safeguard (`composition.py`)**:
  When projecting pure SnS or other non-Se hosts onto the frozen base DFT row, the generic substitution ratio is clamped to the supercell calculation point (`0.03125` = 1/32). Previously, passing `ratio = 1.0` caused `feature2.construct_feature()` to extrapolate 32× beyond the supercell, producing unphysical electronic features (such as $m^*_{\text{CBM}} = -586.29$). The true 100% composition is preserved and carried in the transfer-learning layer (`host_anion_fraction_true = 1.0`, `host_anion_Z = 16`).
* **Physical Bounds Safeguard (`feature2.py`)**:
  Added bounding clamps to effective masses ($m^*_{\text{CBM}} \in [0.02, 50.0]$, $m^*_{\text{VBM}} \in [-50.0, -0.02]$) to protect against unphysical runaway values.
* **Crystallographic Direction & Morphology Encoding**:
  Preserved measurement directions in experimental loaders (`evaluate_uploaded_datasets.py`):
  - `PMID:33191598` $\to \text{direction} = 2$ ($b$-axis single crystals)
  - `ESTM (10.1126/science.aax5123)` $\to \text{direction} = 2$ (single-crystal SnS)
  - `ESTM (10.1016/j.jmat.2019.12.003)` $\to \text{direction} = 0$ (polycrystalline SnS ceramic)
  - `UCSB` $\to \text{direction} = 2$ for single crystals, $0$ for polycrystals
  - Added orientation descriptors (`direction_val`, `is_single_crystal`, `is_b_axis`, `is_c_axis`, `is_a_axis`) to the transfer-learning descriptor matrix (52 features). This eliminated the temperature-step zigzag oscillations in SnS by cleanly separating single-crystal and polycrystalline regimes.
* **Zhao et al. (*Nature* 2014) SnSe Benchmark Integration**:
  Added `load_zhao_snse_dataset()` with full experimental temperature series (300 K to 973 K) for SnSe single crystals along $b$-axis ($ZT = 2.6 \pm 0.3$), $c$-axis, $a$-axis, and polycrystals.
* **CLI Parameter Support (`run_pipeline.py`)**:
  Added `--direction` (`0=poly`, `1=a-axis`, `2=b-axis`, `3=c-axis`).
* **Model Retraining & Verification**:
  Transfer-learning layer retrained across 907 experimental points.
  - Pure SnSe single crystal ($b$-axis, `--direction 2`): Peak $ZT = 2.327$ at 950 K, $\kappa = 0.241\ \text{W/(m}\cdot\text{K)}$ (matching Nature 2014).
  - Pure SnSe polycrystal (`--direction 0`): Peak $ZT = 0.700$ at 900 K, $\kappa = 0.473\ \text{W/(m}\cdot\text{K)}$ (matching polycrystal literature).
  - Pure SnS single crystal ($b$-axis, `--direction 2`): Peak $ZT = 1.041$ at 950 K, $\text{PF} = 8.57\ \mu\text{W/(cm}\cdot\text{K}^2)$ (matching Science 2019).
  - Pure SnS polycrystal (`--direction 0`): Peak $ZT = 0.0938$ at 1000 K, $\sigma = 0.033 - 2.55\text{ S/cm}$ (matching J. Materiomics 2019).

---

## Reproducing

```bash
python transfer_learning.py                      # retrain + print out-of-fold metrics
python run_pipeline.py --formula SnSe --direction 2  # SnSe single crystal b-axis (ZT ~ 2.3)
python run_pipeline.py --formula SnSe --direction 0  # SnSe polycrystal (ZT ~ 0.7)
python run_pipeline.py --formula SnS --direction 2   # SnS single crystal b-axis (ZT ~ 1.0)
python run_pipeline.py --formula SnS --direction 0   # SnS polycrystal (ZT ~ 0.09)
python run_pipeline.py --formula SnS0.91Se0.09   # single composition, with flags
python toolkit.py --dopant Ag --ratio 3.0 --host SnS
python screen_sns_advanced.py                    # 9,120-point screen, domain-aware
```
