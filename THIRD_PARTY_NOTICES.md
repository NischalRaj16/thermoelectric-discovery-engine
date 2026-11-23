# Third-Party Notices and Provenance

This repository is a **derivative work**. This file records what came from
where, so that credit sits with the right people and so that anyone reusing
this code knows which terms apply to which parts.

If you only read one thing: the Apache-2.0 licence in `LICENSE` covers the
calibration layer written by Nischal Sadashivaiah. It does **not** cover the
upstream pipeline, the vendored `elements.py`, or any experimental dataset.

---

## 1. Upstream codebase — KRICT-DATA

The first commit of this repository (`6e019b5`, 2022-04-28) was authored by
**KRICT-DATA**, the Korea Research Institute of Chemical Technology. It is
preserved here unmodified as the provenance record.

That commit contributed the foundation this project builds on:

| Component | What it is |
|---|---|
| `1_feature_construct.ipynb`, `2_gbrt_model_fit.ipynb`, `3_find_highZT-*.ipynb`, `4_draw_sn_vacancy-final.ipynb`, `SUPP_1`–`SUPP_5` | The original research notebooks |
| `1_predicting_new_compositions/feature.py`, `feature2.py`, `gbrtmodel.py`, `myfunction.py`, `util.py` | Feature construction and model-fitting modules |
| `1_predicting_new_compositions/Results_Summary_Final.xlsx` | First-principles multi-valley DFT descriptors for 84 elements |
| `1_predicting_new_compositions/mendeleev.csv` | Elemental property table |
| `1_predicting_new_compositions/*_seed_0.json`, `*.joblib` | The pre-trained 5-fold gradient-boosted transport models |

The associated publication, which should be cited by anyone using this work,
is understood to be:

> Lee, Y.-L.; Lee, H.; Kim, T.; Byun, S.; Lee, Y. K.; Jang, S.; Chung, I.;
> Chang, H.; Im, J. *Data-Driven Enhancement of ZT in SnSe-Based Thermoelectric
> Systems.* Journal of the American Chemical Society **2022**, 144 (30),
> 13748–13763. https://doi.org/10.1021/jacs.2c04741

**Licence status: unresolved.** No `LICENSE` file accompanied the upstream
commit. Absent an explicit grant, the default position under copyright is that
all rights are reserved by the original authors. Redistribution of the upstream
portion of this repository has therefore not been confirmed as permitted.
Anyone relying on this repository — including its author — should obtain
written clarification from KRICT before redistributing the upstream files or
building commercial work on them. This notice exists so that the question is
visible rather than silently assumed away.

### Changes made on top of the upstream work

Required by Apache-2.0 §4(b), and useful regardless:

- Added a multi-dopant composition parser (`composition.py`) with a cap on
  host-anion substitution at the DFT supercell ratio (1/32).
- Added physical bounding of the effective-mass tensor in the feature path.
- Added five anisotropy encodings and an explicit crystallographic `--direction`
  argument, separating single-crystal from polycrystalline transport.
- Added a transfer-learning residual correction layer (`transfer_learning.py`)
  trained against experimental data, with GroupKFold validation on formula.
- Added a domain-validity guard that flags extrapolation in temperature,
  direction and ensemble spread.
- Added Materials Project integration (`mp_client.py`) for convex-hull
  stability screening.
- Added a unified CLI and interactive console (`run_pipeline.py`), a Python SDK
  (`toolkit.py`) and screening utilities.
- Added the technical report and presentation builders.

---

## 2. `elements.py` — Christoph Gohlke (BSD 3-Clause)

`elements.py` and `1_predicting_new_compositions/elements.py` (identical files)
are third-party code, redistributed under the BSD 3-Clause licence. The notice
below is reproduced from the file header as that licence requires.

```
Copyright (c) 2005-2019, Christoph Gohlke
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

---

## 3. Experimental datasets — not redistributed

The transfer-learning layer is calibrated against experimental measurements
that are **not included in this repository**. `uploaded_data/` is gitignored.
Each source carries its own terms; obtain them directly:

| Dataset | Source | Notes |
|---|---|---|
| ESTM | Published experimental thermoelectric database | Check the publisher's terms before redistribution |
| UCSB thermoelectrics (Gaultois et al.) | UCSB / Citrine curated dataset | Openly available; cite the original paper |
| SnS:Na:Ag single-crystal benchmark | Supplementary data, PMID 33191598 (*Science*, 2020) | Publisher-copyrighted supplementary material — download from the publisher, do not redistribute |
| SnSe b-axis single crystal | Zhao et al., *Nature* **2014**, 508, 373–377 | Figure data transcribed from the publication |

See the README for the directory layout these are expected in.

---

## 4. Runtime dependencies

The packages in `requirements.txt` (NumPy, pandas, scikit-learn, XGBoost,
pymatgen, mp-api, matplotlib, joblib, mendeleev, openpyxl, python-docx,
python-pptx, Pillow) are used unmodified under their own licences and are not
redistributed here.

Materials Project data retrieved at runtime through `mp-api` is subject to the
Materials Project's terms of use and licence (CC BY 4.0 at time of writing) and
requires your own API key.

---

*If you believe anything in this repository is attributed incorrectly, please
open an issue — it will be corrected.*
