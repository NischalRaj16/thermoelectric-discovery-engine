# Thermoelectric Discovery Engine: Physics-Informed Multi-Valley ML, Transfer Learning & Materials Project API

An end-to-end, high-precision materials informatics framework designed to predict, optimize, and discover record-efficiency thermoelectric materials across **100 K to 1000 K**.

By integrating **first-principles density functional theory (DFT) electronic structure parameters** with **5-fold ensemble machine learning**, **transfer learning domain adaptation**, and the **Materials Project API (MP_API)**, this engine accurately captures multi-valley band convergence, defect chemistry, and thermodynamic phase stability.

---

## 🔬 Core Scientific Methodology

```
                                  [ CHEMICAL COMPOSITION ]
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
         [ FIRST-PRINCIPLES DFT DESCRIPTORS ]           [ MATERIALS PROJECT API (MP_API) ]
         • Multi-valley masses: m*(VBM1,2,3), m*(CBM1,2,3) • Energy Above Hull (ΔE_hull in eV/atom)
         • Valley energy offsets & direct bandgaps     • Spacegroup symmetry & Crystal system
         • Mendeleev elemental physical properties      • Thermodynamic synthesizability check
                      │                                               │
                      └───────────────────────┬───────────────────────┘
                                              │
                                              ▼
                             [ 5-FOLD ENSEMBLE XGBOOST ENGINE ]
                             • Electrical Conductivity: σ(T) [S/cm]
                             • Seebeck Coefficient:     S(T) [µV/K]
                             • Thermal Conductivity:    κ(T) [W/m·K]
                                              │
                                              ▼
                           [ TRANSFER LEARNING DOMAIN ADAPTATION ]
                           • Trained on experimental datasets (ESTM, UCSB, Zhao et al. Science)
                           • Calibrates real-world crystal field & carrier activation
                           • >94% MAE reduction over theoretical polycrystals
                                              │
                                              ▼
                             [ PHYSICAL TRANSPORT FORMULATION ]
                             • Power Factor: PF = S²·σ  [µW/(cm·K²)]
                             • Figure of Merit: ZT = (S²·σ·T / κ) · 10⁻¹⁰
                                              │
                                              ▼
                        [ 100 K – 1000 K TEMPERATURE SWEEP & LEADERBOARD ]
                        • Identifies Champion Compositions & Optimal Temp (T_opt)
```

---

## 🚀 Key Innovations

### 1. Multi-Valley Electronic Structure Featurization
Unlike standard black-box machine learning models that rely solely on elemental properties, this framework integrates **first-principles DFT multi-valley band descriptors** across 84 elements:
- Valence band maximum energies ($E_{\text{VBM1,2,3}}$) and effective masses ($m^*_{\text{VBM1,2,3}}$)
- Conduction band minimum energies ($E_{\text{CBM1,2,3}}$) and effective masses ($m^*_{\text{CBM1,2,3}}$)
- Direct electronic bandgaps, deformation potentials, and deep-level defect states

### 2. Independent Transport Property Modeling
Rather than predicting $ZT$ directly via arbitrary regression, the engine models the three underlying transport mechanisms independently using a 5-fold ensemble of XGBoost models:
- **Electrical Conductivity**: $\sigma(T)$ in $\text{S}/\text{cm}$
- **Seebeck Coefficient**: $S(T)$ in $\mu\text{V}/\text{K}$
- **Thermal Conductivity**: $\kappa(T)$ in $\text{W}/(\text{m}\cdot\text{K})$

The thermoelectric Figure of Merit is then strictly calculated through fundamental physics:

$$\text{Power Factor } (PF) = S^2 \sigma \times 10^{-6}\quad \left[\mu\text{W}/(\text{cm}\cdot\text{K}^2)\right]$$

$$ZT = \frac{S^2 \sigma T}{\kappa} \times 10^{-10} = \frac{PF \times 10^{-4} \times T}{\kappa}$$

### 3. Transfer Learning Domain Adaptation
Idealized DFT models severely under-predict experimental breakthroughs in high-quality single crystals (e.g., Zhao et al. *Science* 2020). Our **Transfer Learning Engine** bridges this gap:
- **Residual Domain Adaptation**: Calibrates theoretical polycrystal baselines using 870 experimental measurements from `ESTM`, `UCSB`, and single-crystal literature.
- **Accuracy Improvement**:
  - **Seebeck Error**: Drops from $272.3\ \mu\text{V}/\text{K}$ down to **$9.0\ \mu\text{V}/\text{K}$** (**$96.7\%$ improvement**)
  - **Thermal Conductivity Error**: Drops from $1.66\ \text{W/m}\cdot\text{K}$ down to **$0.098\ \text{W/m}\cdot\text{K}$** (**$94.1\%$ improvement**)
  - **ZT Error (MAE)**: Drops from $0.2396$ down to **$0.0157$** ($R^2 = 0.995$)

### 4. Materials Project API (`MP_API`) Phase Stability Screening
Integrated directly with the Materials Project database to evaluate thermodynamic synthesizability:
- Checks if a composition is a stable ground state or metastable phase ($\Delta E_{\text{hull}}$ in $\text{eV}/\text{atom}$).
- Identifies crystal spacegroups (`Pnma`, `Fm-3m`, etc.) and DFT bandgaps.
- Implements intelligent local caching (`mp_cache.json`) for instant, offline-tolerant queries.

---

## 📊 Benchmark Validations

### A. Landmark Single-Crystal Benchmark: $\text{Sn}_{0.96}\text{Na}_{0.01}\text{Ag}_{0.03}\text{S}$ (*Science* 2020)
| Temperature | Real Experiment | Base DFT Model | **Transfer-Learned Model** | Accuracy |
| :---: | :---: | :---: | :---: | :---: |
| **300 K** | $1.058$ | $0.168$ | **$1.058$** | **$99.98\%$** |
| **373 K** | $1.354$ | $0.486$ | **$1.365$** | **$99.19\%$** |
| **523 K** | $1.974$ | $0.762$ | **$1.945$** | **$98.53\%$** |
| **573 K / 700 K** | $1.997$ | $0.955$ | **$1.985$** | **$99.39\%$** |

### B. Isostructural Chalcogenide Solid Solution: $\text{SnS}_{0.91}\text{Se}_{0.09}$
| Temperature | Real Experiment | **Transfer-Learned Model** | Seebeck Exp vs Pred | Thermal Cond. Exp vs Pred |
| :---: | :---: | :---: | :---: | :---: |
| **300 K** | $0.686$ | **$0.687$** | $203.5$ vs **$208.6\ \mu\text{V}/\text{K}$** | $2.30$ vs **$2.35\ \text{W}/(\text{m}\cdot\text{K})$** |
| **723 K** | $1.424$ | **$1.185$** | $375.4$ vs **$328.4\ \mu\text{V}/\text{K}$** | $0.87$ vs **$1.01\ \text{W}/(\text{m}\cdot\text{K})$** |
| **873 K / 900 K** | $1.643$ | **$1.622$** | $379.3$ vs **$370.6\ \mu\text{V}/\text{K}$** | $0.63$ vs **$0.69\ \text{W}/(\text{m}\cdot\text{K})$** |

---

## 💻 Programmatic Toolkit Usage (No Browser Needed)

The toolkit is completely self-contained in [toolkit.py](file:///C:/Users/kraj8/Downloads/SnSe_MLcopy/toolkit.py) and runs directly from the terminal or in Python scripts/Jupyter notebooks without opening any web browser.

### 1. Predict Single Alloy Configuration via CLI
```bash
# Predict with Pb 3% and Sn vacancy 0.5% on SnSe:
python toolkit.py --dopant Pb --ratio 3.0 --vacancy 0.5

# Predict with Bi 2% on SnSe (generates 300 K – 820 K curves, CSV & high-res plot):
python toolkit.py --dopant Bi --ratio 2.0 --vacancy 0.5

# Predict on SnS host lattice:
python toolkit.py --dopant Ag --ratio 3.0 --host SnS
```

### 2. Multi-Dopant Automated Screening Sweep
Screen multiple dopants across ratios and output a ranked leaderboard:
```bash
python toolkit.py --screen
```

### 3. Python SDK / Programmatic Integration
```python
from toolkit import SnSeToolkit

toolkit = SnSeToolkit()

# Predict across 300 K to 820 K
df_results, summary = toolkit.predict(
    dopant="Pb",
    ratio=3.0,
    vacancy=0.5,
    host="SnSe",
    save_plot=True,
    export_csv=True
)

print(f"Peak ZT: {summary['max_zt']:.3f} at {summary['max_zt_temp']} K")
print(df_results.head())
```

---

## 🖥️ Unified Pipeline Console (`run_pipeline.py`)

### 1. Interactive Console Mode
Launch the menu-driven discovery interface:
```bash
python run_pipeline.py -i
```
From the menu, you can:
* Enter any chemical composition to sweep from 100 K to 1000 K.
* Screen curated benchmark datasets (`ESTM`, `UCSB`, single-crystal $\text{SnS}$).
* Query the Materials Project API for phase diagrams and thermodynamic stability.
* Re-train / update the transfer learning model with new datasets.

### 2. Direct Single-Composition Prediction
Predict transport properties and $ZT$ across the full temperature range:
```bash
python run_pipeline.py --formula Sn0.96Na0.01Ag0.03S
python run_pipeline.py --formula SnSe0.25Te0.75
python run_pipeline.py --formula SnS0.91Se0.09
```

### 3. Screen All Benchmark Datasets
Run an end-to-end evaluation across all experimental datasets and export leaderboards:
```bash
python run_pipeline.py --screen-datasets
```

### 4. Search Materials Project Database Directly
Query the Materials Project API for thermodynamic stability, spacegroups, and band gaps:
```bash
python run_pipeline.py --mp-search SnSe
python run_pipeline.py --mp-search Sn-Se-Te
```

### 5. Re-Train Transfer Learning Models
Calibrate the transfer learning models using newly added experimental data:
```bash
python run_pipeline.py --train-tl
```

---

## 📁 Repository Structure

```text
├── run_pipeline.py                 # Main unified entry point & interactive CLI
├── transfer_learning.py            # Transfer learning & domain adaptation engine
├── mp_client.py                    # Materials Project API client with local caching
├── evaluate_uploaded_datasets.py   # Benchmark evaluation & dataset screening pipeline
├── run_snse_ml.py                  # Core screening & 5-fold ensemble XGBoost pipeline
├── models_transfer_learning/       # Persisted transfer-learned XGBoost models
│   ├── tl_seebeck_residual.joblib
│   ├── tl_kappa_residual.joblib
│   └── tl_zt_residual.joblib
├── 1_predicting_new_compositions/  # First-principles DFT database & XGBoost models
│   ├── Results_Summary_Final.xlsx  # Multi-valley effective masses & bandgaps for 84 elements
│   ├── mendeleev.csv               # Mendeleev elemental physical descriptors
│   └── *.json / *.joblib           # Pre-trained 5-fold ensemble XGBoost models
├── uploaded_data/                  # Curated benchmark experimental datasets
│   ├── 33191598/                   # Landmark Science SnS:Na:Ag experimental dataset
│   ├── ESTM_Dataset/               # ESTM cleaned thermoelectric database
│   └── ucsb_thermoelectrics.json   # Curated UCSB Gaultois experimental database
└── outputdata_search/              # Output predictions, sweeps & leaderboards
    ├── uploaded_datasets_ranked_champions_leaderboard.csv
    ├── uploaded_datasets_temperature_sweeps_100K_to_1000K.csv
    └── uploaded_datasets_predictions_and_experimental_comparison.csv
```

---

## 🛠️ Requirements & Dependencies

```bash
pip install numpy pandas scikit-learn xgboost pymatgen mp-api joblib bokeh
```
* Python 3.10+ (tested on Python 3.12 64-bit)
* Materials Project API key (configured in `mp_client.py` or via `MP_API_KEY` environment variable)
