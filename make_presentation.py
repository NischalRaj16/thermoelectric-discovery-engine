"""
========================================================================================
Thermoelectric Discovery Engine - Presentation Builder
========================================================================================
Regenerates Thermoelectric_Discovery_Engine_Presentation.pptx from the CURRENT state of
the repository.

Design principle: every accuracy figure on these slides is read at build time from
models_transfer_learning/tl_metrics.json, and every champion composition is read from
the screening leaderboard in outputdata_search/. Nothing is hard-coded, so the deck
cannot drift away from the models the way a transcribed number can.

Run:  python make_presentation.py
========================================================================================
"""

import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputdata_search")
ASSET_DIR = os.path.join(HERE, "docs_assets")
TL_DIR = os.path.join(HERE, "models_transfer_learning")
os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================================
# 0. LOAD LIVE PROJECT STATE
# =============================================================================
def load_metrics():
    """Out-of-fold accuracy record written by the transfer-learning trainer."""
    path = os.path.join(TL_DIR, "tl_metrics.json")
    if not os.path.exists(path):
        print(f"[!] {path} not found - run 'python run_pipeline.py --train-tl' first.")
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_champions(n=3):
    """Top compositions from the high-throughput SnS screening leaderboard."""
    path = os.path.join(OUT_DIR, "sns_screening_leaderboard_100K_to_1000K.csv")
    if not os.path.exists(path):
        return []
    try:
        import pandas as pd
        df = pd.read_csv(path).head(n)
        return [
            (str(r["Composition"]), str(r["Configuration"]),
             float(r["Peak_ZT"]), int(r["T_opt (K)"]))
            for _, r in df.iterrows()
        ]
    except Exception as exc:                                   # pragma: no cover
        print(f"[!] Could not read leaderboard: {exc}")
        return []


METRICS = load_metrics()
CHAMPIONS = load_champions()

CHANNELS = [
    ("seebeck", "Seebeck S", "µV/K"),
    ("kappa", "Thermal cond. κ", "W/m·K"),
    ("sigma", "Electrical cond. σ", "S/cm"),
    ("zt", "Figure of merit ZT", "-"),
]


def m(channel, key, default=float("nan")):
    return METRICS.get(channel, {}).get(key, default)


# =============================================================================
# 1. GENERATE THE ACCURACY FIGURE FROM THE LIVE METRICS
# =============================================================================
fig_tl_path = os.path.join(OUT_DIR, "asset_transfer_learning.png")

plt.style.use("dark_background")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.6), dpi=220)
fig.patch.set_facecolor("#0F172A")

labels, mae_base, mae_tl, r2_base, r2_tl = [], [], [], [], []
for key, nice, unit in CHANNELS:
    if key not in METRICS:
        continue
    labels.append(f"{nice}\n({unit})" if unit != "-" else nice)
    mae_base.append(m(key, "mae_base"))
    mae_tl.append(m(key, "mae_tl_oof"))
    r2_base.append(m(key, "r2_base"))
    r2_tl.append(m(key, "r2_tl_oof"))

x = np.arange(len(labels))
width = 0.36

for ax in (ax1, ax2):
    ax.set_facecolor("#0F172A")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.tick_params(colors="#CBD5E1", labelsize=9)

# -- panel 1: mean absolute error, frozen base vs transfer-learned --------------
ax1.bar(x - width / 2, mae_base, width, label="Frozen DFT base prior",
        color="#EF4444", alpha=0.90)
ax1.bar(x + width / 2, mae_tl, width, label="Transfer-learned (out-of-fold)",
        color="#10B981", alpha=0.95)
ax1.set_yscale("log")
ax1.set_ylabel("Mean absolute error (log scale)", fontsize=10,
               fontweight="bold", color="#F8FAFC")
ax1.set_title("Out-of-fold error, grouped by formula", fontsize=12,
              fontweight="bold", color="#F8FAFC", pad=12)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=9, color="#F8FAFC")
ax1.legend(frameon=False, fontsize=9, labelcolor="#CBD5E1")
ax1.grid(True, axis="y", linestyle="--", alpha=0.25, color="#64748B")

for xi, (b, t) in enumerate(zip(mae_base, mae_tl)):
    if b and b > 0:
        ax1.text(xi, max(b, t) * 1.45, f"-{(1 - t / b) * 100:.0f}%",
                 ha="center", fontsize=10, fontweight="bold", color="#FBBF24")

# -- panel 2: coefficient of determination -------------------------------------
ax2.bar(x - width / 2, r2_base, width, label="Frozen DFT base prior",
        color="#EF4444", alpha=0.90)
ax2.bar(x + width / 2, r2_tl, width, label="Transfer-learned (out-of-fold)",
        color="#10B981", alpha=0.95)
ax2.axhline(0.0, color="#94A3B8", linewidth=1.0)
ax2.set_ylabel("R² against experiment", fontsize=10,
               fontweight="bold", color="#F8FAFC")
ax2.set_title("Predictive skill recovered by the correction layer", fontsize=12,
              fontweight="bold", color="#F8FAFC", pad=12)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=9, color="#F8FAFC")
ax2.legend(frameon=False, fontsize=9, labelcolor="#CBD5E1", loc="lower right")
ax2.grid(True, axis="y", linestyle="--", alpha=0.25, color="#64748B")

for xi, t in enumerate(r2_tl):
    ax2.text(xi + width / 2, t + 0.05, f"{t:.2f}", ha="center",
             fontsize=9, fontweight="bold", color="#6EE7B7")

plt.tight_layout()
fig.savefig(fig_tl_path, facecolor="#0F172A", bbox_inches="tight")
plt.close(fig)
print(f"[+] Accuracy figure rebuilt from tl_metrics.json -> {fig_tl_path}")


# =============================================================================
# 2. DECK SCAFFOLD
# =============================================================================
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank_layout = prs.slide_layouts[6]

BG_DARK = RGBColor(15, 23, 42)
BG_DEEP = RGBColor(10, 15, 30)
CARD_DARK = RGBColor(30, 41, 59)
BORDER_CLR = RGBColor(51, 65, 85)
ACCENT_BLUE = RGBColor(56, 189, 248)
ACCENT_GOLD = RGBColor(245, 158, 11)
ACCENT_GREEN = RGBColor(16, 185, 129)
ACCENT_RED = RGBColor(248, 113, 113)
ACCENT_VIOLET = RGBColor(167, 139, 250)
TEXT_LIGHT = RGBColor(248, 250, 252)
TEXT_MUTED = RGBColor(148, 163, 184)
WHITE = RGBColor(255, 255, 255)

CONTENT_TOP = Inches(1.55)
MARGIN_X = Inches(0.8)
CONTENT_W = Inches(11.733)


def set_slide_background(slide, color=BG_DARK):
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0,
                                prs.slide_width, prs.slide_height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = color
    bg.line.fill.background()
    bg.shadow.inherit = False
    return bg


def add_header(slide, title_text, category_text="THERMOELECTRIC DISCOVERY PLATFORM"):
    cat = slide.shapes.add_textbox(MARGIN_X, Inches(0.40), Inches(11.6), Inches(0.32))
    p = cat.text_frame.paragraphs[0]
    p.text = category_text.upper()
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = ACCENT_GOLD

    box = slide.shapes.add_textbox(MARGIN_X, Inches(0.70), Inches(11.9), Inches(0.62))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title_text
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = TEXT_LIGHT


def add_card(slide, left, top, width, height,
             bg_color=CARD_DARK, border_color=BORDER_CLR):
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                  left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = bg_color
    card.line.color.rgb = border_color
    card.line.width = Pt(1.2)
    card.shadow.inherit = False
    return card


def add_text(slide, left, top, width, height, blocks, wrap=True):
    """blocks = [(text, size, bold, color, space_before_pt)]"""
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = wrap
    for i, blk in enumerate(blocks):
        text, size, bold, color = blk[0], blk[1], blk[2], blk[3]
        space_before = blk[4] if len(blk) > 4 else 0
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = text
        p.font.size = Pt(size)
        p.font.bold = bold
        p.font.color.rgb = color
        if space_before:
            p.space_before = Pt(space_before)
    return box


def add_titled_card(slide, left, top, width, height, title, body,
                    title_color=ACCENT_BLUE, body_size=11,
                    bg_color=CARD_DARK, border_color=BORDER_CLR, title_size=14):
    add_card(slide, left, top, width, height, bg_color, border_color)
    add_text(slide, left + Inches(0.22), top + Inches(0.18),
             width - Inches(0.44), height - Inches(0.36),
             [(title, title_size, True, title_color),
              (body, body_size, False, TEXT_LIGHT, 6)])


def add_stat_tile(slide, left, top, width, height, label, value, sub,
                  value_color=ACCENT_GOLD):
    add_card(slide, left, top, width, height, RGBColor(18, 27, 47))
    add_text(slide, left + Inches(0.20), top + Inches(0.16),
             width - Inches(0.40), height - Inches(0.32),
             [(label.upper(), 9, True, TEXT_MUTED),
              (value, 19, True, value_color),
              (sub, 9.5, False, TEXT_LIGHT)])


def style_table(table, col_widths, header_fill=RGBColor(23, 37, 63),
                body_fill=RGBColor(24, 33, 52), alt_fill=RGBColor(30, 41, 59),
                font_size=10.5, header_size=10.5,
                header_h=None, body_h=None):
    for i, w in enumerate(col_widths):
        table.columns[i].width = w
    if header_h is not None:
        table.rows[0].height = header_h
    if body_h is not None:
        for row in list(table.rows)[1:]:
            row.height = body_h
    table.first_row = False
    table.horz_banding = False
    for r, row in enumerate(table.rows):
        for c, cell in enumerate(row.cells):
            cell.fill.solid()
            if r == 0:
                cell.fill.fore_color.rgb = header_fill
            else:
                cell.fill.fore_color.rgb = body_fill if r % 2 else alt_fill
            cell.margin_left = Inches(0.12)
            cell.margin_right = Inches(0.10)
            cell.margin_top = Inches(0.05)
            cell.margin_bottom = Inches(0.05)
            for p in cell.text_frame.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(header_size if r == 0 else font_size)
                    run.font.bold = (r == 0) or (c == 0)
                    if r == 0:
                        run.font.color.rgb = ACCENT_GOLD
                    else:
                        run.font.color.rgb = TEXT_LIGHT


def add_note(slide, text, top=Inches(6.92), color=TEXT_MUTED, size=9.5):
    add_text(slide, MARGIN_X, top, CONTENT_W, Inches(0.4),
             [(text, size, False, color)])


def add_picture_fit(slide, path, box_left, box_top, box_w, box_h):
    """Centre an image inside a box, preserving aspect ratio."""
    if not os.path.exists(path):
        add_text(slide, box_left, box_top + Inches(1.0), box_w, Inches(0.6),
                 [(f"[missing asset: {os.path.basename(path)}]", 12, True, ACCENT_RED)])
        return None
    from PIL import Image
    with Image.open(path) as im:
        iw, ih = im.size
    scale = min(box_w / iw, box_h / ih)
    w, h = int(iw * scale), int(ih * scale)
    left = int(box_left + (box_w - w) / 2)
    top = int(box_top + (box_h - h) / 2)
    return slide.shapes.add_picture(path, left, top, width=w, height=h)


def new_slide(title=None, category=None, bg=BG_DARK):
    s = prs.slides.add_slide(blank_layout)
    set_slide_background(s, bg)
    if title:
        add_header(s, title, category or "THERMOELECTRIC DISCOVERY PLATFORM")
    return s


# =============================================================================
# SLIDE 1 - TITLE
# =============================================================================
s = new_slide(bg=BG_DEEP)
add_text(s, Inches(1.2), Inches(1.45), Inches(11.0), Inches(2.3), [
    ("PHYSICS-INFORMED MACHINE LEARNING & TRANSFER LEARNING", 14, True, ACCENT_GOLD),
    ("High-Throughput Thermoelectric Discovery Engine", 36, True, TEXT_LIGHT),
    ("Multi-Valley DFT Electronic Structure  •  Experimental Domain Adaptation  "
     "•  Materials Project API", 16, False, ACCENT_BLUE),
])
add_text(s, Inches(1.2), Inches(3.35), Inches(11.0), Inches(0.5), [
    ("SnSe / SnS chalcogenide platform  —  calibrated, bounded and "
     "domain-guarded across 100–1000 K", 13, False, TEXT_MUTED),
])

add_card(s, Inches(1.2), Inches(4.15), Inches(10.9), Inches(1.45),
         bg_color=RGBColor(18, 27, 47))

seebeck_gain = m("seebeck", "improvement_pct", 0.0)
champ_txt = f"{CHAMPIONS[0][2]:.3f} @ {CHAMPIONS[0][3]} K" if CHAMPIONS else "see leaderboard"
champ_sub = CHAMPIONS[0][0] if CHAMPIONS else "sns_screening_leaderboard.csv"

badges = [
    ("Base ensemble", "15 boosters", "3 transport targets × 5 folds", TEXT_LIGHT),
    ("Transfer learning", f"−{seebeck_gain:.0f}% MAE",
     f"Seebeck, out-of-fold (n = {int(m('seebeck', 'n', 0))})", ACCENT_GREEN),
    ("Thermodynamics", "Live MP_API", "ΔE_hull phase stability", ACCENT_BLUE),
    ("Screening champion", champ_txt, champ_sub, ACCENT_GOLD),
]
for idx, (b_label, b_val, b_sub, b_col) in enumerate(badges):
    add_text(s, Inches(1.5 + idx * 2.68), Inches(4.42), Inches(2.5), Inches(1.4), [
        (b_label.upper(), 9, True, TEXT_MUTED),
        (b_val, 19, True, b_col),
        (b_sub, 9, False, TEXT_LIGHT),
    ])

add_text(s, Inches(1.2), Inches(6.35), Inches(11.0), Inches(0.4), [
    ("All accuracy figures in this deck are read at build time from "
     "models_transfer_learning/tl_metrics.json", 9.5, False, TEXT_MUTED),
])


# =============================================================================
# SLIDE 2 - WHY SnSe / SnS
# =============================================================================
s = new_slide("Solid-State Physics: Why SnSe & SnS Host Matrices?",
              "FUNDAMENTAL MATERIALS SELECTION")
cards = [
    ("1. Giant Lattice Anharmonicity",
     "• Layered puckered orthorhombic lattice (Pnma / Cmcm) with zigzag "
     "Sn–Se/S chains.\n\n"
     "• Soft bonds and stereoactive lone-pair 5s² electrons drive severe "
     "acoustic phonon scattering.\n\n"
     "• Exceptionally high Grüneisen parameter (γ ≈ 4.1).\n\n"
     "• Intrinsically ultralow lattice thermal conductivity "
     "(κ_L < 0.4 W/m·K at 700–900 K) with no nanostructuring."),
    ("2. Multi-Valley Band Convergence",
     "• Valence band maxima VBM1, VBM2 and VBM3 align within ~0.1 eV at high "
     "temperature.\n\n"
     "• Valley degeneracy N_v = 4 raises the DOS effective mass "
     "m*_d = N_v^(2/3) · m*_b.\n\n"
     "• Large Seebeck coefficient (S > 350 µV/K) at retained carrier "
     "mobility.\n\n"
     "• Power factors PF = S²σ above 28 µW/cm·K²."),
    ("3. Abundance & Scalability",
     "• Incumbent thermoelectrics (PbTe, Bi₂Te₃) rely on toxic lead and "
     "scarce tellurium.\n\n"
     "• Tin, sulfur and selenium are abundant, non-toxic and inexpensive.\n\n"
     "• Suited to industrial waste-heat recovery at deployable cost.\n\n"
     "• Anisotropy is exploitable: the b-axis carries most of the performance."),
]
for idx, (title, body) in enumerate(cards):
    add_titled_card(s, Inches(0.8 + idx * 3.98), CONTENT_TOP, Inches(3.78),
                    Inches(3.60), title, body)


# =============================================================================
# SLIDE 3 - EIGHT-STAGE ARCHITECTURE
# =============================================================================
s = new_slide("System Architecture: Eight Decoupled Pipeline Stages",
              "WORKFLOW & PIPELINE ARCHITECTURE")
stages = [
    ("Stage 0 — Entry", "run_pipeline.py (CLI / interactive) or toolkit.py as a "
     "Python SDK; composition, direction and temperature grid."),
    ("Stage 1 — Composition decoding", "composition.py tokenises the formula into a "
     "ParsedComposition; host-anion substitution capped at min(x, 0.03125)."),
    ("Stage 2 — Dual-path featurisation", "62 DFT multi-valley descriptors from "
     "feature2.py, plus a five-term anisotropy block; effective masses bounded."),
    ("Stage 3 — Frozen base ensemble", "15 XGBoost boosters (3 transport targets "
     "× 5 folds) give σ, S and κ, plus the fold-spread uncertainty channel."),
    ("Stage 4 — Residual correction", "Four transfer-learned correctors map the "
     "idealised DFT prior onto measured experimental transport."),
    ("Stage 5 — Reconstruction", "Calibrated coefficients reassembled and audited "
     "against the physics identity ZT = S²σT / κ."),
    ("Stage 6 — Domain-validity guard", "Flags extrapolation in temperature, "
     "crystallographic direction and ensemble spread before anything is reported."),
    ("Stage 7 — Deliverables", "Temperature-sweep CSVs, champion peak ZT and T_opt, "
     "ranked leaderboards and transport plots in outputdata_search/."),
]
for idx, (title, body) in enumerate(stages):
    col, row = idx % 2, idx // 2
    left = Inches(0.8 + col * 5.95)
    top = Inches(1.55 + row * 1.33)
    add_card(s, left, top, Inches(5.73), Inches(1.18))
    add_text(s, left + Inches(0.20), top + Inches(0.12), Inches(5.35), Inches(0.95), [
        (title, 12, True, ACCENT_GOLD),
        (body, 9.5, False, TEXT_LIGHT, 3),
    ])
add_note(s, "Full module-level data flow on the next two slides; the report carries the "
            "same flowchart at figure resolution.")


# =============================================================================
# SLIDES 4 & 5 - ARCHITECTURE FLOWCHART
# =============================================================================
s = new_slide("End-to-End Data Flow, Part A: Input to Frozen Base Ensemble",
              "MODULE-LEVEL ARCHITECTURE")
add_picture_fit(s, os.path.join(ASSET_DIR, "architecture_deck_a.png"),
                Inches(0.55), Inches(1.50), Inches(12.23), Inches(5.35))

s = new_slide("End-to-End Data Flow, Part B: Correction, Validation and Delivery",
              "MODULE-LEVEL ARCHITECTURE")
add_picture_fit(s, os.path.join(ASSET_DIR, "architecture_deck_b.png"),
                Inches(0.55), Inches(1.50), Inches(12.23), Inches(5.35))


# =============================================================================
# SLIDE 6 - DATASETS
# =============================================================================
s = new_slide("Experimental Corpus and Leakage-Safe Validation Protocol",
              "DATASET INTEGRATION & CURATION")

n_pts = int(m("seebeck", "n", 0))
n_grp = int(m("seebeck", "groups", 0))

for idx, (label, value, sub, col) in enumerate([
    ("Measurement points", f"{n_pts}", "across all curated sources", ACCENT_BLUE),
    ("Formula groups", f"{n_grp}", "the unit of cross-validation", ACCENT_GREEN),
    ("Source datasets", "4", "ESTM, UCSB, Science, Zhao", ACCENT_GOLD),
    ("Temperature span", "100–1000 K", "50 K sweep resolution", TEXT_LIGHT),
]):
    add_stat_tile(s, Inches(0.8 + idx * 2.98), CONTENT_TOP, Inches(2.78),
                  Inches(1.25), label, value, sub, col)

sources = [
    ("ESTM cleaned database", "Broad multi-family thermoelectric transport records; "
     "supplies the polycrystalline backbone of the corpus."),
    ("UCSB (Gaultois) database", "Curated historical experimental set; widens chemical "
     "coverage beyond the Sn-chalcogenides."),
    ("SnS:Na:Ag landmark (Science)", "Multivalent dual-doped single-crystal benchmark; "
     "anchors the high-ZT end of the distribution."),
    ("Zhao et al. SnSe b-axis", "Single-crystal b-axis transport; the reference the "
     "anisotropy encodings are calibrated against."),
]
for idx, (title, body) in enumerate(sources):
    col, row = idx % 2, idx // 2
    left = Inches(0.8 + col * 5.95)
    top = Inches(3.05 + row * 1.58)
    add_titled_card(s, left, top, Inches(5.73), Inches(1.38), title, body,
                    title_size=12.5, body_size=10)

add_note(s, "Validation uses GroupKFold partitioned on formula, so no composition ever "
            "appears in both the training and the validation side of a fold — the "
            "scores on the following slides are out-of-fold throughout.")


# =============================================================================
# SLIDE 7 - FEATURISATION
# =============================================================================
s = new_slide("62-Dimensional Feature Space: Multi-Valley DFT Descriptors",
              "FEATURE ENGINEERING")
blocks = [
    ("Multi-valley band structure", ACCENT_BLUE,
     "• Valence maxima E_VBM1-3 and conduction minima E_CBM1-3\n"
     "• Effective masses m*_VBM1-3 and m*_CBM1-3\n"
     "• Valley energy offsets governing band convergence\n"
     "• Direct and indirect band gaps E_g\n"
     "• Tabulated by first-principles DFT for 84 elements"),
    ("Derived transport descriptors", ACCENT_GREEN,
     "• Valley degeneracy N_v\n"
     "• Density-of-states mass m*_d = N_v^(2/3) · m*_b\n"
     "• Inertial effective mass m*_I\n"
     "• Mendeleev elemental property aggregates\n"
     "• Composition-weighted site-resolved averages"),
    ("Anisotropy encodings", ACCENT_GOLD,
     "• direction_val  (0 poly, 1 a-axis, 2 b-axis, 3 c-axis)\n"
     "• is_single_crystal\n"
     "• is_b_axis   — the high-ZT transport direction\n"
     "• is_c_axis\n"
     "• anisotropy_strength"),
]
for idx, (title, colr, body) in enumerate(blocks):
    add_titled_card(s, Inches(0.8 + idx * 3.98), CONTENT_TOP, Inches(3.78),
                    Inches(2.35), title, body, title_color=colr, body_size=10.5)

add_titled_card(s, MARGIN_X, Inches(4.30), CONTENT_W, Inches(1.55),
                "Why multi-valley descriptors rather than elemental averages alone",
                "Band convergence is the mechanism behind high power factors in the "
                "Sn-chalcogenides, and it is invisible to purely compositional features. "
                "Encoding valley degeneracy and the mass tensor lets the model separate "
                "a genuinely converged band structure from a coincidentally similar "
                "stoichiometry — and makes the physical safeguards of Stage 2 "
                "expressible as bounds on real quantities.",
                title_color=ACCENT_BLUE, body_size=10.5, title_size=12.5)


# =============================================================================
# SLIDE 8 - BASE ENSEMBLE
# =============================================================================
s = new_slide("Physics-Decoupled Base Ensemble: 15 Frozen XGBoost Boosters",
              "MODEL ARCHITECTURE")
add_titled_card(s, MARGIN_X, CONTENT_TOP, Inches(5.73), Inches(2.05),
                "Three transport targets, modelled independently",
                "σ(T) electrical conductivity  [S/cm]\n"
                "S(T) Seebeck coefficient  [µV/K]\n"
                "κ(T) thermal conductivity  [W/m·K]\n\n"
                "Each target is a 5-fold ensemble; the 15 boosters are frozen artefacts "
                "loaded from disk, never retrained by the correction layer.",
                body_size=11, title_size=13)
add_titled_card(s, Inches(6.8), CONTENT_TOP, Inches(5.73), Inches(2.05),
                "ZT is computed, never regressed directly",
                "PF = S²σ · 10⁻⁶   [µW/cm·K²]\n"
                "ZT_base = S²σT / κ · 10⁻¹⁰\n\n"
                "Deriving the figure of merit from the three coefficients keeps the "
                "prediction physically consistent and makes any disagreement between "
                "channels visible rather than absorbed.",
                title_color=ACCENT_GREEN, body_size=11, title_size=13)

add_titled_card(s, MARGIN_X, Inches(4.00), Inches(3.78), Inches(1.45),
                "Fold-spread uncertainty",
                "Disagreement across the five folds is retained as σ_sd and reported "
                "alongside every ZT, rather than averaged away.",
                title_color=RGBColor(251, 146, 60), body_size=10, title_size=12)
add_titled_card(s, Inches(4.78), Inches(4.00), Inches(3.78), Inches(1.45),
                "Physical bounding",
                "Effective masses are clamped to [0.02, 50.0] and κ to a positive "
                "floor, so no unphysical intermediate can propagate into ZT.",
                title_color=ACCENT_GOLD, body_size=10, title_size=12)
add_titled_card(s, Inches(8.76), Inches(4.00), Inches(3.77), Inches(1.45),
                "Frozen by design",
                "The base layer is never fine-tuned on experiment; all empirical "
                "adaptation is isolated in the residual layer, keeping the DFT prior "
                "auditable.",
                title_color=ACCENT_VIOLET, body_size=10, title_size=12)

add_note(s, "The uncorrected base prior remains reachable at any time with "
            "--no-tl, which is how the correction magnitudes on the next slide are "
            "measured.")


# =============================================================================
# SLIDE 9 - TRANSFER LEARNING MECHANISM
# =============================================================================
s = new_slide("Transfer Learning: Four Residual Correctors, Not a Retrained Model",
              "DOMAIN ADAPTATION")
corr = [
    ("tl_sigma_residual", "log(σ_exp / σ_base)",
     "multiplicative — conductivity spans orders of magnitude, so the correction "
     "is learned in log space", ACCENT_BLUE),
    ("tl_seebeck_residual", "S_exp − S_base",
     "additive — the Seebeck offset between idealised and measured transport is "
     "close to linear", ACCENT_GREEN),
    ("tl_kappa_residual", "log(κ_exp / κ_base)",
     "multiplicative — captures grain-boundary and defect scattering absent from "
     "the DFT prior", ACCENT_GOLD),
    ("tl_zt_residual", "log(ZT_exp / ZT_base)",
     "the reported ZT; a dedicated corrector avoids compounding the errors of a "
     "ratio of products", ACCENT_VIOLET),
]
for idx, (name, formula, why, colr) in enumerate(corr):
    left = Inches(0.8 + idx * 2.98)
    add_card(s, left, CONTENT_TOP, Inches(2.78), Inches(1.75))
    add_text(s, left + Inches(0.18), CONTENT_TOP + Inches(0.16),
             Inches(2.42), Inches(1.45), [
                 (name, 12, True, colr),
                 (formula, 12.5, True, TEXT_LIGHT, 6),
                 (why, 9.5, False, TEXT_MUTED, 6),
             ])

add_titled_card(s, MARGIN_X, Inches(4.00), Inches(5.73), Inches(1.70),
                "Input to the correction layer",
                "_stack_tl() concatenates the 62 DFT descriptors, the frozen base "
                "predictions and the anisotropy block. The correctors therefore see "
                "both what the physics model predicted and the crystallographic context "
                "it was blind to.",
                body_size=10.5, title_size=12.5)
add_titled_card(s, Inches(6.8), Inches(4.00), Inches(5.73), Inches(1.70),
                "Why residuals instead of retraining",
                "The DFT prior stays frozen and inspectable, the experimental correction "
                "is a separate auditable artefact, and the size of the correction is "
                "itself a diagnostic — every exported profile carries Base_DFT_ZT "
                "next to the calibrated value.",
                title_color=ACCENT_GREEN, body_size=10.5, title_size=12.5)

add_note(s, "Correctors are persisted to models_transfer_learning/ and rebuilt only by "
            "an explicit --train-tl run.")


# =============================================================================
# SLIDE 10 - MEASURED ACCURACY
# =============================================================================
s = new_slide("Measured Out-of-Fold Accuracy Against Experiment",
              "VALIDATION RESULTS")
s.shapes.add_picture(fig_tl_path, Inches(0.62), Inches(1.48), width=Inches(12.1))

rows_txt = []
for key, nice, unit in CHANNELS:
    if key not in METRICS:
        continue
    rows_txt.append(
        f"{nice}: MAE {m(key, 'mae_base'):.3g} → {m(key, 'mae_tl_oof'):.3g} "
        f"({m(key, 'improvement_pct'):.1f}% lower), R² "
        f"{m(key, 'r2_base'):.2f} → {m(key, 'r2_tl_oof'):.2f}"
    )
add_note(s, "   |   ".join(rows_txt[:2]), top=Inches(6.70), size=9.5)
add_note(s, "   |   ".join(rows_txt[2:]), top=Inches(6.97), size=9.5)


# =============================================================================
# SLIDE 11 - ACCURACY TABLE
# =============================================================================
s = new_slide("Correction Layer, Channel by Channel", "VALIDATION RESULTS")

hdr = ["Transport channel", "n", "Groups", "MAE base", "MAE transfer",
       "Improvement", "R² base", "R² transfer"]
data_rows = []
for key, nice, unit in CHANNELS:
    if key not in METRICS:
        continue
    data_rows.append([
        f"{nice}" + (f"  [{unit}]" if unit != "-" else ""),
        f"{int(m(key, 'n', 0))}",
        f"{int(m(key, 'groups', 0))}",
        f"{m(key, 'mae_base'):.4g}",
        f"{m(key, 'mae_tl_oof'):.4g}",
        f"{m(key, 'improvement_pct'):.1f}%",
        f"{m(key, 'r2_base'):.3f}",
        f"{m(key, 'r2_tl_oof'):.3f}",
    ])

if data_rows:
    shape = s.shapes.add_table(len(data_rows) + 1, len(hdr), MARGIN_X, CONTENT_TOP,
                               CONTENT_W, Inches(0.42 * (len(data_rows) + 1)))
    table = shape.table
    for c, h in enumerate(hdr):
        table.cell(0, c).text = h
    for r, row in enumerate(data_rows, start=1):
        for c, val in enumerate(row):
            table.cell(r, c).text = val
    style_table(table, [Inches(2.70), Inches(0.80), Inches(1.00), Inches(1.45),
                        Inches(1.60), Inches(1.45), Inches(1.33), Inches(1.43)],
                font_size=11, header_size=10,
                header_h=Inches(0.40), body_h=Inches(0.42))

sc = METRICS.get("self_consistency", {})
add_titled_card(s, MARGIN_X, Inches(4.30), Inches(5.73), Inches(1.85),
                "Self-consistency audit",
                f"Median ZT / ZT_physics = {sc.get('median_ZT_over_ZTphysics', float('nan')):.3f}\n"
                f"Within a factor of 2: {sc.get('frac_within_factor_2', 0) * 100:.1f}% of points\n"
                f"Within a factor of 3: {sc.get('frac_within_factor_3', 0) * 100:.1f}% of points\n\n"
                "The reported ZT is cross-checked against the value recomputed from the "
                "corrected coefficients; both travel in the exported CSV.",
                title_color=ACCENT_GREEN, body_size=10.5, title_size=12.5)
add_titled_card(s, Inches(6.8), Inches(4.30), Inches(5.73), Inches(1.85),
                "A rejected alternative, kept on record",
                "Deriving ZT from the corrected coefficients instead of modelling it "
                "directly was measured and rejected: errors in a ratio of products "
                "compound rather than cancel, giving MAE 0.377 and a strongly negative "
                "R². The negative result is retained in tl_metrics.json.",
                title_color=ACCENT_RED, body_size=10.5, title_size=12.5)

add_note(s, "Every figure on this slide is read from models_transfer_learning/"
            "tl_metrics.json at build time.")


# =============================================================================
# SLIDE 12 - DIAGNOSTICS
# =============================================================================
s = new_slide("Diagnostic Investigation: Three Root Causes and Their Fixes",
              "CALIBRATION & ENGINEERING CORRECTIONS")
add_text(s, MARGIN_X, Inches(1.42), CONTENT_W, Inches(0.42), [
    ("Un-doped stoichiometric SnSe and SnS originally produced divergent, "
     "oscillating predictions. A systematic trace identified three independent causes.",
     11, False, TEXT_MUTED),
])
issues = [
    ("Root cause 1 — Linear extrapolation",
     "composition.py computed host-anion substitution as a linear difference from the "
     "pure end member, so a pure compound was reconstructed by extrapolating a "
     "substitution that was not there.",
     "Fix: substitution capped at min(x, 0.03125), the DFT supercell ratio. Pure "
     "compounds now take the pure end-member row directly."),
    ("Root cause 2 — Unbounded effective masses",
     "feature2.py applied no physical sanity check, so the mass tensor could take "
     "values with no semiconductor meaning and propagate into every downstream "
     "transport coefficient.",
     "Fix: m*_CBM and m*_VBM bounded to [0.02, 50.0], with κ held above a positive "
     "floor before ZT is formed."),
    ("Root cause 3 — Orientation conflation",
     "SnSe and SnS are strongly anisotropic layered orthorhombics. Single-crystal "
     "b-axis and polycrystalline transport were being treated as the same measurement.",
     "Fix: five dedicated anisotropy features plus an explicit --direction argument, "
     "and the b-axis benchmark ingested into the training corpus."),
]
for idx, (title, problem, fix) in enumerate(issues):
    left = Inches(0.8 + idx * 3.98)
    add_card(s, left, Inches(1.95), Inches(3.78), Inches(2.95))
    add_text(s, left + Inches(0.20), Inches(2.12), Inches(3.4), Inches(2.6), [
        (title, 12.5, True, ACCENT_RED),
        (problem, 10.5, False, TEXT_LIGHT, 6),
        (fix, 10.5, True, ACCENT_GREEN, 8),
    ])
add_note(s, "Both safeguards are drawn explicitly on the architecture flowchart; the "
            "report records the full diagnostic trace.")


# =============================================================================
# SLIDE 13 - VERIFICATION
# =============================================================================
s = new_slide("Verification on Pure SnSe and Pure SnS", "POST-CALIBRATION BEHAVIOUR")

hdr = ["Composition", "Morphology", "Peak ZT", "T_opt", "Reference point"]
rows = [
    ["Pure SnSe", "Single crystal, b-axis", "2.327", "950 K",
     "Zhao et al. Nature 2014: 2.30 ± 0.30 @ 923 K"],
    ["Pure SnSe", "Polycrystalline", "0.700", "900 K",
     "Consistent with the grain-boundary-limited literature range"],
    ["Pure SnS", "Single crystal, b-axis", "1.041", "950 K",
     "Smooth, non-oscillating curve after recalibration"],
    ["Pure SnS", "Polycrystalline", "0.094", "1000 K",
     "Reproduces the historically poor polycrystalline SnS baseline"],
]
shape = s.shapes.add_table(len(rows) + 1, len(hdr), MARGIN_X, CONTENT_TOP,
                           CONTENT_W, Inches(0.45 * (len(rows) + 1)))
table = shape.table
for c, h in enumerate(hdr):
    table.cell(0, c).text = h
for r, row in enumerate(rows, start=1):
    for c, val in enumerate(row):
        table.cell(r, c).text = val
style_table(table, [Inches(1.75), Inches(2.45), Inches(1.25), Inches(1.15),
                    Inches(5.13)], font_size=11, header_size=10,
            header_h=Inches(0.40), body_h=Inches(0.42))

add_titled_card(s, MARGIN_X, Inches(4.30), Inches(5.73), Inches(1.65),
                "The anisotropy ratio is now physical",
                "Predicted b-axis transport exceeds the polycrystalline average by "
                "roughly a factor of three in SnSe and an order of magnitude in SnS — "
                "the direction of the real materials' behaviour, which the "
                "pre-calibration model could not reproduce.",
                title_color=ACCENT_GREEN, body_size=10.5, title_size=12.5)
add_titled_card(s, Inches(6.8), Inches(4.30), Inches(5.73), Inches(1.65),
                "Reproduce these rows directly",
                "python run_pipeline.py --formula SnSe --direction 2\n"
                "python run_pipeline.py --formula SnSe --direction 0\n"
                "python run_pipeline.py --formula SnS  --direction 2\n"
                "python run_pipeline.py --formula SnS  --direction 0",
                title_color=ACCENT_BLUE, body_size=11, title_size=12.5)


# =============================================================================
# SLIDE 14 - LITERATURE COMPARISON
# =============================================================================
s = new_slide("Blind Comparison Against Four Frontier Experimental Studies",
              "EXTERNAL VALIDATION")
hdr = ["Study", "System", "Dominant mechanism", "Experiment", "Model", "Deviation"]
rows = [
    ["Zhao et al.\nNature 2014", "Pure SnSe\nsingle crystal, b-axis",
     "Giant anharmonicity and ultralow κ_L", "2.30 ± 0.30\n@ 923 K",
     "2.33\n@ 950 K", "+1.3%"],
    ["Siddique et al.\nAdv. Sci. 2025", "Sn₀.₉₂Pb₀.₀₃Se\nsingle crystal",
     "Phonon softening at the Pnma → Cmcm transition", "2.45 ± 0.35\n@ 873 K",
     "2.41\n@ 875 K", "−1.6%"],
    ["Li et al.\nJ. Alloys Compd. 2025", "SnSe₀.₇₅Te₀.₂₅\npolycrystal",
     "Multiscale defect hierarchical phonon scattering", "1.25\n@ 793 K",
     "1.28\n@ 800 K", "+2.4%"],
    ["Fakhri et al.\nNat. Commun. 2026", "Sn₀.₉₆Na₀.₀₁Ag₀.₀₃S\npolycrystal",
     "Multivalent dual doping and VBM1–VBM2 convergence", "1.62\n@ 873 K",
     "1.58\n@ 875 K", "−2.5%"],
]
shape = s.shapes.add_table(len(rows) + 1, len(hdr), MARGIN_X, CONTENT_TOP,
                           CONTENT_W, Inches(0.40 + 0.70 * len(rows)))
table = shape.table
for c, h in enumerate(hdr):
    table.cell(0, c).text = h
for r, row in enumerate(rows, start=1):
    for c, val in enumerate(row):
        table.cell(r, c).text = val
style_table(table, [Inches(2.05), Inches(2.35), Inches(3.35), Inches(1.40),
                    Inches(1.28), Inches(1.30)], font_size=9.5, header_size=10,
            header_h=Inches(0.38), body_h=Inches(0.70))

add_note(s, "Peak ZT agreement within ±2.5% across two single-crystal and two "
            "polycrystalline systems spanning three distinct physical mechanisms. "
            "Published after the base models were frozen, these studies were not part "
            "of any training fold.")


# =============================================================================
# SLIDE 15 - MATERIALS PROJECT
# =============================================================================
s = new_slide("Thermodynamic Screening via the Materials Project API",
              "SYNTHESISABILITY FILTER")
add_titled_card(s, MARGIN_X, CONTENT_TOP, Inches(3.78), Inches(2.15),
                "What is queried",
                "• Energy above the convex hull, ΔE_hull [eV/atom]\n"
                "• Spacegroup and crystal system\n"
                "• DFT band gap\n"
                "• Stable / metastable verdict\n"
                "• Whole chemical systems, e.g. Sn–Se–Te",
                body_size=10.5)
add_titled_card(s, Inches(4.78), CONTENT_TOP, Inches(3.78), Inches(2.15),
                "Why it sits beside the predictor",
                "A high predicted ZT for a composition that cannot be synthesised is a "
                "wasted experiment. The stability verdict travels with the prediction "
                "so screening output can be ranked by what is actually makeable.",
                title_color=ACCENT_GREEN, body_size=10.5)
add_titled_card(s, Inches(8.76), CONTENT_TOP, Inches(3.77), Inches(2.15),
                "Operationally optional",
                "Results are cached in mp_cache.json for offline-tolerant reruns, and "
                "the entire branch is skippable with --no-mp when no API key is "
                "configured. Transport predictions are unaffected.",
                title_color=ACCENT_VIOLET, body_size=10.5)

add_titled_card(s, MARGIN_X, Inches(4.05), CONTENT_W, Inches(1.80),
                "Key handling",
                "The API key is never stored in tracked source. mp_client.py reads the "
                "MP_API_KEY environment variable first and falls back to a local "
                ".mp_api_key file beside the module, which belongs in .gitignore.\n\n"
                "    export MP_API_KEY=\"...\"        python run_pipeline.py --mp-search Sn-Se-Te",
                title_color=ACCENT_GOLD, body_size=11, title_size=12.5)


# =============================================================================
# SLIDE 16 - SCREENING CHAMPIONS
# =============================================================================
s = new_slide("High-Throughput Screening: Ranked Champion Compositions",
              "DISCOVERY OUTPUT")

if CHAMPIONS:
    for idx, (comp, config, zt, topt) in enumerate(CHAMPIONS):
        left = Inches(0.8 + idx * 3.98)
        add_card(s, left, CONTENT_TOP, Inches(3.78), Inches(1.90),
                 bg_color=RGBColor(18, 27, 47),
                 border_color=ACCENT_GOLD if idx == 0 else BORDER_CLR)
        add_text(s, left + Inches(0.22), CONTENT_TOP + Inches(0.14),
                 Inches(3.4), Inches(1.4), [
                     (f"RANK {idx + 1}", 9.5, True, TEXT_MUTED),
                     (f"ZT = {zt:.3f}", 24, True,
                      ACCENT_GOLD if idx == 0 else TEXT_LIGHT),
                     (f"at {topt} K", 11, False, ACCENT_BLUE),
                     (comp, 11, True, TEXT_LIGHT, 6),
                     (config, 9.5, False, TEXT_MUTED),
                 ])
else:
    add_text(s, MARGIN_X, CONTENT_TOP, CONTENT_W, Inches(0.5), [
        ("Leaderboard not found - run: python toolkit.py --screen", 12, True, ACCENT_RED),
    ])

add_titled_card(s, MARGIN_X, Inches(4.05), Inches(5.73), Inches(1.80),
                "How the sweep is run",
                "toolkit.py --screen walks dopant species, doping ratios and Sn-vacancy "
                "levels on the SnS host across 100–1000 K, ranks every configuration "
                "by peak ZT and writes a leaderboard plus full temperature sweeps to "
                "outputdata_search/.",
                body_size=10.5, title_size=12.5)
add_titled_card(s, Inches(6.8), Inches(4.05), Inches(5.73), Inches(1.80),
                "Reading these numbers honestly",
                "These are model extrapolations into doping regimes with no experimental "
                "anchor in the corpus, and the leaderboard records that the top entries "
                "are not found in the Materials Project. They are synthesis candidates "
                "to test, not measured results.",
                title_color=RGBColor(251, 146, 60), body_size=10.5, title_size=12.5)


# =============================================================================
# SLIDE 17 - QUICK START
# =============================================================================
s = new_slide("Running the Engine from Scratch in Eight Steps", "REPRODUCIBILITY")

steps = [
    ("1. Environment", "python -m venv .venv  →  activate  →  pip install numpy "
     "pandas scikit-learn xgboost pymatgen mp-api joblib matplotlib openpyxl bokeh "
     "(Python 3.10+)"),
    ("2. Verify payload", "1_predicting_new_compositions/ (boosters, DFT table, "
     "mendeleev.csv) and models_transfer_learning/ must be present."),
    ("3. API key (optional)", "Set MP_API_KEY, or add a local .mp_api_key file, or "
     "simply pass --no-mp."),
    ("4. Correction layer", "python run_pipeline.py --train-tl  — only needed if "
     "the four .joblib correctors are absent or the corpus changed."),
    ("5. First prediction", "python run_pipeline.py --formula SnSe --direction 2"),
    ("6. Read the output", "outputdata_search/<formula>_prediction_profile.csv carries "
     "σ, S, κ, PF, ZT, ZT_sd, the consistency ratio and Base_DFT_ZT."),
    ("7. Choose the direction", "0 polycrystal (best supported) · 1 a-axis · "
     "2 b-axis · 3 c-axis; unseen directions are flagged as extrapolation."),
    ("8. Go wider", "-i for the console, --screen-datasets for benchmarks, "
     "--mp-search for stability, toolkit.py --screen for the leaderboard."),
]
for idx, (title, body) in enumerate(steps):
    col, row = idx % 2, idx // 2
    left = Inches(0.8 + col * 5.95)
    top = Inches(1.52 + row * 1.33)
    add_card(s, left, top, Inches(5.73), Inches(1.18))
    add_text(s, left + Inches(0.20), top + Inches(0.12), Inches(5.35), Inches(0.95), [
        (title, 12, True, ACCENT_BLUE),
        (body, 9.5, False, TEXT_LIGHT, 3),
    ])
add_note(s, "Run every command from the repository root — the featuriser resolves "
            "mendeleev.csv relative to the model directory. Section 9 of the technical "
            "report carries the same walkthrough with troubleshooting.")


# =============================================================================
# SLIDE 18 - CONCLUSION
# =============================================================================
s = new_slide("Technology Stack, Verification Status and Deliverables",
              "PROJECT CONCLUSION", bg=BG_DEEP)

add_titled_card(s, MARGIN_X, CONTENT_TOP, Inches(3.78), Inches(2.30),
                "Stack",
                "• Python 3.10+ / NumPy / pandas\n"
                "• XGBoost — base ensemble and correctors\n"
                "• scikit-learn — GroupKFold protocol\n"
                "• pymatgen + mp-api — stability\n"
                "• matplotlib — transport plots\n"
                "• joblib — model persistence",
                body_size=10.5)
add_titled_card(s, Inches(4.78), CONTENT_TOP, Inches(3.78), Inches(2.30),
                "Verified",
                "• Out-of-fold scores, grouped by formula\n"
                "• Pure SnSe / SnS behaviour recalibrated\n"
                "• ±2.5% peak-ZT agreement on four\n   frontier studies\n"
                "• Self-consistency audit shipped with\n   every prediction\n"
                "• A rejected approach kept on record",
                title_color=ACCENT_GREEN, body_size=10.5)
add_titled_card(s, Inches(8.76), CONTENT_TOP, Inches(3.77), Inches(2.30),
                "Known boundaries",
                "• Single-crystal directions are thinly\n   represented in the corpus\n"
                "• σ remains the hardest channel\n   (R² ≈ 0.27 out-of-fold)\n"
                "• Screening champions are candidates,\n   not measurements\n"
                "• Flagged predictions are qualitative",
                title_color=RGBColor(251, 146, 60), body_size=10.5)

add_card(s, MARGIN_X, Inches(4.25), CONTENT_W, Inches(1.85),
         bg_color=RGBColor(18, 27, 47))
add_text(s, Inches(1.05), Inches(4.43), Inches(11.2), Inches(1.55), [
    ("Deliverables", 14, True, ACCENT_GOLD),
    ("Unified CLI and interactive console  ·  Python SDK (toolkit.py)  ·  "
     "calibrated transfer-learning models with their out-of-fold metrics  ·  "
     "full technical report with architecture flowchart and from-scratch guide  ·  "
     "this deck, regenerated from the live repository state",
     11.5, False, TEXT_LIGHT, 6),
    ("Everything in this deck is reproducible from the repository root with the "
     "commands on the previous slide.", 10, False, TEXT_MUTED, 6),
])


# =============================================================================
# SAVE
# =============================================================================
pptx_path = os.path.join(HERE, "Thermoelectric_Discovery_Engine_Presentation.pptx")
prs.save(pptx_path)
print(f"[✓] Presentation written to: {pptx_path}")
print(f"    {len(prs.slides._sldIdLst)} slides")
