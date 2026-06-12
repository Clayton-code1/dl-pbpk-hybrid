"""
Generate Figures 3, 4, 5 for the manuscript.
Run from project root: python experiments/manuscript_figures/generate_figures.py

Reads ONLY from existing result files — no data synthesis.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams
from sklearn.metrics import r2_score

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]   # c:\Users\Admin\dl-pbpk-hybrid
OUT  = Path(__file__).resolve().parent        # experiments/manuscript_figures/

THEOPH_CSV  = ROOT / "experiments/real_theoph/real_theoph_predictions.csv"
WARF_CSV    = ROOT / "experiments/warfarin_validation/warfarin_predictions.csv"
SHAP_MD     = ROOT / "experiments/results/phase3_shap_interpretation.md"

# ── global style ─────────────────────────────────────────────────────────────
rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "DejaVu Sans"],
    "font.size":         11,
    "axes.linewidth":    0.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "figure.dpi":        150,
    "savefig.dpi":       300,
})

DPI = 300


def check_file(p: Path) -> None:
    if not p.exists():
        print(f"MISSING FILE: {p}")
        print("STOPPING — cannot generate figure without real data.")
        sys.exit(1)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Theophylline observed vs predicted
# ═══════════════════════════════════════════════════════════════════════════
def figure3() -> None:
    check_file(THEOPH_CSV)
    df = pd.read_csv(THEOPH_CSV)

    required = {"subject_id", "conc_obs_mgL", "conc_pred_mgL"}
    if not required.issubset(df.columns):
        print(f"ERROR: expected columns {required}, got {list(df.columns)}")
        sys.exit(1)

    n_rows = len(df)
    n_subj = df["subject_id"].nunique()
    y_obs  = df["conc_obs_mgL"].values
    y_pred = df["conc_pred_mgL"].values

    r2   = r2_score(y_obs, y_pred)
    rmse_val = rmse(y_obs, y_pred)

    print(f"\nFigure 3 — Theophylline")
    print(f"  Source  : {THEOPH_CSV}")
    print(f"  Points  : {n_rows} observations, {n_subj} subjects")
    print(f"  R²      : {r2:.4f}   (manuscript: 0.6725)")
    print(f"  RMSE    : {rmse_val:.4f} mg/L  (manuscript: 1.6346 mg/L)")

    # warn if material discrepancy
    if abs(r2 - 0.6725) > 0.01:
        print(f"  *** WARNING: R² discrepancy vs manuscript (|{r2:.4f} - 0.6725| > 0.01)")
    if abs(rmse_val - 1.6346) > 0.05:
        print(f"  *** WARNING: RMSE discrepancy vs manuscript (|{rmse_val:.4f} - 1.6346| > 0.05)")

    # ── colour subjects ───────────────────────────────────────────────────
    subjects = sorted(df["subject_id"].unique())
    cmap = plt.get_cmap("tab20", len(subjects))
    subj_colour = {s: cmap(i) for i, s in enumerate(subjects)}
    colours = [subj_colour[s] for s in df["subject_id"]]

    # ── plot ─────────────────────────────────────────────────────────────
    lo = min(y_obs.min(), y_pred.min()) - 0.2
    hi = max(y_obs.max(), y_pred.max()) + 0.5

    fig, ax = plt.subplots(figsize=(4.8, 4.8))

    ax.scatter(y_obs, y_pred, c=colours, s=28, alpha=0.85,
               linewidths=0.3, edgecolors="white", zorder=3)

    # identity line
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1.0, color="#444444",
            label="Identity (y = x)", zorder=2)

    # annotation box
    ann = f"$R^2$ = {r2:.3f}\nRMSE = {rmse_val:.3f} mg/L\n$n$ = {n_rows} obs, {n_subj} subj"
    ax.text(0.04, 0.97, ann, transform=ax.transAxes,
            va="top", ha="left", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9))

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Observed concentration (mg/L)", fontsize=11)
    ax.set_ylabel("Predicted concentration (mg/L)", fontsize=11)
    ax.set_title("Theophylline — R Theoph dataset\n(real human data, forward-only inference)",
                 fontsize=10.5, pad=6)
    ax.legend(fontsize=9, frameon=False, loc="lower right")

    fig.tight_layout()
    out_path = OUT / "Figure3_theophylline_obs_vs_pred.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved   : {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Warfarin absorption subgroup observed vs predicted
# ═══════════════════════════════════════════════════════════════════════════
def figure4() -> None:
    check_file(WARF_CSV)
    df_all = pd.read_csv(WARF_CSV)

    required = {"subject_id", "conc_obs_mgL", "conc_pred_mgL", "abs_group"}
    if not required.issubset(df_all.columns):
        print(f"ERROR: expected columns {required}, got {list(df_all.columns)}")
        sys.exit(1)

    # filter to absorption-present subgroup only
    df = df_all[df_all["abs_group"] == True].copy()

    n_rows = len(df)
    n_subj = df["subject_id"].nunique()
    y_obs  = df["conc_obs_mgL"].values
    y_pred = df["conc_pred_mgL"].values

    r2       = r2_score(y_obs, y_pred)
    rmse_val = rmse(y_obs, y_pred)

    print(f"\nFigure 4 — Warfarin (absorption-present subgroup)")
    print(f"  Source  : {WARF_CSV}  [abs_group == True]")
    print(f"  Points  : {n_rows} observations, {n_subj} subjects")
    print(f"  R²      : {r2:.4f}   (manuscript: 0.6681)")
    print(f"  RMSE    : {rmse_val:.4f} mg/L  (manuscript: 2.6966 mg/L)")

    if abs(r2 - 0.6681) > 0.01:
        print(f"  *** WARNING: R² discrepancy vs manuscript (|{r2:.4f} - 0.6681| > 0.01)")
    if abs(rmse_val - 2.6966) > 0.05:
        print(f"  *** WARNING: RMSE discrepancy vs manuscript (|{rmse_val:.4f} - 2.6966| > 0.05)")

    # ── colour subjects ───────────────────────────────────────────────────
    subjects = sorted(df["subject_id"].unique())
    cmap = plt.get_cmap("tab20", len(subjects))
    subj_colour = {s: cmap(i) for i, s in enumerate(subjects)}
    colours = [subj_colour[s] for s in df["subject_id"]]

    lo = min(y_obs.min(), y_pred.min()) - 0.3
    hi = max(y_obs.max(), y_pred.max()) + 0.5

    fig, ax = plt.subplots(figsize=(4.8, 4.8))

    ax.scatter(y_obs, y_pred, c=colours, s=28, alpha=0.85,
               linewidths=0.3, edgecolors="white", zorder=3)

    ax.plot([lo, hi], [lo, hi], ls="--", lw=1.0, color="#444444",
            label="Identity (y = x)", zorder=2)

    ann = (f"$R^2$ = {r2:.3f}\nRMSE = {rmse_val:.3f} mg/L\n"
           f"$n$ = {n_rows} obs, {n_subj} subj\n"
           f"(absorption-present group)")
    ax.text(0.04, 0.97, ann, transform=ax.transAxes,
            va="top", ha="left", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9))

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Observed concentration (mg/L)", fontsize=11)
    ax.set_ylabel("Predicted concentration (mg/L)", fontsize=11)
    ax.set_title(
        "Warfarin — nlmixr2data dataset\n"
        "Absorption-present subgroup ($n$ = 13 subjects)",
        fontsize=10.5, pad=6)
    ax.legend(fontsize=9, frameon=False, loc="lower right")

    fig.tight_layout()
    out_path = OUT / "Figure4_warfarin_obs_vs_pred.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved   : {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5 — KernelSHAP patient-feature attribution (all 6 drugs)
# ═══════════════════════════════════════════════════════════════════════════
def parse_shap_md(path: Path) -> dict[str, dict[str, float]]:
    """
    Parse the per-drug KernelSHAP tables from phase3_shap_interpretation.md.
    Returns {drug: {feature: mean_abs_shap}}.
    """
    text = path.read_text(encoding="utf-8")
    # Split by drug header: "### <drugname>"
    drug_blocks = re.split(r"###\s+", text)

    results: dict[str, dict[str, float]] = {}
    table_row = re.compile(r"^\|\s*\d+\s*\|\s*(\S+)\s*\|\s*([\d.]+)\s*\|", re.MULTILINE)

    for block in drug_blocks[1:]:          # skip preamble
        lines = block.strip().splitlines()
        drug = lines[0].strip().lower()    # e.g. "theophylline"
        features: dict[str, float] = {}
        for m in table_row.finditer(block):
            feat, val = m.group(1).strip(), float(m.group(2))
            features[feat] = val
        if features:
            results[drug] = features
    return results


# canonical display names for features
FEAT_LABEL = {
    "dose_mgkg":           "Dose/weight\n(mg/kg)",
    "dose_mg_per_kg":      "Dose/weight\n(mg/kg)",
    "log_dose_mg_per_kg":  "log(Dose/weight)\n(mg/kg)",
    "weight_kg":           "Body weight\n(kg)",
    "sex":                 "Sex",
    "age_years":           "Age\n(years)",
    "dose_mg":             "Dose\n(mg)",
}
FEAT_ORDER = ["dose_mgkg", "dose_mg_per_kg", "log_dose_mg_per_kg",
              "weight_kg", "sex", "age_years", "dose_mg"]

DRUG_LABEL = {
    "theophylline": "Theophylline",
    "warfarin":     "Warfarin",
    "midazolam":    "Midazolam",
    "caffeine":     "Caffeine",
    "acetaminophen":"Acetaminophen",
    "digoxin":      "Digoxin",
}
DRUG_ORDER = ["theophylline", "warfarin", "midazolam", "caffeine",
              "acetaminophen", "digoxin"]


def figure5() -> None:
    check_file(SHAP_MD)
    shap_data = parse_shap_md(SHAP_MD)

    if not shap_data:
        print("ERROR: could not parse SHAP tables from", SHAP_MD)
        sys.exit(1)

    print(f"\nFigure 5 — KernelSHAP patient-feature attribution")
    print(f"  Source  : {SHAP_MD}")
    print(f"  Drugs   : {list(shap_data.keys())}")
    for drug in DRUG_ORDER:
        if drug in shap_data:
            print(f"  {drug:<15}: {shap_data[drug]}")

    # ── build canonical feature set and matrix ────────────────────────────
    # Collect all feature keys across drugs, mapped to canonical display names
    # Group dose-related features under one bucket where possible
    canonical_map = {
        "dose_mgkg":           "Dose/wt (mg/kg)",
        "dose_mg_per_kg":      "Dose/wt (mg/kg)",
        "log_dose_mg_per_kg":  "log(Dose/wt)",
        "weight_kg":           "Weight (kg)",
        "sex":                 "Sex",
        "age_years":           "Age (yr)",
        "dose_mg":             "Dose (mg)",
    }
    display_order = ["Dose/wt (mg/kg)", "log(Dose/wt)", "Weight (kg)", "Sex", "Age (yr)", "Dose (mg)"]

    # Build per-drug rows for the display features
    records = []
    for drug in DRUG_ORDER:
        if drug not in shap_data:
            continue
        row = {v: 0.0 for v in display_order}
        for raw_feat, val in shap_data[drug].items():
            canon = canonical_map.get(raw_feat)
            if canon and canon in row:
                row[canon] = max(row[canon], val)  # take max if multiple map to same
        records.append((DRUG_LABEL.get(drug, drug), row))

    # ── layout: grouped bar chart (features on y-axis, bars per drug) ─────
    n_drugs = len(records)
    n_feat  = len(display_order)

    # Separate theophylline (large scale) from the others (small scale)
    # to avoid theophylline's large values squishing the rest.
    # Layout: two panels side by side with independent x-scales.

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(10, 4.2),
        gridspec_kw={"wspace": 0.55}
    )

    colours_drug = plt.get_cmap("tab10")

    def draw_drug_bars(ax, drug_name, row_dict, colour, x_label=True):
        feats  = display_order
        values = [row_dict.get(f, 0.0) for f in feats]
        y_pos  = np.arange(len(feats))
        bars = ax.barh(y_pos, values, height=0.55, color=colour, alpha=0.85,
                       edgecolor="white", linewidth=0.4)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feats, fontsize=10)
        ax.invert_yaxis()
        ax.axvline(0, color="black", lw=0.6)
        ax.set_title(drug_name, fontsize=11, pad=4)
        if x_label:
            ax.set_xlabel("Mean |SHAP| on predicted AUC", fontsize=9.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=9)
        # value labels on bars
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_width() + ax.get_xlim()[1] * 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va="center", ha="left", fontsize=8.5)

    # Left panel: theophylline alone (large scale values)
    theo_drug, theo_row = records[0]   # theophylline is first
    draw_drug_bars(ax_left, theo_drug, theo_row,
                   colour=colours_drug(0), x_label=True)

    # Right panel: remaining 5 drugs, stacked as a grouped bar chart
    # Use small multiples — too cramped for one axes; use a matrix heatmap instead
    ax_right.remove()
    # Replace with a proper heatmap for the 5 remaining drugs
    fig.tight_layout()

    # Redo layout as 1x3: theophylline bar | other-5 heatmap
    fig.clf()

    fig = plt.figure(figsize=(11.5, 4.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.8], wspace=0.45)
    ax_bar  = fig.add_subplot(gs[0])
    ax_heat = fig.add_subplot(gs[1])

    # --- Theophylline bar chart ---
    theo_drug, theo_row = records[0]
    feats  = display_order
    values = [theo_row.get(f, 0.0) for f in feats]
    y_pos  = np.arange(len(feats))
    bar_colour = "#2166ac"
    bars = ax_bar.barh(y_pos, values, height=0.55, color=bar_colour,
                       alpha=0.88, edgecolor="white", linewidth=0.3)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(feats, fontsize=10)
    ax_bar.invert_yaxis()
    ax_bar.axvline(0, color="black", lw=0.5)
    ax_bar.set_title(f"Theophylline\n(KernelSHAP on AUC)", fontsize=10.5, pad=5)
    ax_bar.set_xlabel("Mean |SHAP| on predicted AUC", fontsize=9.5)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.tick_params(axis="x", labelsize=9)
    # value labels
    xlim_right = max(values) * 1.18
    ax_bar.set_xlim(0, xlim_right)
    for bar, val in zip(bars, values):
        if val > 0:
            ax_bar.text(bar.get_width() + xlim_right * 0.015,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va="center", ha="left", fontsize=8.5)

    # --- All-drugs heatmap ---
    # Normalise within each drug (so we compare relative importance, not absolute)
    drug_names_all = [r[0] for r in records]
    matrix = np.zeros((len(records), len(feats)))
    for i, (_, row_dict) in enumerate(records):
        vals = np.array([row_dict.get(f, 0.0) for f in feats])
        total = vals.sum()
        matrix[i] = vals / total if total > 0 else vals

    # mask the zero column (Dose (mg) is always 0)
    nonzero_cols = [j for j in range(len(feats)) if matrix[:, j].sum() > 0]
    matrix_nz = matrix[:, nonzero_cols]
    feats_nz   = [feats[j] for j in nonzero_cols]

    im = ax_heat.imshow(matrix_nz, cmap="YlOrRd", aspect="auto",
                        vmin=0, vmax=1)
    ax_heat.set_xticks(np.arange(len(feats_nz)))
    ax_heat.set_xticklabels(feats_nz, fontsize=9.5, rotation=30, ha="right")
    ax_heat.set_yticks(np.arange(len(drug_names_all)))
    ax_heat.set_yticklabels(drug_names_all, fontsize=10)
    ax_heat.set_title("Relative feature importance across drugs\n(normalised within each drug)",
                      fontsize=10.5, pad=5)

    # cell text
    for i in range(matrix_nz.shape[0]):
        for j in range(matrix_nz.shape[1]):
            val = matrix_nz[i, j]
            txt_colour = "white" if val > 0.6 else "black"
            ax_heat.text(j, i, f"{val:.2f}", ha="center", va="center",
                         fontsize=7.5, color=txt_colour)

    cbar = fig.colorbar(im, ax=ax_heat, shrink=0.82, pad=0.02)
    cbar.set_label("Normalised mean |SHAP|", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        "Patient-feature attribution (KernelSHAP on predicted AUC)",
        fontsize=11.5, y=1.01
    )
    fig.tight_layout()

    out_path = OUT / "Figure5_shap_attribution.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved   : {out_path}")

    # confirm dose/weight dominate vs age/sex
    theo_vals = records[0][1]
    dose_wt = theo_vals.get("Dose/wt (mg/kg)", 0.0)
    weight  = theo_vals.get("Weight (kg)", 0.0)
    age     = theo_vals.get("Age (yr)", 0.0)
    sex_val = theo_vals.get("Sex", 0.0)
    print(f"  Theophylline SHAP check:")
    print(f"    Dose/wt = {dose_wt:.4f}  (expected ~1.608)")
    print(f"    Weight  = {weight:.4f}  (expected ~1.060)")
    print(f"    Sex     = {sex_val:.4f}  (expected ~0.492)")
    print(f"    Age     = {age:.4f}    (expected ~0.131)")
    if dose_wt > weight > age:
        print(f"    OK: Dose/wt > Weight > Age -- consistent with manuscript")
    else:
        print(f"    *** ORDER MISMATCH -- check against manuscript")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Branch      : experiment/warfarin-validation  (confirmed before running)")
    print(f"Output dir  : {OUT}")
    print("=" * 60)
    figure3()
    figure4()
    figure5()
    print("\n" + "=" * 60)
    print("All figures saved.")
