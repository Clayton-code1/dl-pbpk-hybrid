"""
Level 3 Signal Check — Go/No-Go for GNN-based therapeutic window prediction.
Pure analysis script. No model files are written; no project code is imported.
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "experiments/data/therapeutic_windows/therapeutic_window_dataset_filtered.csv"
OUT_DIR   = Path(__file__).parent
REPORT    = OUT_DIR / "signal_check_report.md"

# ── Step 0 — load & filter ────────────────────────────────────────────────────
print("=== Level 3 Signal Check ===\n")

raw = pd.read_csv(DATA_PATH)
df  = raw[raw["likely_therapeutic_agent"] == True].copy()
print(f"Rows after likely_therapeutic_agent filter: {len(df)}")

# ── Step 1 — target: log10(midpoint) ─────────────────────────────────────────
df["midpoint"] = (df["therapeutic_min_mg_L"] + df["therapeutic_max_mg_L"]) / 2.0

# Guard against zero/negative midpoints (should not happen given the data, but be safe)
bad_mid = df["midpoint"] <= 0
if bad_mid.sum():
    print(f"  Dropping {bad_mid.sum()} rows with midpoint <= 0")
    df = df[~bad_mid]

df["log10_midpoint"] = np.log10(df["midpoint"])

tgt = df["log10_midpoint"]
print(f"\nStep 1 — Target distribution (log10 midpoint mg/L):")
print(f"  n     = {len(tgt)}")
print(f"  min   = {tgt.min():.3f}")
print(f"  max   = {tgt.max():.3f}")
print(f"  mean  = {tgt.mean():.3f}")
print(f"  std   = {tgt.std():.3f}")
print(f"  p25   = {tgt.quantile(0.25):.3f}")
print(f"  p50   = {tgt.quantile(0.50):.3f}")
print(f"  p75   = {tgt.quantile(0.75):.3f}")

# Build a text histogram
bins = np.arange(np.floor(tgt.min()), np.ceil(tgt.max()) + 1)
hist, edges = np.histogram(tgt, bins=bins)
print("\n  Histogram (log10 midpoint | count):")
for left, right, cnt in zip(edges[:-1], edges[1:], hist):
    bar = "#" * int(cnt / max(hist) * 30)
    print(f"  [{left:+.0f},{right:+.0f}) | {bar} {cnt}")

# ── Step 2 — RDKit descriptors ────────────────────────────────────────────────
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

print(f"\nStep 2 — Computing RDKit descriptors …")

def compute_descriptors(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "mw":         Descriptors.MolWt(mol),
        "logP":       Descriptors.MolLogP(mol),
        "tpsa":       Descriptors.TPSA(mol),
        "hbd":        rdMolDescriptors.CalcNumHBD(mol),
        "hba":        rdMolDescriptors.CalcNumHBA(mol),
        "rot_bonds":  rdMolDescriptors.CalcNumRotatableBonds(mol),
        "arom_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
    }

records = []
failed  = []
for _, row in df.iterrows():
    desc = compute_descriptors(row["smiles"])
    if desc is None:
        failed.append(row["drug_name"])
    else:
        desc["drug_name"]     = row["drug_name"]
        desc["log10_midpoint"] = row["log10_midpoint"]
        records.append(desc)

print(f"  Valid SMILES:  {len(records)}")
print(f"  Failed parse:  {len(failed)}")
if failed:
    print(f"  Dropped drugs: {', '.join(failed[:10])}{'…' if len(failed)>10 else ''}")

feat_df = pd.DataFrame(records)
FEATURES = ["mw", "logP", "tpsa", "hbd", "hba", "rot_bonds", "arom_rings"]
X = feat_df[FEATURES].values
y = feat_df["log10_midpoint"].values

# ── Step 3 — Models ───────────────────────────────────────────────────────────
from sklearn.linear_model  import LinearRegression
from sklearn.ensemble      import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics       import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline      import Pipeline

print(f"\nStep 3 — Training models (n={len(X)}, seed=42) …")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

def evaluate(name, pipeline):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    test_r2  = r2_score(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="r2")
    return {
        "name":      name,
        "test_r2":   test_r2,
        "cv_r2_mean": cv_scores.mean(),
        "cv_r2_std":  cv_scores.std(),
        "test_rmse": test_rmse,
    }

# Naive baseline: predict global mean
y_naive   = np.full_like(y_test, y_train.mean())
naive_r2  = r2_score(y_test, y_naive)
naive_rmse = np.sqrt(mean_squared_error(y_test, y_naive))

lin_pipe = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])
rf_pipe  = Pipeline([("rf", RandomForestRegressor(n_estimators=300, random_state=42))])

lin_res  = evaluate("Linear Regression", lin_pipe)
rf_res   = evaluate("Random Forest",     rf_pipe)

results = [lin_res, rf_res]

print(f"\n  Naive baseline (predict mean): R²={naive_r2:.4f}, RMSE={naive_rmse:.3f}")
for r in results:
    print(f"\n  {r['name']}:")
    print(f"    Test R²      = {r['test_r2']:.4f}")
    print(f"    CV R² (5-fold)= {r['cv_r2_mean']:.4f} ± {r['cv_r2_std']:.4f}")
    print(f"    Test RMSE    = {r['test_rmse']:.3f} log10-units")

# Feature importances from RF
rf_pipe.fit(X_train, y_train)
rf_model     = rf_pipe.named_steps["rf"]
importances  = rf_model.feature_importances_
feat_imp     = sorted(zip(FEATURES, importances), key=lambda x: x[1], reverse=True)

print("\n  Random Forest feature importances:")
for feat, imp in feat_imp:
    bar = "#" * int(imp / feat_imp[0][1] * 20)
    print(f"    {feat:<12} {bar} {imp:.4f}")

# ── Step 4 — Honest interpretation ───────────────────────────────────────────
best_cv_r2 = max(r["cv_r2_mean"] for r in results)

if best_cv_r2 > 0.30:
    signal_verdict = "MEANINGFUL SIGNAL — worth pursuing with a GNN"
    signal_flag    = "positive"
elif best_cv_r2 > 0.15:
    signal_verdict = "WEAK SIGNAL — marginal; GNN would be high-risk"
    signal_flag    = "weak"
else:
    signal_verdict = "NO MEANINGFUL SIGNAL — structure poorly predicts the window"
    signal_flag    = "negative"

print(f"\nStep 4 — Verdict: {signal_verdict} (best CV R² = {best_cv_r2:.4f})")

# ── Write report ──────────────────────────────────────────────────────────────
lin_r = results[0]
rf_r  = results[1]

report_lines = [
    "# Level 3 Signal Check — Report",
    "",
    f"**Date:** 2026-06-06  ",
    f"**Dataset:** `experiments/data/therapeutic_windows/therapeutic_window_dataset_filtered.csv`  ",
    f"**Branch:** `experiment/level3-signal-check`",
    "",
    "---",
    "",
    "## Step 1 — Target Distribution",
    "",
    f"Prediction target: `log10((therapeutic_min + therapeutic_max) / 2)` in mg/L.",
    "",
    f"| Stat | Value |",
    f"|------|-------|",
    f"| n    | {len(y)} |",
    f"| min  | {tgt.min():.3f} |",
    f"| max  | {tgt.max():.3f} |",
    f"| mean | {tgt.mean():.3f} |",
    f"| std  | {tgt.std():.3f} |",
    f"| p25  | {tgt.quantile(0.25):.3f} |",
    f"| p50  | {tgt.quantile(0.50):.3f} |",
    f"| p75  | {tgt.quantile(0.75):.3f} |",
    "",
    "The target spans roughly **{:.1f} orders of magnitude**, confirming the log10 transform is appropriate.".format(tgt.max() - tgt.min()),
    "",
    "---",
    "",
    "## Step 2 — Descriptor Computation",
    "",
    f"| Outcome | Count |",
    f"|---------|-------|",
    f"| Valid SMILES | {len(records)} |",
    f"| Failed to parse | {len(failed)} |",
]

if failed:
    report_lines += [
        "",
        f"Failed drugs: {', '.join(failed)}",
    ]

report_lines += [
    "",
    "Descriptors computed: molecular weight, logP (Crippen), TPSA, H-bond donors, H-bond acceptors, rotatable bonds, aromatic ring count.",
    "",
    "---",
    "",
    "## Step 3 — Model Results",
    "",
    f"Train/test split: 80/20, seed=42. Cross-validation: 5-fold.",
    "",
    "| Model | Test R² | CV R² (mean ± std) | Test RMSE (log10 units) |",
    "|-------|---------|--------------------|------------------------|",
    f"| Naive baseline (predict mean) | {naive_r2:.4f} | — | {naive_rmse:.3f} |",
    f"| Linear Regression | {lin_r['test_r2']:.4f} | {lin_r['cv_r2_mean']:.4f} ± {lin_r['cv_r2_std']:.4f} | {lin_r['test_rmse']:.3f} |",
    f"| Random Forest | {rf_r['test_r2']:.4f} | {rf_r['cv_r2_mean']:.4f} ± {rf_r['cv_r2_std']:.4f} | {rf_r['test_rmse']:.3f} |",
    "",
    "### Random Forest Feature Importances",
    "",
    "| Feature | Importance |",
    "|---------|-----------|",
]

for feat, imp in feat_imp:
    report_lines.append(f"| {feat} | {imp:.4f} |")

# Signal interpretation block
report_lines += [
    "",
    "---",
    "",
    "## Step 4 — Honest Interpretation",
    "",
    f"**Overall verdict: {signal_verdict}**",
    "",
    f"Best CV R² across both models: **{best_cv_r2:.4f}**",
    "",
]

if signal_flag == "positive":
    report_lines += [
        "The random forest achieves a cross-validated R² above the 0.30 threshold, indicating that "
        "simple 2D molecular descriptors alone carry a **meaningful** correlation with the therapeutic "
        "midpoint. This validates the core hypothesis: molecular structure informs the window.",
        "",
        "**Does this justify a GNN-based Level 3?** Tentatively **yes**, with caveats:",
        "- A GNN can exploit graph topology not captured by these scalar descriptors; gains of +0.10–0.20 R² over a descriptor RF are plausible.",
        "- The signal is real but modest — Level 3 predictions will carry uncertainty on the order of "
        f"~{rf_r['test_rmse']:.2f} log10-units (~{10**rf_r['test_rmse']:.1f}× concentration factor) even after a GNN.",
        "- Level 3 should be framed as a rough-order-of-magnitude guide, not a precise pharmacokinetic prediction.",
    ]
elif signal_flag == "weak":
    report_lines += [
        "The best CV R² sits in the 0.15–0.30 range — there is a detectable but weak signal. "
        "Simple descriptors explain only a small fraction of the variance in therapeutic windows.",
        "",
        "**Does this justify a GNN-based Level 3?** **Uncertain / high-risk:**",
        "- A GNN may squeeze additional signal from graph topology, but the baseline is fragile.",
        "- Biological factors (protein binding, active transport, metabolism) that are invisible to "
        "SMILES likely dominate the variance.",
        "- Recommend: before building a GNN, investigate whether adding known PK covariates (logD, pKa, "
        "plasma protein binding) lifts R² above 0.30. If not, Level 3 is unlikely to be clinically useful.",
    ]
else:
    report_lines += [
        "The best CV R² is near zero. Simple molecular descriptors derived from SMILES do **not** "
        "meaningfully predict the therapeutic midpoint across this dataset.",
        "",
        "**Does this justify a GNN-based Level 3?** **No — not without additional data.**",
        "",
        "Possible reasons the signal is absent:",
        "- Therapeutic windows are primarily set by biological context (indication, patient population, "
        "toxicity tolerance) rather than by molecular structure alone.",
        "- The 868-drug dataset spans wildly different drug classes; within-class models might work, "
        "but a cross-class universal predictor may be fundamentally ill-posed.",
        "- SMILES encodes structure; it does not encode the clinical judgement behind 'therapeutic range'.",
        "",
        "**Recommendation:** Consolidate on Levels 1–2. Level 1 (caller-supplied window) already "
        "handles known drugs correctly. Level 2 descriptors improve dosing guidance without requiring "
        "a window prediction. Investing engineering effort in a GNN for Level 3 carries high risk of "
        "producing a model that looks plausible but is unreliable in deployment.",
    ]

report_lines += [
    "",
    "---",
    "",
    "*Generated by `experiments/level3_signal/signal_check.py`*",
]

REPORT.write_text("\n".join(report_lines), encoding="utf-8")
print(f"\nReport written to: {REPORT}")
print("\n=== Done ===")
