"""
Level 3 Extended Signal Check — physicochemical covariate expansion.
Tests whether additional RDKit descriptors (ionization proxies, complexity,
shape) lift CV R2 above the 0.30 go/no-go threshold for a GNN-based Level 3.

Pure analysis. No project model/API code is imported or modified.
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "experiments/data/therapeutic_windows/therapeutic_window_dataset_filtered.csv"
OUT_DIR   = Path(__file__).parent

# ── Data load ─────────────────────────────────────────────────────────────────
raw = pd.read_csv(DATA_PATH)
df  = raw[raw["likely_therapeutic_agent"] == True].copy()
df["midpoint"] = (df["therapeutic_min_mg_L"] + df["therapeutic_max_mg_L"]) / 2.0
df = df[df["midpoint"] > 0].copy()
df["log10_midpoint"] = np.log10(df["midpoint"])
print(f"Dataset: {len(df)} rows after filter")

# ── Descriptor functions ───────────────────────────────────────────────────────
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, QED, Crippen

# SMARTS for ionization proxies
# Basic N: sp3 or aromatic N that is not amide/sulfonamide/pyrrole NH
_BASIC_N   = Chem.MolFromSmarts("[NH2,NH1,NH0;!$(NC=O);!$(NS=O);!$([nH])]")
# Acidic OH: carboxylic, sulfonic, phosphoric
_ACID_OH   = Chem.MolFromSmarts("[OH;$(O-C=O),$(O-S=O),$(O-P=O)]")
# Phenol: aromatic OH (weaker acid, different distribution behaviour)
_PHENOL    = Chem.MolFromSmarts("[OH;c]")


def compute_extended(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # ── Basic (from previous run, kept for baseline comparison) ───────────────
    mw        = Descriptors.MolWt(mol)
    logp      = Crippen.MolLogP(mol)
    tpsa      = rdMolDescriptors.CalcTPSA(mol)
    hbd       = rdMolDescriptors.CalcNumHBD(mol)
    hba       = rdMolDescriptors.CalcNumHBA(mol)
    rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    arom_rings = rdMolDescriptors.CalcNumAromaticRings(mol)

    # ── New: shape/complexity ─────────────────────────────────────────────────
    mol_mr      = Crippen.MolMR(mol)
    frac_csp3   = rdMolDescriptors.CalcFractionCSP3(mol)
    n_stereo    = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
    ring_count  = rdMolDescriptors.CalcNumRings(mol)
    heavy_atoms = mol.GetNumHeavyAtoms()
    qed         = QED.qed(mol)
    formal_chg  = Chem.GetFormalCharge(mol)
    chi0n       = rdMolDescriptors.CalcChi0n(mol)

    # ── New: ionization proxies via SMARTS ────────────────────────────────────
    # These are NOT pKa values. They are structural counts that correlate with
    # ionization state at physiological pH and affect Vd / protein binding.
    n_basic  = len(mol.GetSubstructMatches(_BASIC_N))
    n_acid   = len(mol.GetSubstructMatches(_ACID_OH))
    n_phenol = len(mol.GetSubstructMatches(_PHENOL))

    # Ionization class encoded as binary flags (4 mutually exclusive classes)
    has_basic  = n_basic  > 0
    has_acidic = (n_acid + n_phenol) > 0
    is_base      = int(has_basic and not has_acidic)
    is_acid      = int(has_acidic and not has_basic)
    is_zwitter   = int(has_basic and has_acidic)
    is_neutral   = int(not has_basic and not has_acidic)

    return {
        # basic
        "mw": mw, "logP": logp, "tpsa": tpsa,
        "hbd": hbd, "hba": hba, "rot_bonds": rot_bonds, "arom_rings": arom_rings,
        # new
        "mol_mr": mol_mr, "frac_csp3": frac_csp3, "n_stereo": n_stereo,
        "ring_count": ring_count, "heavy_atoms": heavy_atoms, "qed": qed,
        "formal_chg": formal_chg, "chi0n": chi0n,
        "n_basic_n": n_basic, "n_acid_oh": n_acid, "n_phenol": n_phenol,
        "is_base": is_base, "is_acid": is_acid,
        "is_zwitter": is_zwitter, "is_neutral": is_neutral,
    }


# ── Compute descriptors ────────────────────────────────────────────────────────
print("\nStep 1 — Computing extended descriptors …")
records, failed = [], []
for _, row in df.iterrows():
    desc = compute_extended(row["smiles"])
    if desc is None:
        failed.append(row["drug_name"])
    else:
        desc["log10_midpoint"] = row["log10_midpoint"]
        records.append(desc)

feat_df = pd.DataFrame(records)
print(f"  Valid: {len(records)},  Failed: {len(failed)}")
if failed:
    print(f"  Dropped: {', '.join(failed)}")

BASIC_FEATS = ["mw", "logP", "tpsa", "hbd", "hba", "rot_bonds", "arom_rings"]
NEW_FEATS   = ["mol_mr", "frac_csp3", "n_stereo", "ring_count", "heavy_atoms",
               "qed", "formal_chg", "chi0n",
               "n_basic_n", "n_acid_oh", "n_phenol",
               "is_base", "is_acid", "is_zwitter", "is_neutral"]
ALL_FEATS   = BASIC_FEATS + NEW_FEATS

print(f"\n  Descriptor breakdown:")
print(f"    Basic (previous run):  {len(BASIC_FEATS)} features — {BASIC_FEATS}")
print(f"    New additions:         {len(NEW_FEATS)} features — {NEW_FEATS}")
print(f"    Total expanded set:    {len(ALL_FEATS)} features")
print()
print("  Descriptors NOT included and why:")
print("    pKa:              RDKit has no pKa function; Epik/ACD/pkCSM required")
print("    logD at pH 7.4:   Requires pKa to apply Henderson-Hasselbalch correction;")
print("                      using logP as-is is the best pure-SMILES approximation")
print("    Fraction unbound: Measured PK property; no reliable SMARTS-based predictor;")
print("                      OMITTED to avoid adding noise that masquerades as signal")

X_all   = feat_df[ALL_FEATS].values
X_basic = feat_df[BASIC_FEATS].values
y       = feat_df["log10_midpoint"].values

# ── Models ────────────────────────────────────────────────────────────────────
from sklearn.linear_model    import LinearRegression
from sklearn.ensemble        import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics         import r2_score, mean_squared_error
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline

print("\nStep 2 — Training models (seed=42, same split as signal_check.py) …")

X_train_all,  X_test_all,  y_train, y_test = train_test_split(
    X_all, y, test_size=0.20, random_state=42)

# Re-derive basic split from same indices for fair comparison
X_train_basic, X_test_basic, _, _ = train_test_split(
    X_basic, y, test_size=0.20, random_state=42)

def evaluate(name, pipeline, X_tr, X_te, y_tr, y_te, X_full, y_full):
    pipeline.fit(X_tr, y_tr)
    y_pred   = pipeline.predict(X_te)
    test_r2  = r2_score(y_te, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    cv_scores = cross_val_score(pipeline, X_full, y_full, cv=5, scoring="r2")
    return {
        "name":       name,
        "test_r2":    test_r2,
        "cv_r2_mean": cv_scores.mean(),
        "cv_r2_std":  cv_scores.std(),
        "test_rmse":  test_rmse,
    }

# Previous baseline numbers (from signal_check.py, reproduced here for verification)
lin_basic = evaluate("LR  — basic (7 feats)",
    Pipeline([("sc", StandardScaler()), ("lr", LinearRegression())]),
    X_train_basic, X_test_basic, y_train, y_test, X_basic, y)

rf_basic  = evaluate("RF  — basic (7 feats)",
    Pipeline([("rf", RandomForestRegressor(n_estimators=300, random_state=42))]),
    X_train_basic, X_test_basic, y_train, y_test, X_basic, y)

lin_ext  = evaluate("LR  — extended (22 feats)",
    Pipeline([("sc", StandardScaler()), ("lr", LinearRegression())]),
    X_train_all, X_test_all, y_train, y_test, X_all, y)

rf_ext   = evaluate("RF  — extended (22 feats)",
    Pipeline([("rf", RandomForestRegressor(n_estimators=300, random_state=42))]),
    X_train_all, X_test_all, y_train, y_test, X_all, y)

all_results = [lin_basic, rf_basic, lin_ext, rf_ext]

print(f"\n{'Model':<30} {'Test R2':>8} {'CV R2 mean':>12} {'CV R2 std':>10} {'RMSE':>8}")
print("-" * 70)
for r in all_results:
    print(f"  {r['name']:<28} {r['test_r2']:>8.4f} {r['cv_r2_mean']:>12.4f} "
          f"{r['cv_r2_std']:>10.4f} {r['test_rmse']:>8.3f}")

# Delta vs previous RF baseline
prev_cv_r2  = rf_basic["cv_r2_mean"]
ext_cv_r2   = rf_ext["cv_r2_mean"]
delta       = ext_cv_r2 - prev_cv_r2
threshold   = 0.30

print(f"\n  Previous RF CV R2 (7 basic feats): {prev_cv_r2:.4f}")
print(f"  Extended RF CV R2 (22 feats):      {ext_cv_r2:.4f}")
print(f"  Delta:                             {delta:+.4f}")
print(f"  Threshold:                          0.30")
print(f"  Above threshold:                   {'YES' if ext_cv_r2 >= threshold else 'NO'}")

# ── Feature importances (extended RF) ─────────────────────────────────────────
rf_ext_pipe = Pipeline([("rf", RandomForestRegressor(n_estimators=300, random_state=42))])
rf_ext_pipe.fit(X_train_all, y_train)
importances = rf_ext_pipe.named_steps["rf"].feature_importances_
feat_imp    = sorted(zip(ALL_FEATS, importances), key=lambda x: x[1], reverse=True)

print("\n  Extended RF feature importances (all 22 features):")
for feat, imp in feat_imp:
    bar = "#" * int(imp / feat_imp[0][1] * 24)
    flag = " *NEW*" if feat in NEW_FEATS else ""
    print(f"    {feat:<14} {bar:<24} {imp:.4f}{flag}")

# ── Step 3 verdict ─────────────────────────────────────────────────────────────
best_cv    = max(r["cv_r2_mean"] for r in all_results)
best_model = max(all_results, key=lambda r: r["cv_r2_mean"])

print(f"\nStep 3 — Verdict:")
if best_cv >= 0.30:
    verdict = "ABOVE THRESHOLD — signal now justifies GNN exploration"
    rec = "a"
elif best_cv >= 0.22:
    verdict = "IMPROVED BUT STILL BELOW THRESHOLD — Level 2.5 territory"
    rec = "b"
else:
    verdict = "STILL WEAK — extended descriptors do not rescue the signal"
    rec = "c"

print(f"  {verdict}")
print(f"  Best CV R2: {best_cv:.4f} ({best_model['name']})")
print(f"  Recommendation class: ({rec})")

# ── Write extended section to report ──────────────────────────────────────────
report_path = OUT_DIR / "signal_check_report.md"
existing    = report_path.read_text(encoding="utf-8")

# Strip the trailing generated-by line so we append cleanly before it
footer = "\n---\n\n*Generated by `experiments/level3_signal/signal_check.py`*"
if existing.endswith(footer):
    existing = existing[: -len(footer)]

new_section = [
    "",
    "---",
    "",
    "## Extended Descriptor Analysis",
    "",
    "**Extension date:** 2026-06-06  ",
    "**Script:** `experiments/level3_signal/extended_signal_check.py`",
    "",
    "### Step 1 — Descriptor inventory",
    "",
    "**Reliably computed from SMILES (RDKit only):**",
    "",
    "| Category | Descriptors added |",
    "|----------|-------------------|",
    "| Shape / complexity | `mol_mr` (molar refractivity), `frac_csp3`, `n_stereo` (stereocenters), `ring_count`, `heavy_atoms`, `qed`, `chi0n` (topological Chi index) |",
    "| Ionization proxies | `n_basic_n` (basic-N SMARTS count), `n_acid_oh` (acidic-OH count), `n_phenol` (phenol-OH count), `formal_chg` |",
    "| Ionization class flags | `is_base`, `is_acid`, `is_zwitter`, `is_neutral` (mutually exclusive binary) |",
    "",
    "Total expanded feature set: **22 descriptors** (7 basic + 15 new).",
    "",
    "**Descriptors omitted and why:**",
    "",
    "| Property | Status |",
    "|----------|--------|",
    "| pKa | **Omitted** — RDKit has no pKa function. Reliable values require Epik, ACD/pKa, or pkCSM (external, non-open tools). |",
    "| logD at pH 7.4 | **Omitted** — logD = logP − log(1 + 10^(pKa−pH)) requires pKa. Without it, only logP is available, which is already in the basic set. A Henderson–Hasselbalch approximation would require fabricated pKa values and would add structured noise, not signal. |",
    "| Fraction unbound (fu) | **Omitted** — a measured PK property with no reliable SMARTS-based predictor. Including a SMILES-estimated fu would be noise predicting noise and could spuriously inflate or suppress R². |",
    "",
    "### Step 2 — Results: basic vs extended descriptors",
    "",
    f"Same 80/20 split (seed=42) and 5-fold CV as the initial signal check.",
    "",
    "| Model | Test R² | CV R² (mean ± std) | RMSE (log10) |",
    "|-------|---------|--------------------|-------------|",
]

for r in all_results:
    new_section.append(
        f"| {r['name']} | {r['test_r2']:.4f} | "
        f"{r['cv_r2_mean']:.4f} ± {r['cv_r2_std']:.4f} | {r['test_rmse']:.3f} |"
    )

new_section += [
    "",
    f"**Delta (RF extended vs RF basic):** {delta:+.4f} CV R²  ",
    f"**Above 0.30 threshold:** {'YES' if ext_cv_r2 >= threshold else 'NO'}",
    "",
    "### Extended RF — Feature importances (ranked)",
    "",
    "| Rank | Feature | Importance | New? |",
    "|------|---------|-----------|------|",
]

for rank, (feat, imp) in enumerate(feat_imp, 1):
    is_new = "Yes" if feat in NEW_FEATS else ""
    new_section.append(f"| {rank} | `{feat}` | {imp:.4f} | {is_new} |")

# Interpretation block
new_section += ["", "### Step 3 — Honest verdict", "", f"**{verdict}**", ""]

if rec == "a":
    new_section += [
        f"Best CV R² = **{best_cv:.4f}** — the 0.30 threshold is crossed. The extended "
        "physicochemical descriptors provide sufficient signal to tentatively justify a GNN.",
        "",
        "However, the gain over the basic set should be scrutinised: if most of the lift "
        "comes from complexity proxies (MW, heavy_atoms, chi0n) rather than ionization "
        "features, a GNN's graph-level representation is unlikely to add much beyond a "
        "well-tuned descriptor model.",
        "",
        "**Recommendation (a):** Proceed to a GNN prototype, but set a hard evaluation "
        "criterion: the GNN must beat the best descriptor RF by ≥ 0.05 CV R² to justify "
        "the added complexity. If it doesn't, fall back to the descriptor RF as Level 3.",
    ]
elif rec == "b":
    new_section += [
        f"Best CV R² = **{best_cv:.4f}** — improved over the basic-descriptor baseline "
        f"({prev_cv_r2:.4f}) by **{delta:+.4f}**, but still below 0.30.",
        "",
        "The additional descriptors carry some incremental signal, but the improvement is "
        "modest. A GNN would inherit this same ceiling: the missing variance is almost "
        "certainly in biological properties (protein binding, transporter expression, "
        "metabolism) that SMILES cannot encode.",
        "",
        "**Recommendation (b):** Do not build a full GNN for Level 3 at this time. "
        "Instead, consider a lightweight 'Level 2.5': expose the extended descriptor model "
        "as a rough order-of-magnitude window estimate for completely unknown drugs, with "
        "explicit uncertainty bounds (±1 log10 unit). This is honest about its limitations "
        "while providing some value over no estimate at all. Document the ceiling as a "
        "research limitation in the manuscript.",
    ]
else:
    new_section += [
        f"Best CV R² = **{best_cv:.4f}** — extending to {len(ALL_FEATS)} physicochemical "
        f"descriptors did not meaningfully improve over the basic 7-descriptor result "
        f"({prev_cv_r2:.4f}). The delta is **{delta:+.4f}**.",
        "",
        "This is a confirmed negative. The variance in therapeutic windows across this "
        "868-drug dataset is not recoverable from SMILES-derived properties, whether basic "
        "or extended. A GNN would face the same hard ceiling.",
        "",
        "**Recommendation (c):** Consolidate on Levels 1–2. Document Level 3 as a "
        "research limitation: therapeutic window prediction from structure alone is "
        "ill-posed at the population level, likely because windows are set by biological "
        "context and clinical convention as much as by pharmacokinetic properties. "
        "Future work would require measured fu, pKa, and transporter substrate data — "
        "none of which are computable from SMILES.",
    ]

new_section += [
    "",
    "---",
    "",
    "*Generated by `experiments/level3_signal/signal_check.py` and "
    "`experiments/level3_signal/extended_signal_check.py`*",
]

updated = existing + "\n" + "\n".join(new_section)
report_path.write_text(updated, encoding="utf-8")
print(f"\nReport updated: {report_path}")
print("\n=== Done ===")
