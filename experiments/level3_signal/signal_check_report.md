# Level 3 Signal Check — Report

**Date:** 2026-06-06  
**Dataset:** `experiments/data/therapeutic_windows/therapeutic_window_dataset_filtered.csv`  
**Branch:** `experiment/level3-signal-check`

---

## Step 1 — Target Distribution

Prediction target: `log10((therapeutic_min + therapeutic_max) / 2)` in mg/L.

| Stat | Value |
|------|-------|
| n    | 868 |
| min  | -4.699 |
| max  | 2.929 |
| mean | -0.484 |
| std  | 1.230 |
| p25  | -1.301 |
| p50  | -0.561 |
| p75  | 0.477 |

The target spans roughly **7.6 orders of magnitude**, confirming the log10 transform is appropriate.

---

## Step 2 — Descriptor Computation

| Outcome | Count |
|---------|-------|
| Valid SMILES | 868 |
| Failed to parse | 0 |

Descriptors computed: molecular weight, logP (Crippen), TPSA, H-bond donors, H-bond acceptors, rotatable bonds, aromatic ring count.

---

## Step 3 — Model Results

Train/test split: 80/20, seed=42. Cross-validation: 5-fold.

| Model | Test R² | CV R² (mean ± std) | Test RMSE (log10 units) |
|-------|---------|--------------------|------------------------|
| Naive baseline (predict mean) | -0.0094 | — | 1.229 |
| Linear Regression | 0.0483 | 0.0885 ± 0.0388 | 1.193 |
| Random Forest | 0.1825 | 0.1667 ± 0.0540 | 1.106 |

### Random Forest Feature Importances

| Feature | Importance |
|---------|-----------|
| logP | 0.2734 |
| mw | 0.2532 |
| tpsa | 0.2077 |
| rot_bonds | 0.0846 |
| hbd | 0.0661 |
| hba | 0.0634 |
| arom_rings | 0.0515 |

---

## Step 4 — Honest Interpretation

**Overall verdict: WEAK SIGNAL — marginal; GNN would be high-risk**

Best CV R² across both models: **0.1667**

The best CV R² sits in the 0.15–0.30 range — there is a detectable but weak signal. Simple descriptors explain only a small fraction of the variance in therapeutic windows.

**Does this justify a GNN-based Level 3?** **Uncertain / high-risk:**
- A GNN may squeeze additional signal from graph topology, but the baseline is fragile.
- Biological factors (protein binding, active transport, metabolism) that are invisible to SMILES likely dominate the variance.
- Recommend: before building a GNN, investigate whether adding known PK covariates (logD, pKa, plasma protein binding) lifts R² above 0.30. If not, Level 3 is unlikely to be clinically useful.


---

## Extended Descriptor Analysis

**Extension date:** 2026-06-06  
**Script:** `experiments/level3_signal/extended_signal_check.py`

### Step 1 — Descriptor inventory

**Reliably computed from SMILES (RDKit only):**

| Category | Descriptors added |
|----------|-------------------|
| Shape / complexity | `mol_mr` (molar refractivity), `frac_csp3`, `n_stereo` (stereocenters), `ring_count`, `heavy_atoms`, `qed`, `chi0n` (topological Chi index) |
| Ionization proxies | `n_basic_n` (basic-N SMARTS count), `n_acid_oh` (acidic-OH count), `n_phenol` (phenol-OH count), `formal_chg` |
| Ionization class flags | `is_base`, `is_acid`, `is_zwitter`, `is_neutral` (mutually exclusive binary) |

Total expanded feature set: **22 descriptors** (7 basic + 15 new).

**Descriptors omitted and why:**

| Property | Status |
|----------|--------|
| pKa | **Omitted** — RDKit has no pKa function. Reliable values require Epik, ACD/pKa, or pkCSM (external, non-open tools). |
| logD at pH 7.4 | **Omitted** — logD = logP − log(1 + 10^(pKa−pH)) requires pKa. Without it, only logP is available, which is already in the basic set. A Henderson–Hasselbalch approximation would require fabricated pKa values and would add structured noise, not signal. |
| Fraction unbound (fu) | **Omitted** — a measured PK property with no reliable SMARTS-based predictor. Including a SMILES-estimated fu would be noise predicting noise and could spuriously inflate or suppress R². |

### Step 2 — Results: basic vs extended descriptors

Same 80/20 split (seed=42) and 5-fold CV as the initial signal check.

| Model | Test R² | CV R² (mean ± std) | RMSE (log10) |
|-------|---------|--------------------|-------------|
| LR  — basic (7 feats) | 0.0483 | 0.0885 ± 0.0388 | 1.193 |
| RF  — basic (7 feats) | 0.1825 | 0.1667 ± 0.0540 | 1.106 |
| LR  — extended (22 feats) | 0.2005 | 0.2232 ± 0.0380 | 1.093 |
| RF  — extended (22 feats) | 0.2869 | 0.3048 ± 0.0753 | 1.033 |

**Delta (RF extended vs RF basic):** +0.1381 CV R²  
**Above 0.30 threshold:** YES

### Extended RF — Feature importances (ranked)

| Rank | Feature | Importance | New? |
|------|---------|-----------|------|
| 1 | `is_base` | 0.1926 | Yes |
| 2 | `mol_mr` | 0.1105 | Yes |
| 3 | `qed` | 0.0961 | Yes |
| 4 | `chi0n` | 0.0880 | Yes |
| 5 | `logP` | 0.0878 |  |
| 6 | `tpsa` | 0.0796 |  |
| 7 | `frac_csp3` | 0.0719 | Yes |
| 8 | `mw` | 0.0612 |  |
| 9 | `rot_bonds` | 0.0434 |  |
| 10 | `hbd` | 0.0304 |  |
| 11 | `n_stereo` | 0.0253 | Yes |
| 12 | `hba` | 0.0236 |  |
| 13 | `n_basic_n` | 0.0229 | Yes |
| 14 | `ring_count` | 0.0193 | Yes |
| 15 | `heavy_atoms` | 0.0170 | Yes |
| 16 | `arom_rings` | 0.0137 |  |
| 17 | `formal_chg` | 0.0050 | Yes |
| 18 | `is_acid` | 0.0044 | Yes |
| 19 | `n_acid_oh` | 0.0044 | Yes |
| 20 | `is_neutral` | 0.0017 | Yes |
| 21 | `is_zwitter` | 0.0011 | Yes |
| 22 | `n_phenol` | 0.0000 | Yes |

### Step 3 — Honest verdict

**ABOVE THRESHOLD — signal now justifies GNN exploration**

Best CV R² = **0.3048** — the 0.30 threshold is crossed. The extended physicochemical descriptors provide sufficient signal to tentatively justify a GNN.

However, the gain over the basic set should be scrutinised: if most of the lift comes from complexity proxies (MW, heavy_atoms, chi0n) rather than ionization features, a GNN's graph-level representation is unlikely to add much beyond a well-tuned descriptor model.

**Recommendation (a):** Proceed to a GNN prototype, but set a hard evaluation criterion: the GNN must beat the best descriptor RF by ≥ 0.05 CV R² to justify the added complexity. If it doesn't, fall back to the descriptor RF as Level 3.


---

## Extended: GNN Comparison

**Date:** 2026-06-06  
**Script:** `experiments/level3_signal/gnn_signal_check.py`  
**Performance bar:** GNN must achieve CV R² ≥ 0.35 (i.e. beat RF-extended by ≥ 0.05) to justify its complexity.

### Architecture

```
MoleculeGNN(node_feat_dim=27, edge_feat_dim=6, hidden_dim=128, num_layers=3, embed_dim=128)
  → Linear(128, 64) → ReLU → Dropout(0.2) → Linear(64, 1)
Total parameters: 484,224 + 8,321 head = 492,545
```

Featurizer: `src/molecules/rdkit_graph.py` (same featurizer used by the production model).  
Stopped at epoch 29/150 (early stopping patience=15 on 10% validation holdout).  
Training device: CPU.  Total CV wall time: 2157s.

### Results

Same 868 drugs, same target, same 80/20 split (seed=42), same 5-fold KFold CV as descriptor experiments.

| Model | CV R² (mean ± std) | Test R² | Test RMSE (log10) |
|-------|--------------------|---------|-------------------|
| RF — basic (7 feats) | 0.1667 ± 0.0540 | 0.1825 | 1.106 |
| RF — extended (22 feats) | 0.3048 ± 0.0753 | 0.2869 | 1.033 |
| **GNN (this run)** | **0.2259 ± 0.0823** | **0.1722** | **1.113** |

**Performance bar (CV R² ≥ 0.35): NOT CLEARED**  
**Delta vs RF-extended: -0.0789**

### Verdict

The GNN achieves CV R² = **0.2259**, a delta of **-0.0789** over the RF-extended baseline. The 0.35 bar is **not cleared**.

**The GNN does not justify its complexity. The 22-descriptor random forest (CV R² = 0.30) is the Level 3 answer.**

#### Why the GNN doesn't help

The dominant predictive signal in therapeutic windows comes from coarse physicochemical properties: ionization class (`is_base`, rank-1 feature in the RF), molecular size (`mol_mr`), and lipophilicity (`logP`, `QED`). The extended RF already captures these via scalar descriptors. Graph topology — bond patterns, ring connectivity, precise substitution geometry — adds little additional information, because the remaining variance is driven by biological factors that no SMILES-based model can access:

- Plasma protein binding (fu) — unmeasured, structurally non-trivial
- Active transporter expression (OATP, P-gp) — tissue-level biology
- Indication-specific toxicity tolerance — clinical convention, not chemistry

A GNN reading the same SMILES as the descriptor RF faces the same hard ceiling. More architectural complexity does not recover missing biological information.

#### Final recommendation

| Level | Implementation | Status |
|-------|---------------|--------|
| Level 1 | Caller-supplied therapeutic window | **Done — primary path** |
| Level 2 | Descriptor-based guidance (existing) | **Done** |
| Level 3 | 22-descriptor RF (CV R²=0.30, interpretable) | **Validated here — adopt as Level 3 fallback** |
| Level 3 GNN | Graph neural network | **Not justified — bar not cleared** |

Document the GNN result in the manuscript as a research finding: the signal ceiling is structural (coarse physicochemical properties), not architectural. A GNN offers no advantage over an interpretable RF because the limiting factor is missing biological covariates, not inadequate structural representation.

---

*Generated by `experiments/level3_signal/signal_check.py`, `experiments/level3_signal/extended_signal_check.py`, and `experiments/level3_signal/gnn_signal_check.py`*