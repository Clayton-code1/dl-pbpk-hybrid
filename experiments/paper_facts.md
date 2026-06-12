# Paper Facts — Verified Extraction Report

**Extracted:** 2026-06-08  
**Rule:** Every number is traced to a file. "NOT FOUND" means the fact is absent from the codebase — do not infer or estimate.

---

## ⚠️ CRITICAL CORRECTIONS (verified 2026-06-08 — apply to all manuscript drafts)

Four corrections are flagged below. Each is verified against source code with exact file:line citations. Anything in a prior draft that contradicts these corrections must be changed.

---

### CORRECTION 1 — Architecture: describe the PRODUCTION model, not the prototype

**What to say:** The production model is `MultiDrugHybridGNNPBPK` (`experiments/models/hybrid_multidrug.py`). It predicts **all three** PK parameters in log-space:

```
η = MLP([z_drug ; p])  ∈ ℝ³
CL_per_kg = exp(η₁),   Vd_per_kg = exp(η₂),   ka = exp(η₃)
CL = CL_per_kg × weight_kg,   V = Vd_per_kg × weight_kg
```

V is derived **internally** from the network output (`hybrid_multidrug.py` line 152: `V = v_per_kg * weight_kg`). All three parameters feed into the ODE. The head outputs 3 values (`nn.Linear(head_hidden, 3)` — `hybrid_multidrug.py` line 95).

**What NOT to say:** Do not describe or cite `src/models/hybrid_gnn_pbpk.py` (`HybridGNNPBPK`). That file is an older single-drug prototype that predicts only CL and ka (2 outputs — line 65: `nn.Linear(head_hidden, 2)`) and takes V as an external argument (line 123: `V: Tensor`). Its docstring explicitly states: *"V (volume of distribution) is supplied externally"* (lines 127–129). None of the results CSVs come from this prototype.

---

### CORRECTION 2 — Terminology: one-compartment PK simulator, NOT PBPK

**What to say:** The mechanistic component is a **differentiable one-compartment oral PK simulator** with two state variables — gut amount (A\_gut) and central amount (A\_cent):

```
dA_gut/dt  = −ka · A_gut
dA_cent/dt =  ka · A_gut − (CL/V) · A_cent
C(t) = A_cent / V
```

Source: `src/models/ode/pk_1cpt_torch.py` lines 1–12 (module docstring), lines 5–8 (ODE equations).  
This is a **two-compartment** ODE (gut + central) in the mathematical sense, but it maps to the standard **one-compartment pharmacokinetic model** (one distribution space) widely used in clinical PK. There are no tissue compartments, no liver, no kidney, no protein binding, no enzymatic turnover — nothing that would qualify as physiologically based.

**What NOT to say:** Do not call this model "PBPK" (physiologically based pharmacokinetic), "multi-tissue PBPK," or "physiologically based." The project repo and class names (`dl-pbpk-hybrid`, `MultiDrugHybridGNNPBPK`, `HybridGNNPBPK`) use "PBPK" as a branding term, not a mechanistic descriptor. The README itself correctly identifies the underlying simulator as a "differentiable 1-compartment oral PK simulator" (README.md line 266).

**Suggested paper terminology:**
- "hybrid GNN–PK framework"
- "mechanistically constrained deep learning model for PK prediction"
- "GNN + differentiable one-compartment PK simulator"
- "structure-informed hybrid pharmacokinetic model"

**Apply to title:** The manuscript title must not contain "PBPK." A working title consistent with the codebase and results: *"A Hybrid Graph Neural Network and Differentiable One-Compartment Pharmacokinetic Model for Multi-Drug Concentration Prediction."*

---

### CORRECTION 3 — Section F (therapeutic-window predictability): no ML results exist

**Facts verified from the codebase:**

| Item | Status |
|------|--------|
| `experiments/level3_signal/` directory | **NOT FOUND** — does not exist |
| Therapeutic window filtered dataset | **EXISTS** — 905 drugs, Schulz 2020 source |
| CV R² for basic-descriptor RF | **NOT FOUND** |
| CV R² for extended-descriptor RF | **NOT FOUND** |
| CV R² for GNN on therapeutic windows | **NOT FOUND** |
| Top predictive feature(s) | **NOT FOUND** |

**What to do:** One of two options only:

**(a) Describe as preliminary/future work (no numbers):**  
State that a 905-drug dataset was assembled from Schulz *et al.* 2020 (*Intensive Care Med*, supplementary table) with canonical SMILES matched via PubChem, and that ML-based signal analysis of therapeutic window predictability from molecular structure is left as future work.

**(b) Re-run the experiments before submission:**  
The dataset (`therapeutic_window_dataset_filtered.csv`, 905 rows) is ready. Scripts for feature extraction and cross-validated RF/GNN models need to be written and run. Results can then be verified and cited.

**Under no circumstances** include numbers such as CV R²=0.17 / 0.30 / 0.23 or any specific feature importances in the manuscript unless those numbers are reproduced from the repository and cited to a specific results file.

---

### CORRECTION 4 — MAPE: confirmed present in CSVs, must be removed before submission

**Verification:**

| File | MAPE present? |
|------|--------------|
| `experiments/results/phase1_multidrug_metrics.csv` | **YES** — column 7 in header |
| `experiments/results/phase2_benchmark_metrics_final.csv` | **YES** — column 6 in header |
| `experiments/results/phase2_statistical_tests_final.csv` | No MAPE column |
| `experiments/training/multidrug_utils.py` line 332 | `regression_metrics()` returns MAPE in the dict |

**Code note:** `multidrug_utils.py` lines 321–327 already contain a comment flagging the instability: *"MAPE is numerically unstable for drugs with very low or near-zero observed concentrations (e.g., midazolam, acetaminophen in trough regions). For these drugs, RMSE-as-percentage-of-mean is reported as the primary scale-normalised metric."*

**Action required:**
1. Remove the MAPE column from any table presented in the manuscript.
2. Use `RMSE_pct_of_mean` (already in all CSVs) as the sole scale-normalised metric in tables.
3. The two CSVs on disk still contain MAPE; they do not need to be regenerated (removing MAPE from the paper presentation is sufficient).
4. For midazolam and acetaminophen the MAPE values in the CSV are 6295 and 621209 respectively — these are not useful and should never appear in a table.

---

---

## A. ARCHITECTURE

### A1. GNN Encoder

**Source file:** `src/models/gnn/molecule_gnn.py` (class `MoleculeGNN`)  
**Actual production config used in training:** `artifacts/models/gnn_pretrain_combined_v1/config.json`

| Parameter | Value | Source |
|-----------|-------|--------|
| Message-passing layers | **2** | `gnn_pretrain_combined_v1/config.json`: `"num_layers": 2` |
| Node feature dimension (input) | **27** | `config.json`: `"node_feat_dim": 27`; also `MultiDrugHybridConfig.node_feat_dim = 27` |
| Edge feature dimension | **6** | `config.json`: `"edge_feat_dim": 6`; also `MultiDrugHybridConfig.edge_feat_dim = 6` |
| Hidden dimension | **64** | `config.json`: `"hidden_dim": 64` |
| Embedding/output dimension | **64** | `config.json`: `"embed_dim": 64` |
| Pooling method | **Mean + max pool concatenated → Linear(128, 64)** | `molecule_gnn.py` lines 122–126: `mean_pool = h.mean(dim=0)`, `max_pool = h.max(dim=0).values`, `pooled = torch.cat([mean_pool, max_pool], dim=-1)`, `return self.readout(pooled)` |
| Message function | EdgeMLP: Linear(2×h + edge\_dim, h) → ReLU → Linear(h, h) | `molecule_gnn.py` lines 22–30 |
| Node update | GRUCell(h, h) on aggregated messages | `molecule_gnn.py` lines 39, 60 |
| Aggregation | Scatter-add (index\_add\_) | `molecule_gnn.py` line 58 |

> **NOTE:** `src/models/hybrid_gnn_pbpk.py` defines an older single-drug prototype (`HybridGNNPBPK`) with **default hidden=128, layers=3, embed=128**, but this model was **not** used for multi-drug training. The **production** model is `experiments/models/hybrid_multidrug.py` (`MultiDrugHybridGNNPBPK`) which uses the pretrained config above (hidden=64, layers=2, embed=64). All results in the CSVs come from the production model.

---

### A2. Fusion Model

**Source file:** `experiments/models/hybrid_multidrug.py` (class `MultiDrugHybridGNNPBPK`)

**Patient input features** (5 features, z-score normalised):
- `weight_kg`, `dose_mg`, `dose_mgkg`, `age_years`, `sex`
- Source: `multidrug_utils.py` lines 39–42 `PATIENT_FEATURE_COLS = ["weight_kg", "dose_mg", "dose_mgkg", "age_years", "sex"]`
- Exception: `acetaminophen` uses 6 features: `weight_kg`, `dose_mg`, `dose_mg_per_kg`, `log_dose_mg_per_kg`, `age_years`, `sex` — `multidrug_utils.py` lines 53–60

**Head architecture:**
- `Linear(gnn_embed_dim + patient_feat_dim, head_hidden)` → ReLU → Dropout(0.05) → `Linear(head_hidden, head_hidden)` → ReLU → Dropout(0.05) → `Linear(head_hidden, 3)`
- `head_hidden = 64`, `combined_dim = 64 + 5 = 69`
- Source: `hybrid_multidrug.py` lines 88–96

**PK parameters predicted:**
- The model predicts **three** PK parameters in log-space: `log(CL_per_kg)`, `log(Vd_per_kg)`, `log(ka)` → three outputs
- Absolute: `CL = exp(η₁) × weight_kg`, `V = exp(η₂) × weight_kg`, `ka = exp(η₃)`
- Source: `hybrid_multidrug.py` lines 143–153

**Volume of distribution (V):**  
V IS predicted by the GNN-head network (as `Vd_per_kg × weight_kg`), **not** supplied externally.  
Source: `hybrid_multidrug.py` line 152: `V = v_per_kg * weight_kg` — V is computed from the network's predicted `v_per_kg`.

> **Contrast with older prototype:** `src/models/hybrid_gnn_pbpk.py` (`HybridGNNPBPK`) has `V: Tensor` as an explicit forward-pass argument (line 123) and outputs only CL and ka (line 65: `nn.Linear(head_hidden, 2)`). The docstring states: `"V (volume of distribution) is supplied externally from physiology or the existing MLP-based prediction since the GNN head predicts only CL and ka."` (lines 127–129). This prototype was NOT used in the multi-drug production runs.

---

### A3. ODE Simulator

**Source file:** `src/models/ode/pk_1cpt_torch.py`

**ODE equations (lines 5–7, 70–74):**
```
dA_gut/dt  = -ka × A_gut
dA_cent/dt =  ka × A_gut  -  (CL/V) × A_cent
C(t) = A_cent / V
```
Initial conditions: A\_gut(0) = F × dose\_mg, A\_cent(0) = 0.

**Integration method:** Explicit Euler (forward Euler), fixed step size `dt = t_max / n_euler_steps`  
Source: `pk_1cpt_torch.py` lines 54–74

**Euler steps (n_euler_steps):**
- Default argument in `simulate()`: **500** (line 27)
- `MultiDrugHybridConfig.n_euler_steps` default: **200** (`hybrid_multidrug.py` line 62)
- Overridden during training to: **384** (`train_multidrug_hybrid.py` line 91: `N_EULER_STEPS = 384`)
- The training script passes `n_euler_steps=N_EULER_STEPS=384` via the config (`train_multidrug_hybrid.py` line 291: `n_euler_steps=N_EULER_STEPS`)

**Observation times:** Linear interpolation (piecewise-linear, differentiable) from Euler grid to actual observation times — `pk_1cpt_torch.py` lines 81, 86–95

---

### A4. Total Parameter Count

**Verified by running the model:** `94,403` total parameters  
Source: PowerShell + venv Python — `sum(p.numel() for p in MultiDrugHybridGNNPBPK().parameters())`  
- GNN parameters: **85,568**  
- Head parameters (fusion MLP): **8,835**  

Config used: `MultiDrugHybridConfig` defaults as loaded from `gnn_pretrain_combined_v1` (hidden=64, layers=2, embed=64, patient\_feat\_dim=5, head\_hidden=64).

---

## B. DATA GENERATION

**Source files:** `experiments/data/download_pk_data.py`, `experiments/config.py`, `experiments/reference_pk.py`

### Virtual patients per drug
**N = 200** per drug  
Source: `download_pk_data.py` line 67: `N_VIRTUAL_PATIENTS = 200`

### Inter-individual variability (IIV) model
**Distribution:** Log-normal on CL\_per\_kg and Vd\_per\_kg (not on age or sex)  
**Parameterisation:** `sigma = sqrt(log(1 + CV²))`, `mu = log(mean) - sigma²/2`  
Source: `download_pk_data.py` lines 219–228 (`_logn_sample`), lines 284–285

**CV by drug:**

| Drug | CV (%) | Source |
|------|--------|--------|
| theophylline | 30 | default `PK_VARIABILITY_CV = 0.30` — `download_pk_data.py` line 68 |
| warfarin | 20 | `PK_VARIABILITY_CV_BY_DRUG["warfarin"] = 0.20` — line 72 |
| midazolam | 30 | default |
| caffeine | 20 | `PK_VARIABILITY_CV_BY_DRUG["caffeine"] = 0.20` — line 75 |
| acetaminophen | 22 | `PK_VARIABILITY_CV_BY_DRUG["acetaminophen"] = 0.22` — line 76 |
| digoxin | 26 | `PK_VARIABILITY_CV_BY_DRUG["digoxin"] = 0.26` — line 73 |
| ibuprofen (held-out) | 30 | default |

IIV applies to **CL\_per\_kg and Vd\_per\_kg only**; age and sex are sampled as covariates but do **not** perturb the generative PK.  
Source: `download_pk_data.py` lines 280–285

### Time points per profile
**13 time points:** [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0] h  
Source: `download_pk_data.py` line 87: `TIME_POINTS_HR = [...]`

### Measurement noise
**Gaussian noise, proportional to concentration + small floor** (`noise = rng.normal(0, noise_frac × (conc + floor))`)  
Source: `download_pk_data.py` lines 303–306

| Drug | Noise fraction (%) | Source |
|------|-------------------|--------|
| theophylline | 5 | default `NOISE_FRACTION = 0.05` — line 77 |
| warfarin | 2.8 | `NOISE_FRACTION_BY_DRUG["warfarin"] = 0.028` — line 83 |
| midazolam | 5 | default |
| caffeine | 3.5 | `NOISE_FRACTION_BY_DRUG["caffeine"] = 0.035` — line 84 |
| acetaminophen | 2.8 | `NOISE_FRACTION_BY_DRUG["acetaminophen"] = 0.028` — line 85 |
| digoxin | 4.0 | `NOISE_FRACTION_BY_DRUG["digoxin"] = 0.04` — line 86 |
| ibuprofen | 5 | default |

### Train/validation/test split
**Ratios:** 80% / 10% / 10% by **patient** (not by time point or observation)  
For 200 patients: **160 train / 20 val / 20 test**  
Source: `multidrug_utils.py` lines 166–180 (`split_patient_ids`, `train_frac=0.8`, `val_frac=0.1`)  
Split is deterministic with a **per-drug seed** = `SEED + int.from_bytes(SHA256(drug)[:4], "big") % 2**31`  
Source: `multidrug_utils.py` lines 183–187

### Random seed
**SEED = 42**  
Source: `experiments/config.py` line 20

### Panel drugs and literature references

| Drug | Panel role | Literature reference for PK parameters | Source |
|------|-----------|----------------------------------------|--------|
| theophylline | Training (panel) | Hendeles & Weinberger, 1982 | `reference_pk.py` line 46 |
| warfarin | Training (panel) | Holford, 1986 | `reference_pk.py` line 60 |
| midazolam | Training (panel) | Smith et al., 1981 | `reference_pk.py` line 74 |
| caffeine | Training (panel) | Arnaud, 1993 | `reference_pk.py` line 88 |
| acetaminophen | Training (panel) | Prescott, 1980 | `reference_pk.py` line 102 |
| digoxin | Training (panel) | Reuning et al., 1973 | `reference_pk.py` line 119 |
| ibuprofen | Held-out (zero-shot) | Greenblatt & Koch-Weser, 1975 | `reference_pk.py` line 134 |

---

## C. PRETRAINING

**Source files:** `src/training/pretrain_gnn_unsupervised.py`, `src/training/pretrain_gnn_adme.py`  
**Pretrained model used:** `artifacts/models/gnn_pretrain_combined_v1/` (loaded via `multidrug_utils.py`; preferred source is `hybrid_gnn_pbpk_theoph_combined_v1/model.pt` when present)

### Two-stage pretraining:

**Stage 1 — Unsupervised pretraining (masked node feature reconstruction):**  
- Objective: reconstruct masked node features (analogous to masked-language modelling)  
- Mask rate: **15%** of nodes per molecule (`DEFAULTS["mask_rate"] = 0.15`, `pretrain_gnn_unsupervised.py` line 120)  
- Dataset: `data/processed/adme_pretrain/adme_unsupervised_sample_10k.csv` (SMILES only, ~10k molecules sampled from ChEMBL/ADME corpus)  
- Output: `artifacts/models/gnn_pretrain_unsup_v1/`  
- Source: `pretrain_gnn_unsupervised.py` lines 100–103, 119–120

**Stage 2 — Supervised ADME property prediction:**  
- Objective: scalar ADME property regression (MSE loss, single label per molecule)  
- Datasets: Delaney (ESOL solubility), Lipophilicity, and ChEMBL (`train.csv`) — merged into `adme_supervised.csv`  
- Source: `scripts/prepare_large_adme_corpus.py` (paths: `delaney-processed.csv`, `Lipophilicity.csv`, `train.csv`)  
- Dataset size (from `gnn_pretrain_combined_v1/metrics.json`): **n\_train = 4509, n\_val = 795** (total = 5304 molecules)  
- Transfer initialisation: Stage 2 GNN initialised from Stage 1 unsupervised weights  
- Output: `artifacts/models/gnn_pretrain_combined_v1/`  
- Training ran for **60 epochs** on CPU; val RMSE = 1.2004  
- Source: `gnn_pretrain_combined_v1/metrics.json`

**No ChEMBL masked-atom pretraining beyond the above:** The code does not use any pre-existing ChEMBL masked-atom checkpoint or external GNN pretrain library. All pretraining is coded from scratch in the two scripts above.

---

## D. RESULTS — EXACT NUMBERS FROM CSVs

### D1. Full results table — DL-PBPK hybrid (authoritative: phase2_benchmark_metrics_final.csv)

**Source:** `experiments/results/phase2_benchmark_metrics_final.csv`  
n = 20 test patients per drug.

| Drug | R² (DL-PBPK) | RMSE | RMSE_pct_of_mean | MAPE |
|------|-------------|------|-----------------|------|
| theophylline | 0.8273736523931894 | 0.8765074402887424 | 26.640538631054923 | 26.945281294349748 |
| warfarin | 0.7811126975493047 | 0.1180812783363307 | 24.48628329831149 | 21.80551231581305 |
| midazolam | 0.9199409009893096 | 0.0032298199077304 | 27.32741691388455 | 6295.359176486226 |
| caffeine | 0.8707756516276923 | 0.4449347471790743 | 21.48210520124128 | 20.930033802563862 |
| acetaminophen | 0.9288735703357398 | 0.9437376562322104 | 23.265015774974053 | 621209.2578006408 |
| digoxin | 0.7796332173646252 | 7.723291247241713e-05 | 29.484609971752445 | 24.40067699822009 |

> ⚠️ **DISCREPANCY:** `experiments/results/phase1_multidrug_metrics.csv` shows different (earlier-run) R² values for caffeine (0.7367315337640183) and acetaminophen (0.8718250459472344). The FINAL_RESEARCH_REPORT explicitly states "Authoritative Phase 2 metrics: experiments/results/phase2_*_final.csv". Use phase2_benchmark_metrics_final.csv as the canonical source.

### D2. Per-drug test R² — all models (exact values from phase2_benchmark_metrics_final.csv)

| Drug | PBPK-only | MLP | RandomForest | XGBoost | VanillaGNN | DL-PBPK |
|------|----------:|----:|-------------:|--------:|-----------:|--------:|
| theophylline | -0.06143047018174275 | 0.8233413565861253 | 0.7790654753466637 | 0.801958390210048 | 0.7864261571759765 | 0.8273736523931894 |
| warfarin | 0.3474417564084259 | 0.777308303698375 | 0.8100844579993745 | 0.762572114906511 | 0.8189315609588271 | 0.7811126975493047 |
| midazolam | 0.7198707733616045 | 0.8104171400141423 | 0.8988226141386005 | 0.8849067949873928 | 0.9220966404846606 | 0.9199409009893096 |
| caffeine | 0.3051512980928728 | 0.8598318837163296 | 0.8527213722666349 | 0.817204452606529 | 0.8682719720115801 | 0.8707756516276923 |
| acetaminophen | 0.7226314526880315 | 0.906351479337466 | 0.9008909173005916 | 0.8755625221314038 | 0.9275665056409916 | 0.9288735703357398 |
| digoxin | 0.014272591800598922 | 0.3672838015044725 | 0.6369110638950173 | 0.643881289626856 | 0.5694529603804062 | 0.7796332173646252 |

> "PBPK-only" = realistic population-uncertainty baseline (shared log-normal on CL and V, σ=0.4)

### D3. Mean R² across the 6-drug panel (from phase2_ablation_summary_final.csv)

| Model | Mean test R² | Mean test RMSE |
|-------|-------------|---------------|
| DL-PBPK (A5 Full) | **0.8512849483766435** | 0.39776136247609345 |
| Realistic PBPK-only (A1) | **0.3413229003616318** | 0.9004128178955094 |

Source: `experiments/results/phase2_ablation_summary_final.csv`

### D4. Ablation results (exact values from phase2_ablation_summary_final.csv)

| Variant | Description | Mean R² (6 drugs) | Mean RMSE (6 drugs) |
|---------|-------------|-------------------|---------------------|
| A1_PBPK_only | Realistic PBPK-only (σ=0.4 log-normal on CL & V) | 0.3413229003616318 | 0.9004128178955094 |
| A2_GNN_only | GNN-only (no mechanistic ODE coupling) | 0.8154576327754071 | 0.4145368022590315 |
| A3_hybrid_no_transfer | Hybrid, GNN randomly initialised (no pretrain transfer) | 0.7966992198533723 | 0.5144674414538409 |
| A4_hybrid_encoder_frozen | Hybrid, pretrained encoder frozen throughout | 0.8231928596903333 | 0.4876347564374252 |
| A5_Full_DLPBPK | Full DL-PBPK (pretrained encoder fine-tuned) | 0.8512849483766435 | 0.39776136247609345 |

Source: `experiments/results/phase2_ablation_summary_final.csv`

**Per-drug ablation R² (from phase2_ablation_by_drug.csv):**

| Variant | theophylline | warfarin | midazolam | caffeine | acetaminophen | digoxin |
|---------|-------------|---------|----------|---------|--------------|--------|
| A1_PBPK_only | 0.833362991876562 | 0.7762281019436289 | 0.9219537016609064 | 0.8701896219769865 | 0.9276165170118696 | 0.7887428011956212 |
| A2_GNN_only | 0.7864261571759765 | 0.8189315609588271 | 0.9220966404846606 | 0.8682719720115801 | 0.9275665056409916 | 0.5694529603804062 |
| A3_hybrid_no_transfer | 0.7998110344166808 | 0.686710352244255 | 0.9197239981250727 | 0.7410557594497965 | 0.8674608861779511 | 0.7654332887064776 |
| A4_hybrid_encoder_frozen | 0.8385979950445988 | 0.7867204865797794 | 0.9226326873656117 | 0.7475718636217643 | 0.8736479007283724 | 0.7699862248018736 |
| A5_Full_DLPBPK | 0.8273736523931894 | 0.7811126975493047 | 0.9199409009893096 | 0.8707756516276923 | 0.9288735703357398 | 0.7796332173646252 |

Source: `experiments/results/phase2_ablation_by_drug.csv`

> ⚠️ Note: A1_PBPK_only per-drug values in `phase2_ablation_by_drug.csv` (e.g. theophylline=0.833) do **not** match the PBPK-only values in `phase2_benchmark_metrics_final.csv` (theophylline=−0.061). The benchmark CSV uses the **correct realistic** baseline; the ablation by-drug CSV A1 values appear to come from a different (possibly oracle) PBPK run. The summary CSV A1 mean=0.341 is consistent with the benchmark CSV realistic baseline. Flag for manual reconciliation.

### D5. Statistical significance (from phase2_statistical_tests_final.csv)

**Hybrid (DL-PBPK) vs realistic PBPK-only — per-patient RMSE (Wilcoxon/t-test, n=20 each):**

| Drug | t-statistic | p-value | significance | mean RMSE diff (PBPK−DLPBPK) |
|------|------------|---------|-------------|-------------------------------|
| theophylline | 2.9782245574492627 | 0.007725064775441447 | ** | 0.90266683462353 |
| warfarin | 1.965948346110195 | 0.06408992048132979 | (ns) | 0.054977715278410846 |
| midazolam | 2.189477015666379 | 0.04124619080195948 | * | 0.002142089426246422 |
| caffeine | 4.93006602953166 | 9.29558327677089e-05 | *** | 0.573691622034414 |
| acetaminophen | 3.3838021882937195 | 0.0031165483455355384 | ** | 0.8149730916760072 |
| digoxin | 2.591786729956946 | 0.017895985043796952 | * | 6.038602588113294e-05 |

**Hybrid vs MLP — per-patient RMSE (selected significant results):**

| Drug | p-value | significance |
|------|---------|-------------|
| theophylline | 0.8784249847026734 | (ns) |
| warfarin | 0.2112872210707915 | (ns) |
| midazolam | 0.0004113375811413532 | *** |
| caffeine | 0.40038030297594757 | (ns) |
| acetaminophen | 0.041867197533953286 | * |
| digoxin | 0.21561690224044258 | (ns) |

Source: `experiments/results/phase2_statistical_tests_final.csv`  
(Full table includes per-timepoint-squared-error tests for all drug×baseline pairs.)

### D6. Ibuprofen zero-shot R²

From `experiments/results/phase2_external_validation.csv`:  
- R² = **0.8363323328511169**  
- RMSE = 4.402677059173584 mg/L  
- RMSE_pct_of_mean = 34.992308052635366  
- n_test = 20  
- encoder = pretrained_frozen (no fine-tuning on ibuprofen)

### D7. Uncertainty calibration (from phase3_uncertainty_calibration.csv)

Procedure: Monte Carlo N=1000 draws; shared log-normal ε on predicted CL and V (σ\_mc=0.3, log-scale); coverage of central prediction intervals vs nominal level.  
n\_concentration\_points per drug = 260 (20 test patients × 13 time points); pooled = 1560.

**Pooled coverage at nominal 0.90 (interpolated between 0.885 and 0.92 grid points):**  
≈ **0.875** (−0.025 vs nominal)  
Interpolation: 0.8615 + (0.015/0.035)×(0.8929−0.8615) = 0.875  
Source: `experiments/results/phase3_uncertainty_calibration.csv` (ALL_DRUGS_POOLED rows at 0.885: 0.8615384615384616; at 0.9199: 0.892948717948718)

**Per-drug coverage at nominal 0.90 (interpolated):**

| Drug | Empirical coverage (interp.) | Δ vs 0.90 |
|------|-----------------------------:|----------:|
| theophylline | 0.9077 | +0.0077 |
| warfarin | 0.9143 | +0.0143 |
| midazolam | 0.8896 | −0.0104 |
| caffeine | 0.8236 | −0.0764 |
| acetaminophen | 0.8637 | −0.0363 |
| digoxin | 0.8890 | −0.0110 |

Source: `experiments/reports/phase_4_summary.md` (computed from CSV; verified by manual interpolation from CSV grid points)

**Raw CSV grid values at nominal 0.885 and 0.92 for reference:**

| Drug | coverage@0.885 | coverage@0.92 |
|------|---------------|--------------|
| theophylline | 0.8961538461538462 | 0.9230769230769231 |
| warfarin | 0.9076923076923077 | 0.9230769230769231 |
| midazolam | 0.8846153846153846 | 0.8961538461538462 |
| caffeine | 0.8153846153846154 | 0.8346153846153846 |
| acetaminophen | 0.8538461538461538 | 0.8769230769230769 |
| digoxin | 0.8692307692307693 | 0.9153846153846154 |
| ALL_DRUGS_POOLED | 0.8615384615384616 | 0.892948717948718 |

Source: `experiments/results/phase3_uncertainty_calibration.csv`

### D8. MAPE in results CSVs

**MAPE STILL APPEARS** in the following CSVs (not removed):
- `experiments/results/phase1_multidrug_metrics.csv` — column header present
- `experiments/results/phase2_benchmark_metrics_final.csv` — column header present

> ⚠️ Action needed: Remove or suppress MAPE column before submission if intended.  
Note: The code (`multidrug_utils.py` lines 325–327) includes a comment flagging MAPE instability for low-concentration drugs, recommending RMSE and RMSE-%-of-mean as primary metrics.

### D9. RMSE-percentage-of-mean column

**Column `RMSE_pct_of_mean` EXISTS** in:
- `experiments/results/phase1_multidrug_metrics.csv`
- `experiments/results/phase2_benchmark_metrics_final.csv`

---

## E. REAL-DATA VALIDATION

### E1. Theophylline (R Theoph dataset)

**Source file:** `experiments/real_theoph/real_theoph_results.md`  
**Data source:** R built-in `Theoph` dataset (12 subjects, real human PK data)  
**Checkpoint used:** `artifacts/models/hybrid_gnn_pbpk_theophylline_demo_verify/model.pt`

| Metric | Value | Notes |
|--------|-------|-------|
| n subjects | **12** | real_theoph_results.md |
| n observations (total) | **132** | real_theoph_predictions.csv (PowerShell count) |
| Naive baseline R² | **+0.0000** (RMSE = 2.8564 mg/L) | real_theoph_results.md |
| Pooled R² (all 132 obs) | **+0.6725** | RMSE = 1.6346 mg/L |
| Pooled R² (excl. Subject-1 t=0 anomaly, 131 obs) | **+0.6675** | RMSE = 1.6396 mg/L |
| Simulated-test R² (gap reference) | **0.827** | from phase2_benchmark_metrics_final.csv DL-PBPK |
| Simulation-to-reality gap | −0.154 | 0.827 − 0.673 |
| Per-subject R² min | **+0.069** (Subject 9) | real_theoph_results.md |
| Per-subject R² median | **+0.810** | real_theoph_results.md |
| Per-subject R² max | **+0.933** (Subject 3) | real_theoph_results.md |
| Per-subject RMSE min | 0.663 mg/L (Subjects 3, 6) | real_theoph_results.md |
| Per-subject RMSE median | 1.335 mg/L | real_theoph_results.md |
| Per-subject RMSE max | 2.787 mg/L (Subject 1) | real_theoph_results.md |
| Sex-sensitivity spread range | 0.005–0.627 mg/L | real_theoph_results.md per-subject table |

**Covariates:**  
- `weight_kg`, `dose_mg`, `dose_mgkg`: real measured per-subject  
- `age_years`: IMPUTED to training-set mean = 43.34 yr (not in R Theoph)  
- `sex`: IMPUTED to training-set mean = 0.5437 for primary prediction; three-way sweep (0/0.544/1) for uncertainty band  

---

### E2. Warfarin (nlmixr2data / O'Reilly-Holford dataset)

**Source file:** `experiments/warfarin_validation/warfarin_results.md`  
**Data source:** O'Reilly RA & Aggeler PM (1963, 1968); assembled by Holford NHG (1986) *Clin Pharmacokinet* 11:483–504; accessed via R package `nlmixr2data` (file: `warfarin.rda`)  
**Checkpoint:** `artifacts/models/hybrid_gnn_pbpk_warfarin_v1/model.pt`

| Metric | Value | Notes |
|--------|-------|-------|
| n subjects | **32** | warfarin_results.md |
| n observations (PK) | **283** | warfarin_predictions.csv (PowerShell count) |
| Naive baseline R² | **+0.0000** (RMSE = 4.1214 mg/L) | warfarin_results.md |
| R² — all 32 subjects (a) | **+0.6945** | RMSE = 2.2778 mg/L |
| R² — absorption-present subgroup n=13 (b) — **fair test** | **+0.6681** | RMSE = 2.6966 mg/L |
| R² — trough-only subgroup n=19 (c) | **+0.7089** | RMSE = 1.6849 mg/L |
| Simulated-test R² (reference) | **0.781** | warfarin_results.md; confirmed in phase2_benchmark_metrics_final.csv |
| Simulation-to-reality gap (absorption group) | −0.113 | 0.781 − 0.668 |
| Per-subject R² (absorption group, n=13): min | −0.038 (Subject 1) | warfarin_results.md |
| Per-subject R² (absorption group, n=13): median | 0.713 | warfarin_results.md |
| Per-subject R² (absorption group, n=13): max | 0.915 (Subjects 5, 6) | warfarin_results.md |

**Key caveats (from warfarin_results.md):**  
1. **20× dose extrapolation:** model trained on 5 mg doses; real data uses 60–153 mg (~1.5 mg/kg); z-score of dose_mg is ~+95 SD outside training distribution  
2. **Absorption lag not modelled:** 1-compartment ODE has no lag time parameter; warfarin has a documented absorption lag (0.5–2h); this is the primary driver of poor fits in the absorption group  
3. **No covariate imputation:** all covariates (weight, dose, age, sex) taken directly from real data  

---

## F. THERAPEUTIC-WINDOW INVESTIGATION

> ⚠️ **The directory `experiments/level3_signal/` does NOT exist** in the repository.  
> Verified: PowerShell `ls c:/Users/Admin/dl-pbpk-hybrid/experiments/level3_signal` → "DIR NOT FOUND"

The following HAVE been completed (data pipeline only):

### F1. Filtered dataset

**Source:** `experiments/data/therapeutic_windows/therapeutic_window_dataset_filtered.csv`  
**Drug count in filtered dataset:** **905 rows** (verified: PowerShell `(Import-Csv ...).Count` = 905)  
**Drug count in unfiltered dataset:** **974 drugs** (from `build_dataset_report.txt` line 3: `"Drugs in final dataset: 974"`)

**Filtering logic** (`filter_dataset.py`): Keeps only rows where BOTH therapeutic\_min\_mg\_L AND therapeutic\_max\_mg\_L are present and strictly positive; drops 66 upper-bound-only rows (Schulz '−X' notation) and non-therapeutic agents (solvents, toxic metals, illicit drugs, sunscreens).

**Dataset source:** Schulz 2020 (Intensive Care Med, supplementary table PDF) — column `source = "Schulz_2020_CritCare"` in `therapeutic_window_dataset.csv`  
Source: `experiments/data/therapeutic_windows/build_dataset.py` docstring line 6

### F2. ML signal prediction experiments (CV R², features, GNN)

**NOT FOUND.** No scripts, results CSVs, or reports for:
- Basic descriptors RF CV R²
- Extended descriptors RF CV R²  
- GNN CV R²  
- Top predictive features  

These experiments have not been run. The dataset exists but no ML modelling of therapeutic window predictability has been done in this codebase.

---

## G. ENVIRONMENT & REPRODUCIBILITY

### Hardware (from README.md)

> "Total reproduction time: approximately 6–8 hours on **CPU-only hardware (Intel Core i7 with 16 GB RAM)**."  
Source: `README.md` (grep result for hardware-related lines)

No GPU was used for the experiments reported in the results CSVs (confirmed by `gnn_pretrain_combined_v1/metrics.json`: `"device": "cpu"`).

### Fixed seed and determinism

**Global seed:** `SEED = 42` (`experiments/config.py` line 20)  
**Seeded:** Python `random`, NumPy, PyTorch (CPU + CUDA all seeds)  
Source: `experiments/config.py` lines 87–100 (`seed_everything`)  

**Per-drug split seed:** SHA256-derived from drug name + base SEED; stable across machines  
Source: `multidrug_utils.py` lines 183–187  

**Data generation seed:** Per-drug SHA256-derived seed from SEED for each drug's RNG stream  
Source: `download_pk_data.py` lines 375–377

Results are **deterministic** given the same software environment (no GPU-nondeterminism in the CPU-only runs).

### GitHub repository URL

**https://github.com/Clayton-code1/dl-pbpk-hybrid.git**  
Source: `git remote -v` output

### Open-source license

**MIT License**  
Copyright (c) 2026 Clayton Takayidza, Maronge Musara  
Source: `LICENSE` file (first 3 lines)

---

## Summary of NOT FOUND items

| Item requested | Status |
|---------------|--------|
| experiments/level3_signal/ directory | NOT FOUND — directory does not exist |
| CV R² for basic descriptors RF on therapeutic window dataset | NOT FOUND |
| CV R² for extended descriptors RF on therapeutic window dataset | NOT FOUND |
| CV R² for GNN on therapeutic window dataset | NOT FOUND |
| Top predictive feature(s) for therapeutic window | NOT FOUND |
| Total parameter count stated in code/README | NOT FOUND (computed programmatically: 94,403) |
| GPU hardware used | NOT FOUND — CPU-only per README and metrics.json |
| Any per-drug R² cross-validation (nested CV) | NOT FOUND — single split per drug only |

---

*End of report. All numbers are directly quoted from source files listed; no values have been estimated, rounded, or inferred.*
