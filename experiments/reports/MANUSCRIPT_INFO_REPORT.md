# MANUSCRIPT INFORMATION REPORT

Consolidated extraction for manuscript preparation. **No fabricated values** — entries marked NOT FOUND / NOT AVAILABLE are explicit gaps.


---

### SECTION 1 — Mathematical Formulation (verbatim)

# Mathematical formulation (LaTeX-friendly; paste into manuscript)

Notation below matches the multi-drug hybrid in `experiments/models/hybrid_multidrug.py` and the differentiable simulator in `src/models/ode/pk_1cpt_torch.py`.

---

## Drug representation

Let $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ be a molecular graph with node features $\mathbf{X} \in \mathbb{R}^{|\mathcal{V}| \times d_x}$ and edge features per directed edge. A graph neural network encodes the drug:

$$
\mathbf{z}_{\mathrm{drug}} = \mathrm{GNN}(\mathcal{G}) \in \mathbb{R}^{d_e}.
$$

## Patient covariates and fusion

Let $\mathbf{p} \in \mathbb{R}^{d_p}$ denote the patient / regimen feature vector used by the fusion head (e.g. normalised weight, dose, demographic channels). Parameters are predicted from the concatenation $[\mathbf{z}_{\mathrm{drug}} ;\, \mathbf{p}]$:

$$
\begin{aligned}
\boldsymbol{\eta} &= \mathrm{MLP}\bigl([\mathbf{z}_{\mathrm{drug}} ;\, \mathbf{p}]\bigr) \in \mathbb{R}^{3}, \\
\mathrm{CL}_{\mathrm{kg}} &= \exp(\eta_1) \in \mathbb{R}_{>0}, \qquad
\mathrm{V}_{\mathrm{kg}} = \exp(\eta_2) \in \mathbb{R}_{>0}, \qquad
k_a = \exp(\eta_3) \in \mathbb{R}_{>0},
\end{aligned}
$$

with clamps/floors as in code. Total clearance and volume scale with body weight $W\,\mathrm{(kg)}$:

$$
\mathrm{CL} = \mathrm{CL}_{\mathrm{kg}} \cdot W, \qquad V = \mathrm{V}_{\mathrm{kg}} \cdot W.
$$

Oral bioavailability $F \in (0,1]$ scales the absorbed dose:

$$
D_{\mathrm{eff}} = F \cdot D, \qquad D \text{ in mg}.
$$

## One-compartment oral PK simulator

Let $A_{\mathrm{gut}}(t)$ and $A_{\mathrm{cent}}(t)$ be gut and central amounts (mg). The ODE system is

$$
\frac{\mathrm{d}A_{\mathrm{gut}}}{\mathrm{d}t} = -k_a A_{\mathrm{gut}}, \qquad
\frac{\mathrm{d}A_{\mathrm{cent}}}{\mathrm{d}t} = k_a A_{\mathrm{gut}} - \frac{\mathrm{CL}}{V} A_{\mathrm{cent}}.
$$

Initial condition: $A_{\mathrm{gut}}(0) = D_{\mathrm{eff}}$, $A_{\mathrm{cent}}(0) = 0$. Plasma concentration (mg/L) is

$$
C(t) = \frac{A_{\mathrm{cent}}(t)}{V}.
$$

The implementation uses **explicit Euler** on a fine sub-grid and **linear interpolation** to observation times $\{t_j\}_{j=1}^{T}$ so that $C(t_j)$ is differentiable in $(\mathrm{CL}, V, k_a, D_{\mathrm{eff}})$.

## Supervision

For patient $i$ and drug $k$, observed concentrations $\{y_{ij}\}$ are compared to predictions $\{\hat{C}_{ij}\}$ (e.g. MSE or drug-weighted loss). Training uses held-out splits; external drugs test encoder transfer.

## Baselines and uncertainty (reference)

- **Realistic PBPK-only baseline (Phase 2):** literature-anchored $\mathrm{CL}, V$ perturbed by a **shared** log-normal factor $\exp(\sigma z)$ per patient on both $\mathrm{CL}$ and $V$, with $\sigma = 0.4$ on the log scale.
- **Monte Carlo prediction intervals (Phase 3.2):** conditional on hybrid point estimates $(\widehat{\mathrm{CL}}, \widehat{V})$, draws

$$
\mathrm{CL}^{(m)} = \widehat{\mathrm{CL}} \cdot \exp(\sigma_{\mathrm{mc}} \epsilon^{(m)}), \quad
V^{(m)} = \widehat{V} \cdot \exp(\sigma_{\mathrm{mc}} \epsilon^{(m)}), \quad \epsilon^{(m)} \sim \mathcal{N}(0,1),
$$

with $\sigma_{\mathrm{mc}} = 0.3$, $m = 1,\ldots,1000$, and $k_a$ fixed at the model prediction; empirical interval coverage is tabulated against nominal levels.

## Explaining predicted exposure (Phase 3.3)

Let $A\!\mathrm{UC} = \int_0^{T} \hat{C}(t)\,\mathrm{d}t$ (trapezoidal rule on predicted $\hat{C}$). **KernelSHAP** approximates Shapley values for patient features with the molecular graph **fixed**, i.e. attributions are **local to $\mathbf{p}$** conditional on $\mathcal{G}$.

---

Raw LaTeX blocks (if importing `.tex`):

```latex
% Drug embedding
\mathbf{z}_{\mathrm{drug}} = \mathrm{GNN}(\mathcal{G})

% PK head & scaling
\boldsymbol{\eta} = \mathrm{MLP}([\mathbf{z}_{\mathrm{drug}};\mathbf{p}]),\quad
\mathrm{CL} = e^{\eta_1} W,\quad V = e^{\eta_2} W,\quad k_a = e^{\eta_3}

% ODE
\dot{A}_{\mathrm{gut}} = -k_a A_{\mathrm{gut}},\quad
\dot{A}_{\mathrm{cent}} = k_a A_{\mathrm{gut}} - (\mathrm{CL}/V) A_{\mathrm{cent}},\quad
C = A_{\mathrm{cent}}/V
```


#### Code-derived supplements (items 1.1–1.9)

**1.1 GNN message-passing (`src/models/gnn/molecule_gnn.py`)**  
Messages $\mathbf{m}_{j\to i} = \mathrm{EdgeMLP}([\mathbf{h}_j ; \mathbf{h}_i ; \mathbf{e}_{ji}])$; aggregation $\tilde{\mathbf{h}}_i = \sum_{j:(j,i)\in\mathcal{E}} \mathbf{m}_{j\to i}$ via `index_add_` (sum over incoming); update $\mathbf{h}'_i = \mathrm{GRUCell}(\tilde{\mathbf{h}}_i, \mathbf{h}_i)$.

**1.2 Atom features (`src/molecules/rdkit_graph.py`), dim = 27**  
One-hot atom type (10); one-hot degree $\le 5$ (dim 6); formal charge scalar; aromatic flag; one-hot hybridisation (4); one-hot H-count $\le 4$ (dim 5).

**1.3 Bond features, dim = 6**  
One-hot bond type (4); conjugated; in-ring.

**1.4 Readout**  
Concatenate mean and max pooling over nodes $\to$ `Linear` to `embed_dim`.

**1.5 ODE (`src/models/ode/pk_1cpt_torch.py`)**  
$\mathrm{d}A_\mathrm{gut}/\mathrm{d}t = -k_a A_\mathrm{gut}$; $\mathrm{d}A_\mathrm{cent}/\mathrm{d}t = k_a A_\mathrm{gut} - (\mathrm{CL}/V)A_\mathrm{cent}$; $C=A_\mathrm{cent}/V$; explicit Euler + interpolation.

**1.6 Fusion (`experiments/models/hybrid_multidrug.py`)**  
MLP on $[\mathbf{z}_\mathrm{drug}; \mathbf{p}]$ $\to$ three logits (clamped) $\to$ $\exp$ with floors $\to$ $\mathrm{CL}_\mathrm{kg}, V_\mathrm{kg}, k_a$; scale by weight.

**1.7 Loss (`experiments/training/train_multidrug_hybrid.py`)**  
$\mathcal{L} = \mathrm{MSE}_{\mathrm{conc,rel}} + 0.10\,\mathrm{MSE}_{\mathrm{AUC}} + 0.05\,\mathrm{MSE}_{\mathrm{Cmax}} + \lambda_\mathrm{pk} \mathcal{L}_\mathrm{PK}$ with default $\lambda_\mathrm{pk}=0.12$; per-drug: warfarin 0.72, caffeine 0.28, acetaminophen 0.35, digoxin 0.35. $\mathcal{L}_\mathrm{PK}$ is sum of log-MSE for CL and V vs CSV targets when present.

**1.8 Unsupervised objective (`src/training/pretrain_gnn_unsupervised.py`)**  
Masked node-feature reconstruction; `mask_rate` stored as **0.15** in `gnn_pretrain_unsup_v1/metrics.json`.

**1.9 Transfer (`experiments/training/multidrug_utils.py`)**  
Load `gnn.*` from `hybrid_gnn_pbpk_theoph_combined_v1/model.pt` if present, else `gnn_pretrain_combined_v1/model_gnn.pt`; freeze then unfreeze schedule in `train_multidrug_hybrid.py`.


---

### SECTION 2 — GNN Architecture & Training Hyperparameters

#### Full `MoleculeGNN` module (verbatim)

```python
"""Pure-PyTorch Graph Neural Network for molecular encoding.

Implements a message-passing network with:
- EdgeMLP that computes messages from (h_src, h_dst, edge_attr)
- Aggregation via scatter-add (index_add_)
- GRU-based node state update
- Mean + max pool readout -> graph-level embedding
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class EdgeMLP(nn.Module):
    """Compute edge messages from source, destination, and edge features."""

    def __init__(self, node_dim: int, edge_dim: int, msg_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, msg_dim),
            nn.ReLU(),
            nn.Linear(msg_dim, msg_dim),
        )

    def forward(self, h_src: Tensor, h_dst: Tensor, edge_attr: Tensor) -> Tensor:
        inp = torch.cat([h_src, h_dst, edge_attr], dim=-1)
        return self.mlp(inp)


class MessagePassingLayer(nn.Module):
    """One round of message passing + GRU update."""

    def __init__(self, hidden_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.edge_mlp = EdgeMLP(hidden_dim, edge_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(
        self, h: Tensor, edge_index: Tensor, edge_attr: Tensor
    ) -> Tensor:
        """
        Parameters
        ----------
        h : [N, hidden_dim]
        edge_index : [2, E]
        edge_attr : [E, edge_dim]
        """
        if edge_index.size(1) == 0:
            return h

        src, dst = edge_index[0], edge_index[1]
        msgs = self.edge_mlp(h[src], h[dst], edge_attr)  # [E, hidden_dim]

        agg = torch.zeros_like(h)  # [N, hidden_dim]
        agg.index_add_(0, dst, msgs)

        h_new = self.gru(agg, h)
        return h_new


class MoleculeGNN(nn.Module):
    """Graph neural network that maps a molecular graph to a fixed-size embedding.

    Parameters
    ----------
    node_feat_dim : int
        Dimensionality of input node feature vectors.
    edge_feat_dim : int
        Dimensionality of input edge feature vectors.
    hidden_dim : int
        Width of internal node representations.
    num_layers : int
        Number of message-passing rounds.
    embed_dim : int
        Size of the output graph-level embedding.
    """

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        embed_dim: int = 128,
    ) -> None:
        super().__init__()
        self.node_encoder = nn.Linear(node_feat_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [MessagePassingLayer(hidden_dim, edge_feat_dim) for _ in range(num_layers)]
        )
        self.readout = nn.Linear(2 * hidden_dim, embed_dim)

        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embed_dim = embed_dim

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor
    ) -> Tensor:
        """Produce a graph-level embedding.

        Parameters
        ----------
        x : [N, node_feat_dim]
        edge_index : [2, E]
        edge_attr : [E, edge_feat_dim]

        Returns
        -------
        Tensor [embed_dim]
        """
        h = self.node_encoder(x)  # [N, hidden]

        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)

        mean_pool = h.mean(dim=0)  # [hidden]
        max_pool = h.max(dim=0).values  # [hidden]
        pooled = torch.cat([mean_pool, max_pool], dim=-1)  # [2*hidden]

        return self.readout(pooled)  # [embed_dim]

    def config_dict(self) -> dict:
        return {
            "node_feat_dim": self.node_feat_dim,
            "edge_feat_dim": self.edge_feat_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "embed_dim": self.embed_dim,
        }

```

#### Training table (hybrid Phase 1 driver)

| Hyperparameter | Value |
|---|---|
| Number of GNN layers | 2 |
| GNN type | Custom MPNN-style (EdgeMLP + sum agg + GRU) |
| Hidden dimension | 64 |
| Output embedding dimension | 64 |
| Activation | ReLU in MLPs; GRUCell nodes |
| Dropout | 0 in `MoleculeGNN`; head dropout **0.05** |
| Aggregation | Sum to destination (`index_add_`) |
| Readout pooling | concat(mean, max) |
| Trainable params (`MoleculeGNN` 27/6/64/2/64) | **85568** |
| Trainable params (`MultiDrugHybridGNNPBPK`, patient_feat_dim=5, n_euler_steps=384) | **94403** |
| Optimizer | Adam |
| LR head-only | 5e-3 |
| LR after unfreeze | 1e-3 (warfarin **2.5e-5**) |
| LR schedule | Step at unfreeze only |
| Batch size | 16 |
| Weight decay | 1e-5 |
| Gradient clipping | 5.0 |
| Seed | 42 (`experiments.config.SEED`); split also uses `split_rng_seed_for_drug` |
| Hardware (PK training) | **NOT AVAILABLE** in `metrics.json` |
| Hardware (pretrain metrics) | cpu in `gnn_pretrain_unsup_v1/metrics.json` |

#### Per-drug `metrics.json` (final epoch train/val composite loss)

| Drug | n_epochs | best_epoch | train.loss | val.loss |
|---|---:|---:|---:|---:|
| theophylline | 38 | 23 | 0.10444223862796206 | 0.13127138823037968 |
| warfarin | 48 | 2 | 0.11205486458347877 | 0.13166103982366623 |
| midazolam | 18 | 3 | 0.14240711776074022 | 0.14055189602077006 |
| caffeine | 17 | 2 | 0.06756887418741826 | 0.062455340556334706 |
| acetaminophen | 18 | 3 | 0.10794986177934333 | 0.10591744030825793 |
| digoxin | 18 | 3 | 0.11957546591002029 | 0.16258760504424571 |

**Average wall-clock minutes per drug:** **NOT AVAILABLE** (not stored in `metrics.json`; `phase1_train_multidrug.log` mixes multiple job types).


---

### SECTION 3 — Pretraining Details


#### 3.1 Supervised (`artifacts/models/gnn_pretrain_combined_v1/metrics.json`)

| Field | Value |
|---|---|
| n_train | 4509 |
| n_val | 795 |
| Source CSV task counts (`data/processed/adme_pretrain/adme_supervised.csv`) | esol **1123**, lipophilicity **4181** rows; **5304** unique SMILES |
| Final train loss / rmse | 0.8643872522248014 / 0.9297242873488778 |
| Final val loss / rmse | 1.4408720864609774 / 1.2003633060029943 |
| n_epochs | 60 |
| elapsed_seconds | 960.86 |
| Per-task ESOL vs Lipophilicity MAE/RMSE | **NOT AVAILABLE** in metrics.json |

#### 3.2 Unsupervised (`artifacts/models/gnn_pretrain_unsup_v1/metrics.json`)

| Field | Value |
|---|---|
| n_train / n_val | 9000 / 1000 |
| mask_rate | 0.15 |
| Final train loss / rmse | 0.002350799773671497 / 0.04444355801318418 |
| Final val loss / rmse | 0.0022020795417211046 / 0.04387197962482357 |
| n_epochs | 40 |
| elapsed_seconds | 1402.64 |
| Total ChEMBL corpus size | **NOT LOGGED** in metrics |

#### 3.3 Combined / PK initialisation

`gnn_pretrain_combined_v1/config.json`: `transfer_initialisation: true`; points to unsupervised `model_gnn.pt` and supervised CSV. PK fine-tuning loads encoder per `load_pretrained_gnn_into` (theophylline combined hybrid preferred).


---

### SECTION 4 — Paper Skeleton (verbatim)

# Paper skeleton — multi-drug DL–PBPK hybrid

Suggested article type: **Methods / modelling** with computational experiments (synthetic multi-drug PK + external drug transfer). Section titles are placeholders; tailor to target journal guidelines.

---

## Title

Deep graph–physiology hybrid for multi-drug oral pharmacokinetics: benchmarks, calibration, and safety-aware inference.

## Abstract (~200–250 words)

- **Background:** Limitations of purely data-driven vs purely mechanistic PK models across novel chemical matter.
- **Objective:** Joint molecular graph encoding + identifiable one-compartment ODE head; multi-drug training and zero-shot transfer.
- **Methods:** Dataset simulation; `MultiDrugHybridGNNPBPK`; baselines (PBPK-only, tabular ML, vanilla GNN); ablations; literature therapeutic bands; MC calibration; KernelSHAP on patient covariates.
- **Results:** Headline test metrics; realistic PBPK ablation staircase; external drug; calibration coverage highlights; SHAP patterns.
- **Conclusions:** When mechanistic structure helps; caveats (synthetic data, uncertainty model).

---

## 1. Introduction

- Clinical need for **structure-informed** PK early in drug development.
- Gaps in PBPK parameter identification vs black-box ML generalisation.
- Contributions (bullet list): (i) hybrid architecture + training protocol, (ii) rigorous baseline and ablation ladder including **fair** PBPK-only uncertainty, (iii) safety and uncertainty validation, (iv) explainability scope and limitations.

## 2. Related work

- PBPK and compartment models; ML in PK (non-compartmental, NN, GNN on SMILES/graphs).
- Hybrid physics-informed / grey-box models; transfer and multi-task learning in drug discovery.
- Uncertainty quantification and SHAP in healthcare ML (position KernelSHAP **conditional** explanations).

## 3. Methods

### 3.1 Data generation and drugs

- Six training drugs + one external drug; patients per drug; splits (e.g. 80/10/10); reproducible seeds (`experiments/config.py`).
- Simulation noise, IIV, and literature-based PK anchors (`download_pk_data`, `reference_pk`).

### 3.2 Molecular featurisation

- RDKit / graph construction; node and edge dimensions (align with `MoleculeGNN` config).

### 3.3 Hybrid model

- Equations from `mathematical_formulation.md`: GNN $\to$ MLP $\to$ $(\mathrm{CL}_{\mathrm{kg}}, \mathrm{V}_{\mathrm{kg}}, k_a)$; weight scaling; oral $F$; Euler simulator.

### 3.4 Training objective and schedules

- Loss (PK supervision weighting by drug if used); encoder freeze/unfreeze policy for difficult drugs (e.g. warfarin).

### 3.5 Baselines and ablations

- PBPK-only with **population uncertainty** (log-normal $\sigma=0.4$ on CL and V); MLP/RF/XGBoost; vanilla curve GNN; full hybrid (A1–A5 narrative).

### 3.6 Statistical comparison

- Paired tests vs hybrid; multiple testing caveat; report effect sizes and CIs where possible.

### 3.7 Safety and literature thresholds

- Therapeutic concentration bands from `REFERENCE_PK_DATA`; API enrichment (`risk_service`).

### 3.8 Uncertainty calibration

- MC scheme: $N=1000$, shared log-normal on $\mathrm{CL}, V$, $\sigma_{\mathrm{mc}}=0.3$; nominal vs empirical coverage (per drug + pooled).

### 3.9 Explainability

- KernelSHAP on predicted AUC; **graph fixed**; patient vector perturbed; `nsamples`, reference patients; limitation for molecular attribution.

## 4. Results

### 4.1 Predictive accuracy

- Per-drug test $R^2$, RMSE; table from `phase1_multidrug_metrics.csv` / Phase 2 benchmark CSVs (**final** authoritative tables).

### 4.2 Baselines and ablation staircase

- Corrected A1 vs A5; significance heatmap excerpt.

### 4.3 External validation

- Ibuprofen (or other) zero-shot metrics; figure: observed vs predicted.

### 4.4 Calibration

- Calibration plot + per-drug interpolated nominal 0.90 row (`phase_3_calibration_shap_diagnostic.md`).

### 4.5 SHAP

- Multi-panel summary figure; cross-drug feature patterns; flat patient SHAP for select drugs **in this explainer setup**.

## 5. Discussion

- Interpretation of hybrid inductive bias; when graph signal dominates.
- **Heterogeneous** MC calibration across drugs; caffeine/acetaminophen under-coverage at 90%.
- Explainability: patient-conditional SHAP ≠ global structure importance; future graph-level methods.
- Synthetic data limits; path to clinical cohorts and richer PBPK.

## 6. Conclusion

- Three bullet takeaways; reproducibility pointer.

## Declarations

- Code availability, data (synthetic generation scripts), ethics (if human data later).

## References

- Populate from `phase3_safety_thresholds.md` and standard PK / ML citations.

---

## Figure / table list (draft)

| ID | Content |
|----|--------|
| Fig. 1 | Model diagram (GNN + head + ODE). |
| Fig. 2 | Multi-drug predicted vs observed grid. |
| Fig. 3 | Baseline / ablation bar or staircase. |
| Fig. 4 | External drug validation. |
| Fig. 5 | Uncertainty calibration curve. |
| Fig. 6 | SHAP summary multi-drug. |
| Table 1 | Dataset and split summary. |
| Table 2 | Main test metrics + baselines. |
| Table 3 | Therapeutic windows (optional supplement). |


---

### SECTION 5 — Reproducibility Checklist (verbatim)

# Reproducibility checklist — DL–PBPK hybrid project

Root for commands: repository folder **`dl-pbpk-hybrid/`**. Use the project Python environment (e.g. `src/.venv` or documented venv) with pinned versions from project requirements if present.

---

## Environment

- [ ] **OS / Python:** note version; install dependencies (PyTorch, RDKit, scikit-learn, pandas, matplotlib, SHAP, etc.).
- [ ] **Hardware:** GPU optional but note device for training runs.
- [ ] **Determinism:** `experiments/config.py` — `SEED`, `seed_everything`; CUDA deterministic flags if required.

## Data pipeline

- [ ] `python -m experiments.data.download_pk_data` — regenerates per-drug PK CSVs under `experiments/data/processed/` (document any overrides).
- [ ] `python -m experiments.data.featurize_drugs` — molecular descriptors + `experiments/data/processed/graphs/*.pt`.

## Phase 1 — multi-drug hybrid

- [ ] `python -m experiments.training.train_multidrug_hybrid` — models under `artifacts/models/hybrid_gnn_pbpk_{drug}_v1/`.
- [ ] `python -m experiments.evaluation.evaluate_multidrug` — `experiments/results/phase1_multidrug_metrics.csv`, plots in `experiments/plots/`.

## Phase 2 — benchmarks & ablations

- [ ] Run benchmark / ablation training scripts as documented in `experiments/reports/phase_2_summary.md` (paths: `phase2_ablation_*`, baselines).
- [ ] Statistical tests / heatmaps: `experiments/statistics/significance_tests.py` with documented `--prediction-cache` / `--output-csv`.
- [ ] **Authoritative tables:** `experiments/results/phase2_benchmark_metrics_final.csv`, `phase2_ablation_summary_final.csv`, `phase2_statistical_tests_final.csv` (retain `*_corrected` / originals for audit).

## Phase 3 — safety, uncertainty, SHAP

- [ ] Literature thresholds: `python -m experiments.safety.safety_thresholds` → `experiments/results/phase3_safety_thresholds.md`.
- [ ] MC calibration: `python -m experiments.uncertainty.monte_carlo_calibration` → `phase3_uncertainty_calibration.csv`, `experiments/plots/uncertainty_calibration.*`.
- [ ] SHAP: `python -m experiments.explainability.shap_interpretation` → `phase3_shap_interpretation.md`, `shap_summary_multidrug.*`.
- [ ] **Post-approval diagnostic (read-only):** `experiments/reports/phase_3_calibration_shap_diagnostic.md` — per-drug nominal 0.90 interpolation; SHAP scope note.

## API (optional inference)

- [ ] Document environment variables / model paths for `api/` services; smoke-test `assess_risk(..., drug=...)` with literature block.

## Phase 4 — writing assets

- [ ] `paper/mathematical_formulation.md` — notation aligned with code.
- [ ] `paper/paper_skeleton.md` — section map and figure table.
- [ ] This checklist updated if script entry points change.

## Archival bundle (recommended)

- [ ] Commit **hashes** for code + **hashes or manifests** for large artifacts (`model.pt`, CSVs).
- [ ] One-page **commands log** (copy-paste order) attached as supplement or `README` section.
- [ ] Archive `experiments/results/*.csv`, `experiments/reports/phase_*_summary.md`, and key plots referenced in the manuscript.


---

### SECTION 6 — Full SHAP Interpretation

# Phase 3.3 — SHAP interpretation (patient covariates → predicted AUC)

KernelSHAP on the trained hybrid **per drug**, holding the molecular graph fixed. Background = random training patients. For each of **N** reference test patients, we explain the AUC response while holding that patient's dose, weight, F, and time grid fixed; only the z-scored patient tensor entries are coalition-perturbed. Approximate Shapley with ``nsamples=96``.

## Top 5 mean |SHAP| features per drug

### theophylline

| Rank | Feature | mean |SHAP| |
|---:|---|---:|
| 1 | dose_mgkg | 1.6078 |
| 2 | weight_kg | 1.0601 |
| 3 | sex | 0.4920 |
| 4 | age_years | 0.1314 |
| 5 | dose_mg | 0.0000 |

### warfarin

| Rank | Feature | mean |SHAP| |
|---:|---|---:|
| 1 | sex | 0.0192 |
| 2 | dose_mgkg | 0.0174 |
| 3 | weight_kg | 0.0164 |
| 4 | age_years | 0.0044 |
| 5 | dose_mg | 0.0000 |

### midazolam

| Rank | Feature | mean |SHAP| |
|---:|---|---:|
| 1 | sex | 0.0002 |
| 2 | age_years | 0.0000 |
| 3 | weight_kg | 0.0000 |
| 4 | dose_mgkg | 0.0000 |
| 5 | dose_mg | 0.0000 |

### caffeine

| Rank | Feature | mean |SHAP| |
|---:|---|---:|
| 1 | sex | 0.0103 |
| 2 | weight_kg | 0.0102 |
| 3 | dose_mgkg | 0.0083 |
| 4 | age_years | 0.0012 |
| 5 | dose_mg | 0.0000 |

### acetaminophen

| Rank | Feature | mean |SHAP| |
|---:|---|---:|
| 1 | log_dose_mg_per_kg | 0.0339 |
| 2 | weight_kg | 0.0248 |
| 3 | age_years | 0.0246 |
| 4 | sex | 0.0177 |
| 5 | dose_mg_per_kg | 0.0145 |

### digoxin

| Rank | Feature | mean |SHAP| |
|---:|---|---:|
| 1 | dose_mgkg | 0.0000 |
| 2 | sex | 0.0000 |
| 3 | age_years | 0.0000 |
| 4 | weight_kg | 0.0000 |
| 5 | dose_mg | 0.0000 |

## Pharmacological notes

- **weight_kg** (normalised channel): enters the ODE through absolute CL and V; even with dose fixed in mg, anthropometrics shift exposure per volume.
- **dose_mg** / **dose-normalised inputs**: primary driver of AUC for oral absorption.
- **age_years** / **sex**: captured here as coarse covariates in the fusion MLP head.

## Cross-drug patterns

Features appearing most often in this panel's top-5 (mean |SHAP|):

- **weight_kg** — in top-5 for 6 / 6 drugs.
- **sex** — in top-5 for 6 / 6 drugs.
- **age_years** — in top-5 for 6 / 6 drugs.
- **dose_mgkg** — in top-5 for 5 / 6 drugs.
- **dose_mg** — in top-5 for 5 / 6 drugs.
- **log_dose_mg_per_kg** — in top-5 for 1 / 6 drugs.
- **dose_mg_per_kg** — in top-5 for 1 / 6 drugs.



#### Wide table (from markdown; no SHAP pickle/CSV cache enumerated)

| Drug | Rank1 | \|SHAP\| | Rank2 | \|SHAP\| | Rank3 | \|SHAP\| | Rank4 | \|SHAP\| | Rank5 | \|SHAP\| |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| theophylline | dose_mgkg | 1.6078 | weight_kg | 1.0601 | sex | 0.4920 | age_years | 0.1314 | dose_mg | 0.0000 |
| warfarin | sex | 0.0192 | dose_mgkg | 0.0174 | weight_kg | 0.0164 | age_years | 0.0044 | dose_mg | 0.0000 |
| midazolam | sex | 0.0002 | age_years | 0.0000 | weight_kg | 0.0000 | dose_mgkg | 0.0000 | dose_mg | 0.0000 |
| caffeine | sex | 0.0103 | weight_kg | 0.0102 | dose_mgkg | 0.0083 | age_years | 0.0012 | dose_mg | 0.0000 |
| acetaminophen | log_dose_mg_per_kg | 0.0339 | weight_kg | 0.0248 | age_years | 0.0246 | sex | 0.0177 | dose_mg_per_kg | 0.0145 |
| digoxin | dose_mgkg | 0.0000 | sex | 0.0000 | age_years | 0.0000 | weight_kg | 0.0000 | dose_mg | 0.0000 |


---

### SECTION 7 — Statistical Significance Results

**Source:** `experiments/results/phase2_statistical_tests_final.csv`

| drug | baseline | test | statistic | p_value | mean_RMSE_diff_baseline_minus_DLPBPK | CI95_low_RMSE_diff | CI95_high_RMSE_diff | significance |
| theophylline | PBPK-only | per_patient_RMSE | 2.9782245574492627 | 0.007725064775441447 | 0.90266683462353 | 0.26829445553007025 | 1.53703921371699 | ** |
| theophylline | PBPK-only | per_timepoint_squared_error | 6.1376858201738465 | 3.1252755024812002e-09 |  |  |  | *** |
| theophylline | MLP | per_patient_RMSE | 0.15503998670028019 | 0.8784249847026734 | 0.005288288717568127 | -0.06610307120252501 | 0.07667964863766127 |  |
| theophylline | MLP | per_timepoint_squared_error | 0.43811224915485114 | 0.661670084674727 |  |  |  |  |
| theophylline | RandomForest | per_patient_RMSE | 1.4110148529145954 | 0.17440500564486805 | 0.0963870612707521 | -0.046588356285410484 | 0.2393624788269147 |  |
| theophylline | RandomForest | per_timepoint_squared_error | 3.668202400299414 | 0.00029645823584641807 |  |  |  | *** |
| theophylline | XGBoost | per_patient_RMSE | 1.4147563041934694 | 0.17332010940753603 | 0.0914158921017488 | -0.04382694834859452 | 0.22665873255209212 |  |
| theophylline | XGBoost | per_timepoint_squared_error | 1.7680686718979532 | 0.07822668365277949 |  |  |  |  |
| theophylline | VanillaGNN | per_patient_RMSE | 1.1464942742151487 | 0.265824818729503 | 0.06190279015858895 | -0.0511060854641103 | 0.1749116657812882 |  |
| theophylline | VanillaGNN | per_timepoint_squared_error | 4.349896173914319 | 1.9604531930987377e-05 |  |  |  | *** |
| warfarin | PBPK-only | per_patient_RMSE | 1.965948346110195 | 0.06408992048132979 | 0.054977715278410846 | -0.003553670229148785 | 0.11350910078597048 |  |
| warfarin | PBPK-only | per_timepoint_squared_error | 6.255558825740286 | 1.6286157888553026e-09 |  |  |  | *** |
| warfarin | MLP | per_patient_RMSE | 1.2936662930804186 | 0.2112872210707915 | 0.005328456767187057 | -0.0032924590333175798 | 0.013949372567691694 |  |
| warfarin | MLP | per_timepoint_squared_error | 0.6219577608506526 | 0.5345167495250364 |  |  |  |  |
| warfarin | RandomForest | per_patient_RMSE | -0.3757989214459619 | 0.7112293916592791 | -0.003774863291181224 | -0.024799084542655267 | 0.017249357960292817 |  |
| warfarin | RandomForest | per_timepoint_squared_error | -2.023162059756256 | 0.04408364268827253 |  |  |  | * |
| warfarin | XGBoost | per_patient_RMSE | 0.8764016016495704 | 0.3917578664542682 | 0.008256758233461056 | -0.011462048260662511 | 0.027975564727584622 |  |
| warfarin | XGBoost | per_timepoint_squared_error | 1.2368085368754438 | 0.2172784647871265 |  |  |  |  |
| warfarin | VanillaGNN | per_patient_RMSE | -1.1703887457824305 | 0.25631586689861857 | -0.005640719147318593 | -0.015728103276774368 | 0.004446664982137183 |  |
| warfarin | VanillaGNN | per_timepoint_squared_error | -5.994891511889855 | 6.803666455207315e-09 |  |  |  | *** |
| midazolam | PBPK-only | per_patient_RMSE | 2.189477015666379 | 0.04124619080195948 | 0.002142089426246422 | 9.436539729017679e-05 | 0.004189813455202668 | * |
| midazolam | PBPK-only | per_timepoint_squared_error | 4.118317026008761 | 5.137180502248368e-05 |  |  |  | *** |
| midazolam | MLP | per_patient_RMSE | 4.2725289395561825 | 0.0004113375811413532 | 0.0015528365620081823 | 0.0007921338674617908 | 0.002313539256554574 | *** |
| midazolam | MLP | per_timepoint_squared_error | 4.9662454081796445 | 1.2402147357756755e-06 |  |  |  | *** |
| midazolam | RandomForest | per_patient_RMSE | 1.6851758523311373 | 0.108309259318294 | 0.000519536215859584 | -0.00012573875376815675 | 0.0011648111854873247 |  |
| midazolam | RandomForest | per_timepoint_squared_error | 2.3385227793128704 | 0.020121354191656362 |  |  |  | * |
| midazolam | XGBoost | per_patient_RMSE | 1.9215754655922146 | 0.06978852862438449 | 0.0007438054023112468 | -6.63644956253375e-05 | 0.001553975300247831 |  |
| midazolam | XGBoost | per_timepoint_squared_error | 2.86731622307375 | 0.004480236042055641 |  |  |  | ** |
| midazolam | VanillaGNN | per_patient_RMSE | 0.9403152955275764 | 0.3588579533867531 | 0.00014523216903999193 | -0.00017803644598774496 | 0.0004685007840677288 |  |
| midazolam | VanillaGNN | per_timepoint_squared_error | -0.4920806387349585 | 0.6230791592533501 |  |  |  |  |
| caffeine | PBPK-only | per_patient_RMSE | 4.93006602953166 | 9.29558327677089e-05 | 0.573691622034414 | 0.33013497237943656 | 0.8172482716893915 | *** |
| caffeine | PBPK-only | per_timepoint_squared_error | 7.809776750997387 | 1.4278959813384804e-13 |  |  |  | *** |
| caffeine | MLP | per_patient_RMSE | 0.8602426456553701 | 0.40038030297594757 | 0.019655273827650272 | -0.028167234304241565 | 0.06747778195954211 |  |
| caffeine | MLP | per_timepoint_squared_error | 1.4919021091356766 | 0.1369418141898675 |  |  |  |  |
| caffeine | RandomForest | per_patient_RMSE | 0.4582955968529605 | 0.6519386454354106 | 0.013431582418227506 | -0.047910104657019566 | 0.07477326949347457 |  |
| caffeine | RandomForest | per_timepoint_squared_error | 1.5611765121694037 | 0.11970313976031959 |  |  |  |  |
| caffeine | XGBoost | per_patient_RMSE | 1.2394968885499973 | 0.23025390596864612 | 0.054414056162418775 | -0.03746994088343038 | 0.14629805320826794 |  |
| caffeine | XGBoost | per_timepoint_squared_error | 2.6747218725320026 | 0.007955192233182076 |  |  |  | ** |
| caffeine | VanillaGNN | per_patient_RMSE | 0.1518913073183666 | 0.8808739155686902 | 0.0020553301716586788 | -0.026266603222567863 | 0.03037726356588522 |  |
| caffeine | VanillaGNN | per_timepoint_squared_error | 0.5779451081754975 | 0.5638033108569341 |  |  |  |  |
| acetaminophen | PBPK-only | per_patient_RMSE | 3.3838021882937195 | 0.0031165483455355384 | 0.8149730916760072 | 0.31087793786516393 | 1.3190682454868505 | ** |
| acetaminophen | PBPK-only | per_timepoint_squared_error | 6.886530080118926 | 4.31359278157221e-11 |  |  |  | *** |
| acetaminophen | MLP | per_patient_RMSE | 2.182046805342707 | 0.041867197533953286 | 0.15578547708672083 | 0.00635570771989441 | 0.30521524645354725 | * |
| acetaminophen | MLP | per_timepoint_squared_error | 3.3688652140119646 | 0.0008696872475419887 |  |  |  | *** |
| acetaminophen | RandomForest | per_patient_RMSE | 1.8681831216504128 | 0.07723732106206797 | 0.16465588796263247 | -0.01981678509165205 | 0.349128561016917 |  |
| acetaminophen | RandomForest | per_timepoint_squared_error | 4.207314255569251 | 3.565018059557553e-05 |  |  |  | *** |
| acetaminophen | XGBoost | per_patient_RMSE | 2.35339379073218 | 0.029523995296203503 | 0.2588451467768442 | 0.02863755436102894 | 0.4890527391926595 | * |
| acetaminophen | XGBoost | per_timepoint_squared_error | 5.012239142235101 | 9.981443324560797e-07 |  |  |  | *** |
| acetaminophen | VanillaGNN | per_patient_RMSE | 0.8495000314804765 | 0.40618027421403974 | 0.020892027697570814 | -0.03058238654132414 | 0.07236644193646577 |  |
| acetaminophen | VanillaGNN | per_timepoint_squared_error | 0.7431088131238293 | 0.4580893064985092 |  |  |  |  |
| digoxin | PBPK-only | per_patient_RMSE | 2.591786729956946 | 0.017895985043796952 | 6.038602588113294e-05 | 1.1620669048924193e-05 | 0.00010915138271334169 | * |
| digoxin | PBPK-only | per_timepoint_squared_error | 8.375561856492137 | 3.502843390192495e-15 |  |  |  | *** |
| digoxin | MLP | per_patient_RMSE | 1.2809779870774098 | 0.21561690224044258 | 2.123339918678775e-05 | -1.3460417337098341e-05 | 5.592721571067384e-05 |  |
| digoxin | MLP | per_timepoint_squared_error | 3.1998018067924408 | 0.0015466206494969266 |  |  |  | ** |
| digoxin | RandomForest | per_patient_RMSE | 1.0414946102635405 | 0.310718097411698 | 9.947787535847824e-06 | -1.0043634786927426e-05 | 2.9939209858623073e-05 |  |
| digoxin | RandomForest | per_timepoint_squared_error | 4.107986409850181 | 5.35759289069798e-05 |  |  |  | *** |
| digoxin | XGBoost | per_patient_RMSE | 0.9451434809170962 | 0.3564509549404844 | 9.366627596703312e-06 | -1.1375807032969959e-05 | 3.0109062226376584e-05 |  |
| digoxin | XGBoost | per_timepoint_squared_error | 3.7887898490592677 | 0.00018829662953097245 |  |  |  | *** |
| digoxin | VanillaGNN | per_patient_RMSE | 0.977690145968881 | 0.34051288938611596 | 1.0525642570705566e-05 | -1.200749144872237e-05 | 3.30587765901335e-05 |  |
| digoxin | VanillaGNN | per_timepoint_squared_error | 3.1952351408358894 | 0.0015703494718869613 |  |  |  | ** |

#### Derived summary (per-patient RMSE only; $p<0.05$ & positive RMSE diff $\Rightarrow$ DL–PBPK better)

| Drug | Significantly better than | Not significantly different from | Significantly worse than |
|---|---|---|---|
| theophylline | PBPK-only | MLP, RandomForest, XGBoost, VanillaGNN | None |
| warfarin | None | all baselines listed | None |
| midazolam | PBPK-only, MLP | RandomForest, XGBoost, VanillaGNN | None |
| caffeine | PBPK-only | MLP, RandomForest, XGBoost, VanillaGNN | None |
| acetaminophen | PBPK-only, MLP, XGBoost | RandomForest, VanillaGNN | None |
| digoxin | PBPK-only | MLP, RandomForest, XGBoost, VanillaGNN | None |

_See also `per_timepoint_squared_error` rows in the CSV._


---

### SECTION 8 — Complete Phase Results Tables

#### 8.1 Phase 1

| drug | n_test_patients | obs_mean_mg_L | RMSE | RMSE_pct_of_mean | MAE | MAPE | R2 | Cmax_pct_err | AUC_pct_err |
| theophylline | 20 | 3.290126323699951 | 0.8765074014663696 | 26.640539457476397 | 0.6468181610107422 | 26.94527804851532 | 0.8273736650524297 | 18.848615884780884 | 20.540161430835724 |
| warfarin | 20 | 0.4822343587875366 | 0.11808127909898758 | 24.486284924954322 | 0.08638890832662582 | 21.80551290512085 | 0.7811127107375767 | 17.306172847747803 | 15.540257096290588 |
| midazolam | 20 | 0.011818972416222095 | 0.0032298199366778135 | 27.327417500115825 | 0.0019410564564168453 | 6295.359039306641 | 0.9199408972873047 | 11.60033568739891 | 16.412118077278137 |
| caffeine | 20 | 2.071187973022461 | 0.4449347257614136 | 21.48210261726803 | 0.3168658912181854 | 20.930033922195435 | 0.8707756610501393 | 15.052111446857452 | 12.800732254981995 |
| acetaminophen | 20 | 4.056467056274414 | 0.9437376856803894 | 23.265015408430678 | 0.6542503833770752 | 621209.27734375 | 0.9288735716177611 | 13.324865698814392 | 17.92859584093094 |
| digoxin | 20 | 0.000261943117948249 | 7.723291491856799e-05 | 29.484613006317062 | 5.7113498769467697e-05 | 24.400676786899567 | 0.7796332212669119 | 23.980000615119934 | 20.907697081565857 |#### 8.2 Benchmarks final

| drug | model | RMSE | RMSE_pct_of_mean | MAE | MAPE | R2 | Cmax_pct_err | AUC_pct_err | n_test_patients |
| theophylline | PBPK-only | 2.173440933227539 | 66.05949800683365 | 1.4466235637664795 | 52.831798791885376 | -0.06143047018174275 | 43.417707085609436 | 46.06767296791077 | 20 |
| theophylline | MLP | 0.8866853062178571 | 26.94988435706149 | 0.6409568365490835 | 341780.01166369114 | 0.8233413565861253 | 17.503088358794383 | 20.60929035192681 | 20 |
| theophylline | RandomForest | 0.9915939296555788 | 30.13847364559364 | 0.7188730307411515 | 458315.02783003263 | 0.7790654753466637 | 20.99836200756432 | 25.34384138564353 | 20 |
| theophylline | XGBoost | 0.9388155836376242 | 28.534330313380373 | 0.7139150870897888 | 581452.1404064171 | 0.801958390210048 | 19.527965181827287 | 25.008488812484003 | 20 |
| theophylline | VanillaGNN | 0.9749359770540512 | 29.632172376035577 | 0.6931437815341409 | 382875.97676400194 | 0.7864261571759765 | 20.335548790798043 | 22.171873304427507 | 20 |
| theophylline | DL-PBPK | 0.8765074402887424 | 26.640538631054923 | 0.64681821256426 | 26.945281294349748 | 0.8273736523931894 | 18.84861504666921 | 20.54016218415685 | 20 |
| warfarin | PBPK-only | 0.20388270914554596 | 42.27876040557078 | 0.13784557580947876 | 33.18609893321991 | 0.3474417564084259 | 28.88232171535492 | 28.9365291595459 | 20 |
| warfarin | MLP | 0.1191030198969012 | 24.69815984353764 | 0.0902387311220089 | 3787.4521481481993 | 0.777308303698375 | 17.432975474948726 | 16.23117259330165 | 20 |
| warfarin | RandomForest | 0.1099894461661459 | 22.808295917812654 | 0.0823918622664679 | 11059.291714164126 | 0.8100844579993745 | 17.346713634351527 | 15.618316043011344 | 20 |
| warfarin | XGBoost | 0.1229806045432997 | 25.502246973204866 | 0.0932261996845717 | 9613.41409523606 | 0.762572114906511 | 19.55369345827244 | 17.42413711124324 | 20 |
| warfarin | VanillaGNN | 0.1073969976127184 | 22.27070494141149 | 0.0807901505824072 | 6304.419091545227 | 0.8189315609588271 | 16.79246851777201 | 15.009351518514164 | 20 |
| warfarin | DL-PBPK | 0.1180812783363307 | 24.48628329831149 | 0.0863889126889402 | 21.80551231581305 | 0.7811126975493047 | 17.306173231794347 | 15.540257980529107 | 20 |
| midazolam | PBPK-only | 0.006041602697223425 | 51.11783397023469 | 0.003546807449311018 | 7631.464385986328 | 0.7198707733616045 | 26.940953731536865 | 31.212741136550903 | 20 |
| midazolam | MLP | 0.004970184301312 | 42.05259191564035 | 0.0031458405871557 | 31478.76998827368 | 0.8104171400141423 | 20.68121635788363 | 29.43684162526931 | 20 |
| midazolam | RandomForest | 0.0036309033199107 | 30.7209725717904 | 0.0023891133394939 | 30844.99230975522 | 0.8988226141386005 | 16.25667111587164 | 25.26898067111428 | 20 |
| midazolam | XGBoost | 0.0038725568320644 | 32.76559625483973 | 0.0025235463732459 | 22856.00526376589 | 0.8849067949873928 | 16.96653745918008 | 26.09994062255318 | 20 |
| midazolam | VanillaGNN | 0.0031860387332565 | 26.95698560749381 | 0.0020856623153511 | 43145.45420128746 | 0.9220966404846606 | 12.511110704986717 | 23.462696437965175 | 20 |
| midazolam | DL-PBPK | 0.0032298199077304 | 27.32741691388455 | 0.0019410565972324 | 6295.359176486226 | 0.9199409009893096 | 11.600336203169034 | 16.41211640839498 | 20 |
| caffeine | PBPK-only | 1.0959701538085938 | 50.85518266658686 | 0.7956755757331848 | 38.9231413602829 | 0.3051512980928728 | 38.62769901752472 | 30.25098145008087 | 20 |
| caffeine | MLP | 0.4633922501692267 | 22.37326064257411 | 0.3216287602867122 | 147134.96417321084 | 0.8598318837163296 | 16.744927423399194 | 12.436037875151449 | 20 |
| caffeine | RandomForest | 0.4750004404791904 | 22.933721175305458 | 0.3229018211893394 | 167430.6544800075 | 0.8527213722666349 | 15.539427902358588 | 12.295815730016663 | 20 |
| caffeine | XGBoost | 0.5291843040745139 | 25.54979794913399 | 0.3523163128943661 | 205674.20666678107 | 0.817204452606529 | 18.19145970898998 | 13.4199351558413 | 20 |
| caffeine | VanillaGNN | 0.4492243024974856 | 21.6892112526362 | 0.3133813965394578 | 163052.51586531696 | 0.8682719720115801 | 15.029592459641432 | 12.331873993176586 | 20 |
| caffeine | DL-PBPK | 0.4449347471790743 | 21.48210520124128 | 0.3168659003682618 | 20.930033802563862 | 0.8707756516276923 | 15.052110517369952 | 12.800731165844065 | 20 |
| acetaminophen | PBPK-only | 1.922978162765503 | 46.269452915268644 | 1.2898690700531006 | 4374220.703125 | 0.7226314526880315 | 31.20841383934021 | 28.066962957382202 | 20 |
| acetaminophen | MLP | 1.082894730649882 | 26.69551524709618 | 0.7675214185303859 | 1920045.653864032 | 0.906351479337466 | 17.615767093618828 | 19.154677985669952 | 20 |
| acetaminophen | RandomForest | 1.114018769402504 | 27.462784887953323 | 0.7803675295985838 | 3511197.899862622 | 0.9008909173005916 | 16.92195269489811 | 20.070796020591324 | 20 |
| acetaminophen | XGBoost | 1.2482781821934974 | 30.772547231221164 | 0.8358120301785396 | 4064340.387964296 | 0.8755625221314038 | 19.860087574542675 | 18.773159821674437 | 20 |
| acetaminophen | VanillaGNN | 0.9523695434299988 | 23.477809013114097 | 0.6757761113291892 | 2340688.8100243574 | 0.9275665056409916 | 13.68988835296995 | 18.689311052320537 | 20 |
| acetaminophen | DL-PBPK | 0.9437376562322104 | 23.265015774974053 | 0.6542504182895824 | 621209.2578006408 | 0.9288735703357398 | 13.324865539737475 | 17.928597330356183 | 20 |
| digoxin | PBPK-only | 0.00016334572865162045 | 62.359236351572946 | 0.00011407883721403778 | 48.14639687538147 | 0.014272591800598922 | 48.06875288486481 | 47.02695310115814 | 20 |
| digoxin | MLP | 0.0001308681079621 | 49.96050255627272 | 7.548717098845902e-05 | 40.21113649032571 | 0.3672838015044725 | 26.911222518187383 | 27.331139775876057 | 20 |
| digoxin | RandomForest | 9.913701876994223e-05 | 37.846770743469904 | 6.624184983157334e-05 | 50.72673846224173 | 0.6369110638950173 | 27.976073744214936 | 24.937064750000854 | 20 |
| digoxin | XGBoost | 9.818083989773788e-05 | 37.48173775160592 | 6.579678471100218e-05 | 56.692996988880175 | 0.643881289626856 | 27.540083756186235 | 24.7285068334116 | 20 |
| digoxin | VanillaGNN | 0.0001079542266785 | 41.21284781996366 | 6.665451875721172e-05 | 37.871987172114935 | 0.5694529603804062 | 26.63072245675841 | 23.4812009608326 | 20 |
| digoxin | DL-PBPK | 7.723291247241713e-05 | 29.484609971752445 | 5.71135000820041e-05 | 24.40067699822009 | 0.7796332173646252 | 23.979998545660827 | 20.90769904288977 | 20 |#### 8.3 Ablation by drug

_Note: `A1_PBPK_only` per-drug $R^2$ here may not match corrected panel mean in `phase2_ablation_summary_final.csv`._

| variant | drug | R2 | RMSE |
| A1_PBPK_only | theophylline | 0.833362991876562 | 0.8611678258867771 |
| A2_GNN_only | theophylline | 0.7864261571759765 | 0.9749359770540512 |
| A5_Full_DLPBPK | theophylline | 0.8273736523931894 | 0.8765074402887424 |
| A3_hybrid_no_transfer | theophylline | 0.7998110344166808 | 0.9438916444778442 |
| A4_hybrid_encoder_frozen | theophylline | 0.8385979950445988 | 0.8475328087806702 |
| A1_PBPK_only | warfarin | 0.7762281019436289 | 0.1193915345893949 |
| A2_GNN_only | warfarin | 0.8189315609588271 | 0.1073969976127184 |
| A5_Full_DLPBPK | warfarin | 0.7811126975493047 | 0.1180812783363307 |
| A3_hybrid_no_transfer | warfarin | 0.686710352244255 | 0.14126800000667572 |
| A4_hybrid_encoder_frozen | warfarin | 0.7867204865797794 | 0.11655887216329575 |
| A1_PBPK_only | midazolam | 0.9219537016609064 | 0.0031889603012398 |
| A2_GNN_only | midazolam | 0.9220966404846606 | 0.0031860387332565 |
| A5_Full_DLPBPK | midazolam | 0.9199409009893096 | 0.0032298199077304 |
| A3_hybrid_no_transfer | midazolam | 0.9197239981250727 | 0.0032341922633349895 |
| A4_hybrid_encoder_frozen | midazolam | 0.9226326873656117 | 0.003175058402121067 |
| A1_PBPK_only | caffeine | 0.8701896219769865 | 0.4459424907195867 |
| A2_GNN_only | caffeine | 0.8682719720115801 | 0.4492243024974856 |
| A5_Full_DLPBPK | caffeine | 0.8707756516276923 | 0.4449347471790743 |
| A3_hybrid_no_transfer | caffeine | 0.7410557594497965 | 0.6690471768379211 |
| A4_hybrid_encoder_frozen | caffeine | 0.7475718636217643 | 0.6605755686759949 |
| A1_PBPK_only | acetaminophen | 0.9276165170118696 | 0.952040707087993 |
| A2_GNN_only | acetaminophen | 0.9275665056409916 | 0.9523695434299988 |
| A5_Full_DLPBPK | acetaminophen | 0.9288735703357398 | 0.9437376562322104 |
| A3_hybrid_no_transfer | acetaminophen | 0.8674608861779511 | 1.3292839527130127 |
| A4_hybrid_encoder_frozen | acetaminophen | 0.8736479007283724 | 1.2978873252868652 |
| A1_PBPK_only | digoxin | 0.7887428011956212 | 7.56197270527801e-05 |
| A2_GNN_only | digoxin | 0.5694529603804062 | 0.0001079542266785 |
| A5_Full_DLPBPK | digoxin | 0.7796332173646252 | 7.723291247241713e-05 |
| A3_hybrid_no_transfer | digoxin | 0.7654332887064776 | 7.968242425704375e-05 |
| A4_hybrid_encoder_frozen | digoxin | 0.7699862248018736 | 7.890531560406089e-05 |#### 8.4 Ablation summary final

| variant | mean_R2_6drugs | mean_RMSE_6drugs |
| A1_PBPK_only | 0.3413229003616318 | 0.9004128178955094 |
| A2_GNN_only | 0.8154576327754071 | 0.4145368022590315 |
| A3_hybrid_no_transfer | 0.7966992198533723 | 0.5144674414538409 |
| A4_hybrid_encoder_frozen | 0.8231928596903333 | 0.4876347564374252 |
| A5_Full_DLPBPK | 0.8512849483766435 | 0.39776136247609345 |#### 8.5 External validation

| drug | split | n_patients | RMSE | RMSE_pct_of_mean | MAE | R2 | Cmax_pct_err | AUC_pct_err | encoder | fine_tuned_on_ibuprofen |
| ibuprofen | test | 20 | 4.402677059173584 | 34.992308052635366 | 3.0976336002349854 | 0.8363323328511169 | 21.574966609477997 | 28.335583209991455 | pretrained_frozen | False |#### 8.6 Phase 3 uncertainty

| scope | nominal_interval_frac | empirical_coverage | n_mc | mc_log_sd | n_concentration_points |
| theophylline | 0.5 | 0.49615384615384617 | 1000 | 0.3 | 260 |
| theophylline | 0.535 | 0.5269230769230769 | 1000 | 0.3 | 260 |
| theophylline | 0.57 | 0.5692307692307692 | 1000 | 0.3 | 260 |
| theophylline | 0.605 | 0.6153846153846154 | 1000 | 0.3 | 260 |
| theophylline | 0.64 | 0.6307692307692307 | 1000 | 0.3 | 260 |
| theophylline | 0.675 | 0.6461538461538462 | 1000 | 0.3 | 260 |
| theophylline | 0.71 | 0.7115384615384616 | 1000 | 0.3 | 260 |
| theophylline | 0.745 | 0.7384615384615385 | 1000 | 0.3 | 260 |
| theophylline | 0.78 | 0.7807692307692308 | 1000 | 0.3 | 260 |
| theophylline | 0.815 | 0.8076923076923077 | 1000 | 0.3 | 260 |
| theophylline | 0.85 | 0.85 | 1000 | 0.3 | 260 |
| theophylline | 0.885 | 0.8961538461538462 | 1000 | 0.3 | 260 |
| theophylline | 0.9199999999999999 | 0.9230769230769231 | 1000 | 0.3 | 260 |
| theophylline | 0.955 | 0.9346153846153846 | 1000 | 0.3 | 260 |
| theophylline | 0.99 | 0.9615384615384616 | 1000 | 0.3 | 260 |
| warfarin | 0.5 | 0.5115384615384615 | 1000 | 0.3 | 260 |
| warfarin | 0.535 | 0.5346153846153846 | 1000 | 0.3 | 260 |
| warfarin | 0.57 | 0.5423076923076923 | 1000 | 0.3 | 260 |
| warfarin | 0.605 | 0.5615384615384615 | 1000 | 0.3 | 260 |
| warfarin | 0.64 | 0.6153846153846154 | 1000 | 0.3 | 260 |

_Table truncated: header + first 20 data rows of 105 total data rows._

_Total CSV lines (including header): **106**._


---

### SECTION 9 — Figure Inventory

Verified PNG/PDF under `experiments/plots/`.

| Path | Size (KB) | Description |
|---|---:|---|
| experiments/plots/ablation_study.pdf | 16.7 | Ablation study (PDF) |
| experiments/plots/ablation_study.png | 35.0 | Ablation study (PNG) |
| experiments/plots/observed_vs_predicted_grid.pdf | 46.3 | Observed vs predicted grid (PDF) |
| experiments/plots/observed_vs_predicted_grid.png | 269.3 | Observed vs predicted grid (PNG) |
| experiments/plots/phase2_baseline_r2_bar.pdf | 15.1 | Baseline R² bar (PDF) |
| experiments/plots/phase2_baseline_r2_bar.png | 33.0 | Baseline R² bar (PNG) |
| experiments/plots/phase2_external_ibuprofen_scatter.pdf | 18.9 | Ibuprofen external scatter (PDF) |
| experiments/plots/phase2_external_ibuprofen_scatter.png | 58.7 | Ibuprofen external scatter (PNG) |
| experiments/plots/phase2_training_A3_acetaminophen.pdf | 16.1 | A3 training curves acetaminophen (PDF) |
| experiments/plots/phase2_training_A3_acetaminophen.png | 75.3 | A3 training curves acetaminophen (PNG) |
| experiments/plots/phase2_training_A3_caffeine.pdf | 14.6 | A3 training curves caffeine (PDF) |
| experiments/plots/phase2_training_A3_caffeine.png | 57.7 | A3 training curves caffeine (PNG) |
| experiments/plots/phase2_training_A3_digoxin.pdf | 15.6 | A3 training curves digoxin (PDF) |
| experiments/plots/phase2_training_A3_digoxin.png | 43.3 | A3 training curves digoxin (PNG) |
| experiments/plots/phase2_training_A3_midazolam.pdf | 15.3 | A3 training curves midazolam (PDF) |
| experiments/plots/phase2_training_A3_midazolam.png | 60.7 | A3 training curves midazolam (PNG) |
| experiments/plots/phase2_training_A3_theophylline.pdf | 14.8 | A3 training curves theophylline (PDF) |
| experiments/plots/phase2_training_A3_theophylline.png | 55.0 | A3 training curves theophylline (PNG) |
| experiments/plots/phase2_training_A3_warfarin.pdf | 15.2 | A3 training curves warfarin (PDF) |
| experiments/plots/phase2_training_A3_warfarin.png | 59.4 | A3 training curves warfarin (PNG) |
| experiments/plots/phase2_training_A4_acetaminophen.pdf | 14.9 | A4 training curves acetaminophen (PDF) |
| experiments/plots/phase2_training_A4_acetaminophen.png | 98.4 | A4 training curves acetaminophen (PNG) |
| experiments/plots/phase2_training_A4_caffeine.pdf | 15.1 | A4 training curves caffeine (PDF) |
| experiments/plots/phase2_training_A4_caffeine.png | 71.4 | A4 training curves caffeine (PNG) |
| experiments/plots/phase2_training_A4_digoxin.pdf | 15.9 | A4 training curves digoxin (PDF) |
| experiments/plots/phase2_training_A4_digoxin.png | 54.5 | A4 training curves digoxin (PNG) |
| experiments/plots/phase2_training_A4_midazolam.pdf | 15.7 | A4 training curves midazolam (PDF) |
| experiments/plots/phase2_training_A4_midazolam.png | 78.4 | A4 training curves midazolam (PNG) |
| experiments/plots/phase2_training_A4_theophylline.pdf | 15.4 | A4 training curves theophylline (PDF) |
| experiments/plots/phase2_training_A4_theophylline.png | 71.6 | A4 training curves theophylline (PNG) |
| experiments/plots/phase2_training_A4_warfarin.pdf | 16.5 | A4 training curves warfarin (PDF) |
| experiments/plots/phase2_training_A4_warfarin.png | 73.2 | A4 training curves warfarin (PNG) |
| experiments/plots/pk_curves_grid.pdf | 24.3 | PK curves grid (PDF) |
| experiments/plots/pk_curves_grid.png | 373.6 | PK curves grid (PNG) |
| experiments/plots/r2_summary_bar.pdf | 14.3 | R² summary bar (PDF) |
| experiments/plots/r2_summary_bar.png | 31.8 | R² summary bar (PNG) |
| experiments/plots/shap_summary_multidrug.pdf | 204.1 | SHAP multi-drug summary (PDF) |
| experiments/plots/shap_summary_multidrug.png | 153.2 | SHAP multi-drug summary (PNG) |
| experiments/plots/significance_heatmap.pdf | 23.8 | Significance heatmap (PDF) |
| experiments/plots/significance_heatmap.png | 75.5 | Significance heatmap (PNG) |
| experiments/plots/significance_heatmap_corrected.pdf | 23.8 | Significance heatmap corrected (PDF) |
| experiments/plots/significance_heatmap_corrected.png | 76.3 | Significance heatmap corrected (PNG) |
| experiments/plots/training_curves_acetaminophen.pdf | 14.6 | Phase 1 training curves acetaminophen (PDF) |
| experiments/plots/training_curves_acetaminophen.png | 49.8 | Phase 1 training curves acetaminophen (PNG) |
| experiments/plots/training_curves_caffeine.pdf | 14.5 | Phase 1 training curves caffeine (PDF) |
| experiments/plots/training_curves_caffeine.png | 42.1 | Phase 1 training curves caffeine (PNG) |
| experiments/plots/training_curves_digoxin.pdf | 15.1 | Phase 1 training curves digoxin (PDF) |
| experiments/plots/training_curves_digoxin.png | 44.9 | Phase 1 training curves digoxin (PNG) |
| experiments/plots/training_curves_midazolam.pdf | 14.5 | Phase 1 training curves midazolam (PDF) |
| experiments/plots/training_curves_midazolam.png | 52.8 | Phase 1 training curves midazolam (PNG) |
| experiments/plots/training_curves_theophylline.pdf | 14.6 | Phase 1 training curves theophylline (PDF) |
| experiments/plots/training_curves_theophylline.png | 50.4 | Phase 1 training curves theophylline (PNG) |
| experiments/plots/training_curves_warfarin.pdf | 14.7 | Phase 1 training curves warfarin (PDF) |
| experiments/plots/training_curves_warfarin.png | 57.2 | Phase 1 training curves warfarin (PNG) |
| experiments/plots/uncertainty_calibration.pdf | 18.6 | Uncertainty calibration (PDF) |
| experiments/plots/uncertainty_calibration.png | 111.8 | Uncertainty calibration (PNG) |

#### Missing Figures

None of the items listed in phase reports were missing at inventory time; per-drug scatter panels appear in **`observed_vs_predicted_grid`**.


---

### SECTION 10 — Codebase Inventory & File Structure

#### Tree (depth ≤ 3, selected)

```
dl-pbpk-hybrid/
├── api/
├── artifacts/models/
├── data/processed/adme_pretrain/
├── experiments/
│   ├── data/
│   ├── training/
│   ├── evaluation/
│   ├── baselines/
│   ├── ablation/
│   ├── statistics/
│   ├── explainability/
│   ├── uncertainty/
│   ├── safety/
│   ├── phase2/
│   ├── results/
│   ├── plots/
│   ├── logs/
│   └── reports/
├── paper/
└── src/
```

**`frontend/`:** **NOT FOUND**

#### Line counts — `src/**/*.py` (excluding `src/.venv`, `__pycache__`)

| Path | Lines |
|---|---:|
| src/train.py | 12 |
| src/datasets/theoph_loader.py | 78 |
| src/models/gnn/molecule_gnn.py | 135 |
| src/models/hybrid_dl_pk.py | 114 |
| src/models/hybrid_gnn_pbpk.py | 140 |
| src/models/ode/pk_1cpt_torch.py | 95 |
| src/molecules/rdkit_graph.py | 133 |
| src/tests/test_molecule_gnn.py | 53 |
| src/tests/test_rdkit_graph.py | 53 |
| src/training/finetune_gnn_pbpk_theoph.py | 391 |
| src/training/plot_predictions.py | 131 |
| src/training/pretrain_gnn_adme.py | 602 |
| src/training/pretrain_gnn_unsupervised.py | 542 |
| src/training/train_hybrid_theoph.py | 364 |
| (+ `__init__.py` stubs) | |

#### Line counts — `experiments/**/*.py` (excluding `__pycache__`)

| Path | Lines |
|---|---:|
| experiments/config.py | 130 |
| experiments/reference_pk.py | 136 |
| experiments/data/download_pk_data.py | 400 |
| experiments/data/featurize_drugs.py | 142 |
| experiments/models/hybrid_multidrug.py | 184 |
| experiments/training/multidrug_utils.py | 350 |
| experiments/training/train_multidrug_hybrid.py | 546 |
| experiments/evaluation/evaluate_multidrug.py | 280 |
| experiments/baselines/train_baselines.py | 409 |
| experiments/baselines/correct_pbpk_realistic.py | 179 |
| experiments/ablation/ablation_study.py | 183 |
| experiments/statistics/significance_tests.py | 169 |
| experiments/phase2/external_ibuprofen.py | 126 |
| experiments/phase2/utils.py | 167 |
| experiments/explainability/shap_interpretation.py | 217 |
| experiments/uncertainty/monte_carlo_calibration.py | 221 |
| experiments/safety/literature_bounds.py | 19 |
| experiments/safety/safety_thresholds.py | 67 |
| (+ `__init__.py` stubs) | |

#### Line counts — `api/**/*.py` (excluding `__pycache__`)

| Path | Lines |
|---|---:|
| api/app/main.py | 736 |
| api/app/schemas.py | 340 |
| api/app/config.py | 13 |
| api/app/services/hybrid_infer_service.py | 425 |
| api/app/services/pbpk_service.py | 323 |
| api/app/services/report_service.py | 306 |
| api/app/services/xai_service.py | 328 |
| api/app/services/risk_service.py | 180 |
| api/app/services/population_adapter.py | 206 |
| api/app/services/rdkit_graph.py | 107 |
| api/app/services/_gnn_inline.py | 114 |
| api/tests/test_main.py | 148 |
| api/tests/test_pbpk.py | 162 |
| api/tests/test_population.py | 157 |
| api/tests/test_gnn_api.py | 89 |
| api/tests/test_explain_and_report.py | 101 |
| api/tests/test_smiles_graph.py | 65 |
| (+ `__init__.py` stubs) | |

#### Frontend entry points

**NOT FOUND**

#### Reproduction commands (in order; from `dl-pbpk-hybrid/` root)

1. `python -m experiments.data.download_pk_data`
2. `python -m experiments.data.featurize_drugs`
3. _(Rebuild encoders if needed)_ `python src/training/pretrain_gnn_unsupervised.py`
4. _(Rebuild encoders if needed)_ `python src/training/pretrain_gnn_adme.py`
5. _(Produce transfer checkpoint)_ scripts under `src/training/` as used to create `artifacts/models/hybrid_gnn_pbpk_theoph_combined_v1/`
6. `python -m experiments.training.train_multidrug_hybrid`
7. `python -m experiments.evaluation.evaluate_multidrug`
8. `python -m experiments.baselines.train_baselines`
9. `python -m experiments.ablation.ablation_study`
10. `python -m experiments.statistics.significance_tests`
11. `python -m experiments.phase2.external_ibuprofen`
12. `python -m experiments.safety.safety_thresholds`
13. `python -m experiments.uncertainty.monte_carlo_calibration`
14. `python -m experiments.explainability.shap_interpretation`
---

**END OF REPORT**
