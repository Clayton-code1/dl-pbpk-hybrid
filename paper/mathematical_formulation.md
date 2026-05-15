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
