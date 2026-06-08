"""
Real Warfarin validation — forward-only inference on O'Reilly/Holford real human PK data
(available via nlmixr2data R package, warfarin.rda).

SAFETY CONTRACT
- Loads artifacts/models/hybrid_gnn_pbpk_warfarin_v1/model.pt
- Immediately calls model.eval() and runs entirely under torch.no_grad()
- No optimizer, no gradient, no weight update of any kind
- All outputs written to experiments/warfarin_validation/ only

CAVEATS (explicitly disclosed — see warfarin_results.md for full discussion)
1. DOSE EXTRAPOLATION: model trained on ~5 mg warfarin; real data uses ~100 mg doses
   (1.5 mg/kg, a 1960s-era single-dose characterization study). z-score for dose_mg is
   ~+95 relative to the training scaler. The ODE scales concentration linearly with dose,
   but the neural network's PK-parameter predictions are in extreme extrapolation territory.
2. ABSORPTION PHASE: 13/32 subjects have absorption-phase data (first obs <=6h).
   The remaining 19 subjects have first concentration at t>=24h (trough only).
   Results are reported three ways: all 32 / absorption-present (13) / trough-only (19).

COVARIATE MAPPING — ALL REAL, NO IMPUTATION
- weight_kg    : wt column (real individual values)
- dose_mg      : amt at dosing event (real individual values)
- dose_mgkg    : dose_mg / weight_kg (derived)
- age_years    : age column (real individual values; 21-63 yr)
- sex          : sex column, female->0, male->1 (real individual values)

Run from project root:
    python experiments/warfarin_validation/run_warfarin_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.models.hybrid_multidrug import MultiDrugHybridConfig, MultiDrugHybridGNNPBPK
from experiments.training.multidrug_utils import StandardScaler

OUT_DIR    = _HERE.parent
CKPT_DIR   = _ROOT / "artifacts/models/hybrid_gnn_pbpk_warfarin_v1"
GRAPH_PATH = _ROOT / "experiments/data/processed/graphs/warfarin.pt"
DATA_PATH  = OUT_DIR / "raw/warfarin.rda"

# Absorption-phase group: subjects whose first post-dose observation is <=6h
ABS_SUBJS  = {1, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16}


# ── 1. Load model (eval + no_grad only) ───────────────────────────────────────

cfg_dict = json.loads((CKPT_DIR / "config.json").read_text())
cfg = MultiDrugHybridConfig(
    node_feat_dim    = cfg_dict["node_feat_dim"],
    edge_feat_dim    = cfg_dict["edge_feat_dim"],
    gnn_hidden       = cfg_dict["gnn_hidden"],
    gnn_layers       = cfg_dict["gnn_layers"],
    gnn_embed_dim    = cfg_dict["gnn_embed_dim"],
    patient_feat_dim = cfg_dict["patient_feat_dim"],
    head_hidden      = cfg_dict["head_hidden"],
    head_dropout     = cfg_dict["head_dropout"],
    n_euler_steps    = cfg_dict["n_euler_steps"],
)
model = MultiDrugHybridGNNPBPK(cfg)
state = torch.load(CKPT_DIR / "model.pt", map_location="cpu")
model.load_state_dict(state)
model.eval()
print(f"Loaded checkpoint: {CKPT_DIR / 'model.pt'}")
print(f"n_euler_steps={cfg.n_euler_steps}  patient_feat_dim={cfg.patient_feat_dim}")


# ── 2. Scaler ─────────────────────────────────────────────────────────────────

scaler = StandardScaler.from_dict(json.loads((CKPT_DIR / "scaler.json").read_text()))
FEAT = scaler.feature_names   # [weight_kg, dose_mg, dose_mgkg, age_years, sex]
print(f"Feature order: {FEAT}")
print(f"Training scaler means: {dict(zip(FEAT, scaler.mean_))}")
print(f"Training scaler stds:  {dict(zip(FEAT, scaler.std_))}")


# ── 3. Drug graph ──────────────────────────────────────────────────────────────

blob         = torch.load(GRAPH_PATH, map_location="cpu")
graph_x      = blob["x"]
graph_edge_i = blob["edge_index"]
graph_edge_a = blob["edge_attr"]
print(f"Warfarin graph loaded: {graph_x.shape[0]} atoms")


# ── 4. Load real warfarin data ─────────────────────────────────────────────────

import pyreadr
df_raw = pyreadr.read_r(str(DATA_PATH))["warfarin"]
cp_all = df_raw[df_raw["dvid"] == "cp"].copy()

# Dose per subject (single oral dose — the single evid=1 row per subject)
dose_map = df_raw[df_raw["amt"] > 0].groupby("id")["amt"].sum().to_dict()

print(f"\nReal warfarin data: {cp_all['id'].nunique()} subjects, {len(cp_all)} PK rows")


# ── 5. Inference ──────────────────────────────────────────────────────────────

def _feat_tensor(weight_kg: float, dose_mg: float, dose_mgkg: float,
                 age_years: float, sex: float) -> torch.Tensor:
    raw = np.array([[weight_kg, dose_mg, dose_mgkg, age_years, sex]])
    return torch.tensor(scaler.transform(raw)[0], dtype=torch.float32)


flat_rows:   list[dict] = []
per_subject: dict[int, dict] = {}

with torch.no_grad():
    for sid, grp in cp_all.groupby("id"):
        sid = int(sid)
        wt  = float(grp["wt"].iloc[0])
        age = float(grp["age"].iloc[0])
        sex_raw = str(grp["sex"].iloc[0])
        sex_enc = 1.0 if sex_raw == "male" else 0.0

        dose_mg   = float(dose_map[sid])
        dose_mgkg = dose_mg / wt

        times_arr = grp["time"].to_numpy(dtype=float)
        conc_obs  = grp["dv"].to_numpy(dtype=float)

        times_t = torch.tensor(times_arr, dtype=torch.float32)
        feat    = _feat_tensor(wt, dose_mg, dose_mgkg, age, sex_enc)

        conc_pred_t, CL, V, ka = model(
            graph_x, graph_edge_i, graph_edge_a,
            feat, times_t,
            torch.tensor(dose_mg, dtype=torch.float32),
            torch.tensor(wt,      dtype=torch.float32),
        )
        conc_pred = conc_pred_t.cpu().numpy()

        is_abs = sid in ABS_SUBJS
        per_subject[sid] = {
            "times":      times_arr.tolist(),
            "conc_obs":   conc_obs.tolist(),
            "conc_pred":  conc_pred.tolist(),
            "weight_kg":  wt,
            "age_years":  age,
            "sex":        sex_raw,
            "dose_mg":    dose_mg,
            "dose_mgkg":  dose_mgkg,
            "CL_Lh":      float(CL),
            "V_L":        float(V),
            "ka_h":       float(ka),
            "abs_group":  is_abs,
        }

        for t, obs, pred in zip(times_arr, conc_obs, conc_pred):
            flat_rows.append({
                "subject_id":    sid,
                "time_h":        float(t),
                "conc_obs_mgL":  float(obs),
                "conc_pred_mgL": float(pred),
                "abs_group":     is_abs,
            })

df_preds = pd.DataFrame(flat_rows)
df_preds.to_csv(OUT_DIR / "warfarin_predictions.csv", index=False)
print(f"Saved warfarin_predictions.csv ({len(df_preds)} rows)")


# ── 6. Metrics ────────────────────────────────────────────────────────────────

_eps = 1e-8

def _r2_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    err    = y_pred - y_true
    rmse   = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2     = 1.0 - ss_res / (ss_tot + _eps)
    return r2, rmse


y_obs  = df_preds["conc_obs_mgL"].to_numpy()
y_pred = df_preds["conc_pred_mgL"].to_numpy()

# Naive baseline
r2_naive, rmse_naive = _r2_rmse(y_obs, np.full_like(y_obs, y_obs.mean()))

# (a) All 32 subjects
r2_all,  rmse_all  = _r2_rmse(y_obs, y_pred)

# (b) Absorption-present group (13 subjects)
mask_abs  = df_preds["abs_group"].to_numpy()
r2_abs,  rmse_abs  = _r2_rmse(y_obs[mask_abs], y_pred[mask_abs])

# (c) Trough-only group (19 subjects)
mask_tro  = ~mask_abs
r2_tro,  rmse_tro  = _r2_rmse(y_obs[mask_tro], y_pred[mask_tro])

# Per-subject
per_subj_r2:   dict[int, float] = {}
per_subj_rmse: dict[int, float] = {}
for sid, grp in df_preds.groupby("subject_id"):
    r2_s, rmse_s = _r2_rmse(
        grp["conc_obs_mgL"].to_numpy(), grp["conc_pred_mgL"].to_numpy()
    )
    per_subj_r2[int(sid)]   = r2_s
    per_subj_rmse[int(sid)] = rmse_s


# ── 7. Plots ──────────────────────────────────────────────────────────────────

sid_list_abs = sorted([s for s in per_subject if per_subject[s]["abs_group"]])
sid_list_tro = sorted([s for s in per_subject if not per_subject[s]["abs_group"]])
sid_list_all = sorted(per_subject.keys())

# 7a  Pred vs Obs scatter — colour by group
fig, ax = plt.subplots(figsize=(7, 7))
COLORS_ABS = plt.cm.tab20(np.linspace(0, 0.5, len(sid_list_abs)))
COLORS_TRO = plt.cm.tab20(np.linspace(0.55, 1.0, len(sid_list_tro)))

for i, sid in enumerate(sid_list_abs):
    info = per_subject[sid]
    ax.scatter(info["conc_obs"], info["conc_pred"],
               color=COLORS_ABS[i], marker="o", s=45, alpha=0.9, zorder=4,
               label=f"S{sid} (abs)" if i < 5 else "_")
for i, sid in enumerate(sid_list_tro):
    info = per_subject[sid]
    ax.scatter(info["conc_obs"], info["conc_pred"],
               color=COLORS_TRO[i], marker="s", s=35, alpha=0.65, zorder=3,
               label=f"S{sid} (tro)" if i < 3 else "_")

lim = max(y_obs.max(), y_pred.max()) * 1.07
ax.plot([0, lim], [0, lim], "k--", lw=1.5, label="identity")
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("Observed concentration (mg/L)", fontsize=12)
ax.set_ylabel("Predicted concentration (mg/L)", fontsize=12)
ax.set_title(
    "Real Warfarin validation — Predicted vs Observed\n"
    f"All 32 subjects: R²={r2_all:+.3f}  RMSE={rmse_all:.3f} mg/L  (n={len(y_obs)})\n"
    f"Absorption subgroup (n=13): R²={r2_abs:+.3f}  RMSE={rmse_abs:.3f} mg/L\n"
    f"Trough-only (n=19): R²={r2_tro:+.3f}  RMSE={rmse_tro:.3f} mg/L\n"
    f"Naive-mean baseline R²={r2_naive:+.3f}   ●=absorption  ■=trough-only",
    fontsize=8.5,
)
ax.legend(fontsize=6, ncol=3, loc="upper left")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "pred_vs_obs.png", dpi=150)
plt.close(fig)
print("Saved pred_vs_obs.png")


# 7b  Concentration-time curves — absorption group (13 panels)
nabs = len(sid_list_abs)
ncols_a = 4; nrows_a = (nabs + ncols_a - 1) // ncols_a
fig_a, axes_a = plt.subplots(nrows_a, ncols_a, figsize=(16, 4 * nrows_a))
for ax in axes_a.ravel()[nabs:]:
    ax.axis("off")

for i, sid in enumerate(sid_list_abs):
    ax   = axes_a.ravel()[i]
    info = per_subject[sid]
    t    = np.array(info["times"])

    ax.scatter(t, info["conc_obs"], color="tab:blue", s=30, zorder=5,
               label="Observed", clip_on=False)
    ax.plot(t, info["conc_pred"], color="tab:orange", lw=2, label="Predicted")
    ax.set_title(
        f"S{sid} (abs)  R²={per_subj_r2[sid]:+.2f}  RMSE={per_subj_rmse[sid]:.2f} mg/L\n"
        f"wt={info['weight_kg']:.0f}kg  dose={info['dose_mg']:.0f}mg  "
        f"age={info['age_years']:.0f}yr  {info['sex']}\n"
        f"CL={info['CL_Lh']:.2f}L/h  Vd={info['V_L']:.1f}L  ka={info['ka_h']:.2f}/h",
        fontsize=7,
    )
    ax.set_xlabel("Time (h)", fontsize=8)
    ax.set_ylabel("Conc (mg/L)", fontsize=8)
    ax.grid(alpha=0.3)
    if i == 0:
        ax.legend(fontsize=7)

fig_a.suptitle(
    "Warfarin validation — ABSORPTION-PRESENT subgroup (first obs ≤6h, n=13)\n"
    "FAIR test of full model (absorption+elimination both observable)\n"
    f"Subgroup pooled R²={r2_abs:+.3f}  RMSE={rmse_abs:.3f} mg/L  "
    "[CAVEAT: dose 60–135mg vs. 5mg training — 12–27× extrapolation]",
    fontsize=9,
)
fig_a.tight_layout()
fig_a.savefig(OUT_DIR / "concentration_curves.png", dpi=150, bbox_inches="tight")
plt.close(fig_a)
print("Saved concentration_curves.png  (absorption subgroup)")


# 7c  Trough-only group
ntro = len(sid_list_tro)
ncols_t = 5; nrows_t = (ntro + ncols_t - 1) // ncols_t
fig_t, axes_t = plt.subplots(nrows_t, ncols_t, figsize=(18, 4 * nrows_t))
for ax in axes_t.ravel()[ntro:]:
    ax.axis("off")

for i, sid in enumerate(sid_list_tro):
    ax   = axes_t.ravel()[i]
    info = per_subject[sid]
    t    = np.array(info["times"])

    ax.scatter(t, info["conc_obs"], color="steelblue", s=30, zorder=5,
               label="Observed", clip_on=False)
    ax.plot(t, info["conc_pred"], color="darkorange", lw=2, label="Predicted")
    ax.set_title(
        f"S{sid} (trough)  R²={per_subj_r2[sid]:+.2f}  RMSE={per_subj_rmse[sid]:.2f} mg/L\n"
        f"wt={info['weight_kg']:.0f}kg  dose={info['dose_mg']:.0f}mg  "
        f"age={info['age_years']:.0f}yr  {info['sex']}\n"
        f"CL={info['CL_Lh']:.2f}L/h  Vd={info['V_L']:.1f}L  ka={info['ka_h']:.2f}/h",
        fontsize=7,
    )
    ax.set_xlabel("Time (h)", fontsize=8)
    ax.set_ylabel("Conc (mg/L)", fontsize=8)
    ax.grid(alpha=0.3)
    if i == 0:
        ax.legend(fontsize=7)

fig_t.suptitle(
    "Warfarin validation — TROUGH-ONLY subgroup (first obs ≥24h, n=19)\n"
    "Tests elimination phase only — absorption phase NOT observable\n"
    f"Subgroup pooled R²={r2_tro:+.3f}  RMSE={rmse_tro:.3f} mg/L",
    fontsize=9,
)
fig_t.tight_layout()
fig_t.savefig(OUT_DIR / "concentration_curves_trough.png", dpi=150, bbox_inches="tight")
plt.close(fig_t)
print("Saved concentration_curves_trough.png")


# ── 8. Console summary ────────────────────────────────────────────────────────

r2_abs_list  = [per_subj_r2[s]   for s in sid_list_abs]
r2_tro_list  = [per_subj_r2[s]   for s in sid_list_tro]
rm_abs_list  = [per_subj_rmse[s] for s in sid_list_abs]
rm_tro_list  = [per_subj_rmse[s] for s in sid_list_tro]

print("\n" + "=" * 68)
print("REAL WARFARIN VALIDATION — SUMMARY")
print("CHECKPOINT: hybrid_gnn_pbpk_warfarin_v1  (simulated-test R²=0.781)")
print("=" * 68)
print(f"\n  Naive baseline R² (predict mean):        {r2_naive:+.4f}   {rmse_naive:.4f} mg/L")
print(f"\n  (a) ALL 32 subjects pooled:              R²={r2_all:+.4f}   RMSE={rmse_all:.4f} mg/L  n={int(mask_abs.sum())+int(mask_tro.sum())}")
print(f"  (b) Absorption-present group (n=13):     R²={r2_abs:+.4f}   RMSE={rmse_abs:.4f} mg/L  [FAIR TEST]")
print(f"  (c) Trough-only group (n=19):            R²={r2_tro:+.4f}   RMSE={rmse_tro:.4f} mg/L  [elimination only]")

print(f"\n  Per-subject R² — absorption group:")
print(f"    min={min(r2_abs_list):.3f}  median={np.median(r2_abs_list):.3f}  max={max(r2_abs_list):.3f}")
print(f"  Per-subject R² — trough-only group:")
print(f"    min={min(r2_tro_list):.3f}  median={np.median(r2_tro_list):.3f}  max={max(r2_tro_list):.3f}")

print("\n  Per-subject detail:")
print(f"  {'Subj':>5}  {'Group':>8}  {'R²':>7}  {'RMSE':>8}  {'CL':>7}  {'Vd':>7}  {'ka':>6}")
print(f"  {'':->5}  {'':->8}  {'':->7}  {'mg/L':->8}  {'L/h':->7}  {'L':->7}  {'/h':->6}")
for sid in sid_list_all:
    grp_lbl = "abs" if per_subject[sid]["abs_group"] else "trough"
    info    = per_subject[sid]
    print(f"  {sid:>5}  {grp_lbl:>8}  {per_subj_r2[sid]:>+7.3f}  "
          f"{per_subj_rmse[sid]:>8.3f}  "
          f"{info['CL_Lh']:>7.3f}  {info['V_L']:>7.2f}  {info['ka_h']:>6.3f}")

print("\n  CAVEATS ACTIVE IN THIS RUN:")
print("  1. DOSE EXTRAPOLATION: real doses 60-153mg vs training 5mg (12-30x).")
print("     Scaler z-scores: dose_mg~+95, dose_mgkg~+70. ODE concentration")
print("     scales linearly with dose so R² is valid, but PK parameter")
print("     predictions are deeply outside the training feature space.")
print("  2. ABSORPTION PHASE: 19/32 subjects have first obs at t>=24h.")
print("     Trough-only R² tests elimination phase only.")
print("=" * 68)


# ── 9. Write results markdown ─────────────────────────────────────────────────

subj_abs_lines = []
for sid in sid_list_abs:
    info = per_subject[sid]
    subj_abs_lines.append(
        f"| {sid:>4} | {per_subj_r2[sid]:>+6.3f} | {per_subj_rmse[sid]:>8.3f} | "
        f"{info['dose_mg']:>7.0f} | {info['weight_kg']:>5.0f} | "
        f"{info['age_years']:>4.0f} | {info['sex']:>6} | "
        f"{info['CL_Lh']:>6.3f} | {info['V_L']:>7.2f} | {info['ka_h']:>6.3f} |"
    )

subj_tro_lines = []
for sid in sid_list_tro:
    info = per_subject[sid]
    subj_tro_lines.append(
        f"| {sid:>4} | {per_subj_r2[sid]:>+6.3f} | {per_subj_rmse[sid]:>8.3f} | "
        f"{info['dose_mg']:>7.0f} | {info['weight_kg']:>5.0f} | "
        f"{info['age_years']:>4.0f} | {info['sex']:>6} | "
        f"{info['CL_Lh']:>6.3f} | {info['V_L']:>7.2f} | {info['ka_h']:>6.3f} |"
    )

md = f"""# Real Warfarin Validation Results

**Experiment branch:** experiment/warfarin-validation
**Checkpoint:** `artifacts/models/hybrid_gnn_pbpk_warfarin_v1/model.pt`
**Inference mode:** `model.eval()` + `torch.no_grad()` — forward pass only, no retraining
**Simulated-test R² (reference):** 0.781 (val 0.820)

---

## CAVEATS — Read Before Interpreting

### Caveat 1: 20× Dose Extrapolation

The model was trained exclusively on **5 mg warfarin doses** (scaler mean = 5.0 mg, std = 1.0 mg).
This real dataset uses **~60–153 mg doses** (mean 105 mg, the classical 1.5 mg/kg characterization dose
from O'Reilly et al. 1963/1968).

- `dose_mg` scaler z-score for real data: **(100 − 5) / 1 ≈ +95 SD**
- `dose_mgkg` scaler z-score: **(1.5 − 0.077) / 0.021 ≈ +68 SD**

The ODE computes concentration as `C(t) = (F × dose_mg / Vd) × f(CL/Vd, ka, t)`, so concentration
scales **linearly** with `dose_mg` at fixed PK parameters. R² is scale-invariant. These two facts
mean the test remains *geometrically* meaningful — but the neural network is predicting CL, Vd, ka from
feature vectors that are **~70–95 standard deviations** outside its training distribution. Poor R² here
may reflect extrapolation failure rather than a flawed model for clinically-dosed warfarin.

### Caveat 2: Absorption Phase Missing for 19/32 Subjects

The warfarin dataset merges two sub-studies:

| Group | n | Subject IDs | First obs (post-dose) |
|-------|---|-------------|----------------------|
| **Absorption-present** | 13 | 1, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16 | ≤6h — absorption peak visible |
| **Trough-only** | 19 | 2, 10, 17–33 | ≥24h — elimination phase only |

For the 19 trough-only subjects, the model predicts a full absorption+elimination curve but only the
late elimination phase is observed. Per-subject R² for these subjects reflects how well the model
predicts the mono-exponential decline — **not** whether absorption (ka) is correctly captured.

The **absorption-present group (n=13) is the fair, full-model test**.

---

## Covariate Mapping — All Real, No Imputation

| Covariate | Source | Note |
|-----------|--------|------|
| `weight_kg` | `wt` column | Real individual (40–102 kg, mean 70 kg) |
| `dose_mg` | `amt` at dosing event | Real individual (60–153 mg, mean 105 mg) |
| `dose_mgkg` | `dose_mg / weight_kg` | Derived (~1.3–2.0 mg/kg) |
| `age_years` | `age` column | Real individual (21–63 yr, mean 31 yr); training mean was 44.3 yr |
| `sex` | `sex` column, female→0, male→1 | Real individual (27M / 5F) |

Unlike the theophylline validation, **no covariates were imputed**.
The real subjects are systematically younger (mean 31 yr) than the training distribution (mean 44 yr),
which adds a mild covariate shift on top of the dose extrapolation.

---

## Pooled Metrics

| View | R² | RMSE (mg/L) | n obs | Notes |
|------|----|-------------|-------|-------|
| Naive baseline (predict mean) | {r2_naive:+.4f} | {rmse_naive:.4f} | {len(y_obs)} | R² ≈ 0 confirms baseline |
| **(a) All 32 subjects** | **{r2_all:+.4f}** | **{rmse_all:.4f}** | {len(y_obs)} | pooled, all caveats active |
| **(b) Absorption-present (n=13)** | **{r2_abs:+.4f}** | **{rmse_abs:.4f}** | {int(mask_abs.sum())} | **FAIR TEST — both phases** |
| **(c) Trough-only (n=19)** | **{r2_tro:+.4f}** | **{rmse_tro:.4f}** | {int(mask_tro.sum())} | elimination phase only |

---

## Per-Subject Results — Absorption-Present Group (n=13, FAIR TEST)

| Subj | R² | RMSE mg/L | Dose mg | Wt kg | Age | Sex | CL L/h | Vd L | ka /h |
|-----:|---:|----------:|--------:|------:|----:|----:|-------:|-----:|------:|
{chr(10).join(subj_abs_lines)}

Per-subject R² (absorption group): min={min(r2_abs_list):.3f}, median={np.median(r2_abs_list):.3f}, max={max(r2_abs_list):.3f}
Per-subject RMSE: min={min(rm_abs_list):.3f}, median={np.median(rm_abs_list):.3f}, max={max(rm_abs_list):.3f} mg/L

---

## Per-Subject Results — Trough-Only Group (n=19, elimination phase only)

| Subj | R² | RMSE mg/L | Dose mg | Wt kg | Age | Sex | CL L/h | Vd L | ka /h |
|-----:|---:|----------:|--------:|------:|----:|----:|-------:|-----:|------:|
{chr(10).join(subj_tro_lines)}

Per-subject R² (trough group): min={min(r2_tro_list):.3f}, median={np.median(r2_tro_list):.3f}, max={max(r2_tro_list):.3f}
Per-subject RMSE: min={min(rm_tro_list):.3f}, median={np.median(rm_tro_list):.3f}, max={max(rm_tro_list):.3f} mg/L

---

## Interpretation

### Simulation-to-reality gap (reference comparison)

| Drug | Simulated-test R² | Real-data R² (fair subgroup) | Gap |
|------|:-----------------:|:---------------------------:|:---:|
| Theophylline | 0.827 | 0.673 (all 12 subjects) | −0.154 |
| Warfarin | 0.781 | {r2_abs:+.3f} (absorption subgroup, n=13) | {r2_abs - 0.781:+.3f} |

### What this result means

The warfarin validation is a **heavily caveated stress-test**, not a standard validation:

1. **Dose extrapolation dominates uncertainty.** Training on 5 mg, testing on 100 mg places
   the input features ~70–95 standard deviations outside the training manifold. The ODE is
   analytically dose-linear, so R² can be positive — but PK parameters (CL, Vd, ka)
   predicted at these extreme input values may differ systematically from what the model
   learned at 5 mg doses.

2. **Absorption group is the meaningful test.** The 13 subjects with early time points
   allow the model's absorption phase to be assessed. The 19 trough-only subjects test only
   elimination kinetics and inflate or deflate the pooled R² in ways that do not reflect
   the model's overall capability.

3. **What a positive R² proves (even if small):** the model captures the correct general
   shape and direction of warfarin concentration-time decay, and the ODE's linear dose-scaling
   holds over this extreme extrapolation range. This is actually a non-trivial result.

4. **What a negative or near-zero R² would prove:** the neural network's PK-parameter
   regression fails to extrapolate across a 70–95 SD feature gap, or the model's warfarin
   pharmacokinetics at low doses do not generalise to the 1.5 mg/kg dose regime. Neither
   would invalidate the model for its intended use (5 mg therapeutic dosing).

---

## Files

| File | Description |
|------|-------------|
| `run_warfarin_eval.py` | Evaluation script (forward pass only) |
| `warfarin_results.md` | This report |
| `warfarin_predictions.csv` | Per-observation predicted vs observed |
| `pred_vs_obs.png` | Scatter plot, all subjects (circles=absorption, squares=trough) |
| `concentration_curves.png` | 13-panel absorption-subgroup curves |
| `concentration_curves_trough.png` | 19-panel trough-subgroup curves |
| `raw/warfarin.rda` | Original data file (O'Reilly/Holford, via nlmixr2data) |
"""

(OUT_DIR / "warfarin_results.md").write_text(md, encoding="utf-8")
print(f"\nSaved warfarin_results.md")
print("Done.")
