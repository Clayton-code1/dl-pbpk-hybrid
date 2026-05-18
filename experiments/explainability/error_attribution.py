"""Error-focused attribution: what patient covariates track squared prediction error?

For each drug:
1. Sum of squared errors over the held-out test trajectory (13 time points).
2. Univariate linear regression of SSE vs each raw covariate (coefficient, R², p-value).
3. KernelSHAP on the mapping (z-scored patient features) -> SSE for reference patients.

Outputs
-------
- ``experiments/results/phase3b_error_covariate_regression.csv``
- ``experiments/results/phase3b_error_shap.csv``
- ``experiments/plots/error_attribution_<drug>.png``

    python experiments/explainability/error_attribution.py --drugs theophylline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
from scipy import stats

_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.config import DRUGS, PLOTS_DIR, RESULTS_DIR, SEED, ensure_dirs, get_logger, seed_everything  # noqa: E402
from experiments.evaluation.evaluate_multidrug import _load_model  # noqa: E402
from experiments.training.multidrug_utils import (  # noqa: E402
    load_drug_dataset,
    load_drug_graph,
    patient_feature_columns,
)

LOGGER = get_logger("phase3b.error_attr", "phase3b_error_attr.log")
NSAMPLES = 72
BG_SIZE = 32
N_REF = 8


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Error attribution (SSE + SHAP)")
    p.add_argument("--drugs", type=str, default="", help="Comma-separated slugs; default all")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _sse_pred(model: torch.nn.Module, graph: dict[str, torch.Tensor], rec: Any) -> float:
    with torch.no_grad():
        conc, *_ = model(
            graph["x"],
            graph["edge_index"],
            graph["edge_attr"],
            rec.features,
            rec.times_hr,
            rec.dose_mg,
            rec.weight_kg,
            rec.f_bio,
        )
        diff = conc - rec.concentration.to(conc.device)
        return float((diff ** 2).sum().item())


def _predict_sse_for_rec(
    model: torch.nn.Module,
    graph: dict[str, torch.Tensor],
    rec: Any,
    X: np.ndarray,
) -> np.ndarray:
    out: list[float] = []
    with torch.no_grad():
        for j in range(X.shape[0]):
            pf = torch.tensor(X[j], dtype=torch.float32)
            conc, *_ = model(
                graph["x"],
                graph["edge_index"],
                graph["edge_attr"],
                pf,
                rec.times_hr,
                rec.dose_mg,
                rec.weight_kg,
                rec.f_bio,
            )
            diff = conc - rec.concentration.to(conc.device)
            out.append(float((diff ** 2).sum().item()))
    return np.asarray(out, dtype=np.float64)


def _make_kernel_fn(
    model: torch.nn.Module,
    graph: dict[str, torch.Tensor],
    rec: Any,
) -> Callable[[np.ndarray], np.ndarray]:
    return lambda X: _predict_sse_for_rec(model, graph, rec, X)


def run(drugs: list[str]) -> None:
    ensure_dirs()
    seed_everything(SEED)
    rng = np.random.default_rng(SEED)

    reg_lines: list[str] = []
    shap_lines: list[str] = []

    for drug in drugs:
        LOGGER.info("Error attribution: %s", drug)
        train_recs, _, test_recs, _ = load_drug_dataset(drug)
        graph = load_drug_graph(drug)
        model, _ = _load_model(drug)
        model.eval()

        feat_names = patient_feature_columns(drug)

        # Full test SSE and raw covariates (from normalized rec.features is wrong for regression)
        # Reconstruct raw from first row of CSV-like records — use train scaler inverse
        # Instead: load one row per patient from dataset via records — PatientRecord only has z features.
        # We re-read CSV for raw covariates keyed by patient_id.
        import pandas as pd

        from experiments.config import PROCESSED_DATA_DIR

        csv_path = PROCESSED_DATA_DIR / f"{drug}_pk_dataset.csv"
        df = pd.read_csv(csv_path)
        if "dose_mgkg" not in df.columns:
            df["dose_mgkg"] = df["dose_mg"] / df["weight_kg"]
        df["dose_mg_per_kg"] = df["dose_mg"] / df["weight_kg"]
        df["log_dose_mg_per_kg"] = np.log(df["dose_mg_per_kg"] + 1e-8)

        df = df.sort_values("time_h")
        first_rows = df.groupby("patient_id", sort=True).head(1)
        sse_list: list[float] = []
        raw_rows: list[np.ndarray] = []
        for rec in test_recs:
            sse_list.append(_sse_pred(model, graph, rec))
            pid = int(rec.patient_id)
            fr = first_rows[first_rows["patient_id"] == pid]
            if len(fr) == 0:
                raw_rows.append(np.zeros(len(feat_names)))
            else:
                raw_rows.append(fr.iloc[0][feat_names].to_numpy(dtype=float))
        sse_arr = np.array(sse_list, dtype=np.float64)
        raw_X = np.stack(raw_rows, axis=0)

        for j, cname in enumerate(feat_names):
            xcol = raw_X[:, j]
            if np.std(xcol) < 1e-12:
                reg_lines.append(f"{drug},{cname},0.0,0.0,1.0")
                continue
            slope, intercept, r, p, _ = stats.linregress(xcol, sse_arr)
            r2 = float(r**2)
            reg_lines.append(f"{drug},{cname},{slope:.8f},{r2:.8f},{p:.8e}")

        bg_idx = rng.choice(len(train_recs), size=min(BG_SIZE, len(train_recs)), replace=False)
        bg = np.stack([train_recs[i].features.numpy() for i in bg_idx])

        sv_rows: list[np.ndarray] = []
        n_ref = min(N_REF, len(test_recs))
        for rec in test_recs[:n_ref]:
            fn = _make_kernel_fn(model, graph, rec)
            explainer = shap.KernelExplainer(fn, bg)
            phi = explainer.shap_values(rec.features.numpy()[None, :], nsamples=NSAMPLES)
            if isinstance(phi, list):
                phi = phi[0]
            sv_rows.append(phi[0])
        sv = np.stack(sv_rows, axis=0)
        mean_abs = np.mean(np.abs(sv), axis=0)
        order = np.argsort(-mean_abs)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(order) + 1)
        for j in range(len(feat_names)):
            shap_lines.append(
                f"{drug},{feat_names[j]},{mean_abs[j]:.8f},{int(ranks[j])}",
            )

        plt.figure(figsize=(6, 4))
        ord_j = order[: min(8, len(order))]
        plt.barh(range(len(ord_j)), mean_abs[ord_j][::-1])
        plt.yticks(range(len(ord_j)), [feat_names[i] for i in ord_j[::-1]])
        plt.xlabel("mean |SHAP| on trajectory SSE")
        plt.title(f"Error drivers — {drug}")
        plt.tight_layout()
        outp = PLOTS_DIR / f"error_attribution_{drug}.png"
        plt.savefig(outp, dpi=140)
        plt.close()
        LOGGER.info("Wrote %s", outp)

    (RESULTS_DIR / "phase3b_error_covariate_regression.csv").write_text(
        "drug,covariate,coefficient,R2,p_value\n" + "\n".join(reg_lines) + "\n",
        encoding="utf-8",
    )
    (RESULTS_DIR / "phase3b_error_shap.csv").write_text(
        "drug,covariate,mean_abs_shap_on_error,rank\n" + "\n".join(shap_lines) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Wrote regression + SHAP CSVs under %s", RESULTS_DIR)


def main() -> int:
    args = _parse_args()
    drugs_arg = [d.strip() for d in args.drugs.split(",") if d.strip()]
    drugs = drugs_arg if drugs_arg else list(DRUGS)
    if args.dry_run:
        print(f"error_attribution dry-run OK — drugs={drugs}", flush=True)
        return 0
    try:
        run(drugs)
    except Exception as exc:
        print(f"FAILED error_attribution: {exc}", flush=True)
        LOGGER.exception("error_attribution failed")
        raise
    print(f"error_attribution OK — results in {RESULTS_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
