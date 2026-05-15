"""Phase 3.3 — SHAP on patient covariates vs hybrid-predicted AUC.

Uses KernelSHAP on a scalar summary (trapezoidal AUC of the predicted curve):
``(graph, patient features) -> AUC``. Graph/encoder is fixed per drug.
Each SHAP model call perturbs the z-scored patient vector while holding that
patient's dose, weight, sampling times, and F fixed (oral simulation path).

Outputs
-------
- ``experiments/results/phase3_shap_interpretation.md``
- ``experiments/plots/shap_summary_multidrug.png`` (+ PDF) — 2×3 panel figure.

    python -m experiments.explainability.shap_interpretation
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
from PIL import Image

_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.config import (  # noqa: E402
    DRUGS,
    PLOTS_DIR,
    RESULTS_DIR,
    SEED,
    ensure_dirs,
    get_logger,
    seed_everything,
)
from experiments.evaluation.evaluate_multidrug import _load_model  # noqa: E402
from experiments.training.multidrug_utils import (  # noqa: E402
    load_drug_dataset,
    load_drug_graph,
    patient_feature_columns,
)

LOGGER = get_logger("phase3.shap", "phase3_shap.log")

NSAMPLES = 96
BG_SIZE = 36
N_REF_PATIENTS = 8


def _auc_trapz(conc: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    dt = times[1:] - times[:-1]
    return torch.sum(0.5 * (conc[1:] + conc[:-1]) * dt)


def _predict_auc_for_rec(
    model: torch.nn.Module,
    graph: dict[str, torch.Tensor],
    rec: Any,
    X: np.ndarray,
) -> np.ndarray:
    """Rows of ``X`` are perturbed patient-feature vectors for **one** ``rec``."""
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
            out.append(float(_auc_trapz(conc, rec.times_hr).item()))
    return np.asarray(out, dtype=np.float64)


def _make_kernel_fn(
    model: torch.nn.Module,
    graph: dict[str, torch.Tensor],
    rec: Any,
) -> Callable[[np.ndarray], np.ndarray]:
    return lambda X: _predict_auc_for_rec(model, graph, rec, X)


def main() -> int:
    ensure_dirs()
    seed_everything(SEED)
    rng = np.random.default_rng(SEED)

    md_lines = [
        "# Phase 3.3 — SHAP interpretation (patient covariates → predicted AUC)",
        "",
        "KernelSHAP on the trained hybrid **per drug**, holding the molecular graph fixed. "
        "Background = random training patients. For each of **N** reference test patients, "
        "we explain the AUC response while holding that patient's dose, weight, F, and time grid fixed; "
        "only the z-scored patient tensor entries are coalition-perturbed. "
        f"Approximate Shapley with ``nsamples={NSAMPLES}``.",
        "",
        "## Top 5 mean |SHAP| features per drug",
        "",
    ]

    panel_imgs: list[Image.Image] = []
    universal_ranks: dict[str, int] = {}

    for drug in DRUGS:
        train_recs, _, test_recs, _ = load_drug_dataset(drug)
        graph = load_drug_graph(drug)
        model, _ = _load_model(drug)
        model.eval()

        feat_names = patient_feature_columns(drug)
        bg_idx = rng.choice(len(train_recs), size=min(BG_SIZE, len(train_recs)), replace=False)
        bg = np.stack([train_recs[i].features.numpy() for i in bg_idx])

        sv_rows: list[np.ndarray] = []
        x_rows: list[np.ndarray] = []
        n_ref = min(N_REF_PATIENTS, len(test_recs))
        for rec in test_recs[:n_ref]:
            fn = _make_kernel_fn(model, graph, rec)
            explainer = shap.KernelExplainer(fn, bg)
            phi = explainer.shap_values(rec.features.numpy()[None, :], nsamples=NSAMPLES)
            if isinstance(phi, list):
                phi = phi[0]
            sv_rows.append(phi[0])
            x_rows.append(rec.features.numpy())

        sv = np.stack(sv_rows, axis=0)
        X_disp = np.stack(x_rows, axis=0)

        mean_abs = np.mean(np.abs(sv), axis=0)
        order = np.argsort(-mean_abs)[:5]
        md_lines.append(f"### {drug}")
        md_lines.append("")
        md_lines.append("| Rank | Feature | mean |SHAP| |")
        md_lines.append("|---:|---|---:|")
        for rank, j in enumerate(order, start=1):
            fn = feat_names[int(j)]
            md_lines.append(f"| {rank} | {fn} | {mean_abs[j]:.4f} |")
            universal_ranks[fn] = universal_ranks.get(fn, 0) + 1
        md_lines.append("")

        plt.figure(figsize=(5, 3.5))
        shap.summary_plot(
            sv,
            X_disp,
            feature_names=feat_names,
            plot_type="dot",
            max_display=5,
            show=False,
        )
        plt.title(f"SHAP — {drug}")
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=140, bbox_inches="tight")
        plt.close()
        buf.seek(0)
        panel_imgs.append(Image.open(buf).convert("RGB"))

    md_lines.extend(
        [
            "## Pharmacological notes",
            "",
            "- **weight_kg** (normalised channel): enters the ODE through absolute CL and V; "
            "even with dose fixed in mg, anthropometrics shift exposure per volume.",
            "- **dose_mg** / **dose-normalised inputs**: primary driver of AUC for oral absorption.",
            "- **age_years** / **sex**: captured here as coarse covariates in the fusion MLP head.",
            "",
            "## Cross-drug patterns",
            "",
            "Features appearing most often in this panel's top-5 (mean |SHAP|):",
            "",
        ]
    )
    for fname, cnt in sorted(universal_ranks.items(), key=lambda x: -x[1]):
        md_lines.append(f"- **{fname}** — in top-5 for {cnt} / {len(DRUGS)} drugs.")
    md_lines.append("")

    out_md = RESULTS_DIR / "phase3_shap_interpretation.md"
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    LOGGER.info("Wrote %s", out_md)

    w_max = max(im.size[0] for im in panel_imgs)
    h_max = max(im.size[1] for im in panel_imgs)
    canvas = Image.new("RGB", (w_max * 3, h_max * 2), color="white")
    for idx, im in enumerate(panel_imgs):
        cx, cy = idx % 3, idx // 3
        canvas.paste(
            im,
            (cx * w_max + (w_max - im.size[0]) // 2, cy * h_max + (h_max - im.size[1]) // 2),
        )
    outp = PLOTS_DIR / "shap_summary_multidrug.png"
    canvas.save(outp)
    try:
        canvas.save(outp.with_suffix(".pdf"), resolution=150.0)
    except Exception:  # pragma: no cover
        LOGGER.warning("PDF export for SHAP grid skipped")
    LOGGER.info("Saved %s", outp)

    return 0


if __name__ == "__main__":
    sys.exit(main())
