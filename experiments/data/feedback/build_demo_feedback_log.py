"""Build ``feedback_log.csv`` demo rows from theophylline test patients (model pred + noise).

Run from repo root::

    python experiments/data/feedback/build_demo_feedback_log.py
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.config import PROCESSED_DATA_DIR  # noqa: E402
from experiments.evaluation.evaluate_multidrug import _load_model  # noqa: E402
from experiments.reference_pk import REFERENCE_PK_DATA  # noqa: E402
from experiments.training.multidrug_utils import load_drug_dataset, load_drug_graph  # noqa: E402

import pandas as pd  # noqa: E402


def main() -> None:
    drug = "theophylline"
    _, _, test_recs, _ = load_drug_dataset(drug)
    model, _ = _load_model(drug)
    graph = load_drug_graph(drug)
    model.eval()
    rng = np.random.default_rng(42)

    df = pd.read_csv(PROCESSED_DATA_DIR / f"{drug}_pk_dataset.csv")
    ts = datetime.now(timezone.utc).isoformat()
    rows: list[list] = []
    for rec in test_recs[:3]:
        fr = df[df["patient_id"] == rec.patient_id].iloc[0]
        with torch.no_grad():
            pred, *_ = model(
                graph["x"],
                graph["edge_index"],
                graph["edge_attr"],
                rec.features,
                rec.times_hr,
                rec.dose_mg,
                rec.weight_kg,
                rec.f_bio,
            )
        pred_np = pred.numpy()
        obs = pred_np + rng.normal(0, 0.05, size=pred_np.shape)
        for t_h, p, o in zip(rec.times_hr.numpy(), pred_np, obs, strict=True):
            rows.append(
                [
                    ts,
                    drug,
                    rec.patient_id,
                    float(fr["weight_kg"]),
                    float(fr["dose_mg"]),
                    float(fr["age_years"]),
                    float(fr["sex"]),
                    float(t_h),
                    float(p),
                    float(o),
                    "seed_demo_test_split",
                ],
            )

    out = _ROOT / "experiments" / "data" / "feedback" / "feedback_log.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "timestamp",
                "drug",
                "patient_id",
                "weight_kg",
                "dose_mg",
                "age_years",
                "sex",
                "t_hours",
                "predicted_mg_per_L",
                "observed_mg_per_L",
                "source",
            ],
        )
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out} (F_ref={REFERENCE_PK_DATA[drug]['F']})", flush=True)


if __name__ == "__main__":
    main()
