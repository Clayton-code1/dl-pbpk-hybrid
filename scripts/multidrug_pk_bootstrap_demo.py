#!/usr/bin/env python3
"""Local parametric spread of panel hybrid PK parameters (minimal bootstrap-style demo).

Perturbs normalized patient features with Gaussian noise, re-runs the panel forward,
and prints CL/V/ka quantiles. Does not call HTTP.

Run from repository root with API venv, app on path:

  cd api
  ..\\.venv\\Scripts\\python.exe ..\\scripts\\multidrug_pk_bootstrap_demo.py warfarin --B 50

Unix:

  cd api && python ../scripts/multidrug_pk_bootstrap_demo.py warfarin --B 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("drug", help="panel slug, e.g. warfarin")
    ap.add_argument("--B", type=int, default=40, help="bootstrap draws")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dose-mg", type=float, default=10.0)
    ap.add_argument("--weight-kg", type=float, default=70.0)
    ap.add_argument("--age", type=float, default=40.0)
    ap.add_argument("--sex", type=float, default=0.0)
    ap.add_argument("--noise", type=float, default=0.08, help="sigma on normalized features")
    args = ap.parse_args()

    api_dir = Path(__file__).resolve().parents[1] / "api"
    sys.path.insert(0, str(api_dir))
    repo = api_dir.parent
    sys.path.insert(0, str(repo))

    from app.services import hybrid_infer_service as infer
    from app.services import multidrug_bundle as mdb

    drug = args.drug.strip().lower()
    bundle = mdb.load_multidrug_bundle(drug)
    if bundle is None:
        print(f"No bundle for {drug!r}; install artifacts + graph.", file=sys.stderr)
        return 1

    full = mdb.build_raw_feature_row(drug, args.dose_mg, args.weight_kg, args.age, args.sex)
    idx_map = {n: i for i, n in enumerate(mdb.patient_feature_column_names(drug))}
    ref = np.array([full[idx_map[n]] for n in bundle.feature_names], dtype=np.float64)
    mean, std = bundle.mean.astype(np.float64), bundle.std.astype(np.float64)
    z0 = (ref - mean) / std

    rng = np.random.default_rng(args.seed)
    CLS, VS, KAS = [], [], []
    for _ in range(args.B):
        z = z0 + rng.normal(0.0, args.noise, size=z0.shape)
        raw = z * std + mean
        pk = infer.predict_multidrug_pk_from_raw(drug, raw.astype(np.float32), float(raw[bundle.feature_names.index("weight_kg")]))
        if pk:
            CLS.append(pk[0])
            VS.append(pk[1])
            KAS.append(pk[2])

    if not CLS:
        print("No successful forward passes.")
        return 1

    def q(x):
        a = np.array(x)
        return float(np.percentile(a, 5)), float(np.median(a)), float(np.percentile(a, 95))

    print(f"Drug={drug} B={args.B} noise_sigma={args.noise} (normalized space)")
    print(f"CL  p5/median/p95: {q(CLS)}")
    print(f"V   p5/median/p95: {q(VS)}")
    print(f"ka  p5/median/p95: {q(KAS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
