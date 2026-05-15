#!/usr/bin/env python3
"""Call the live FastAPI multi-drug paths (health, predict, explain).

Example:
  python scripts/demo_multidrug_api.py --drug digoxin --dose-mg 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def post_json(url: str, body: dict) -> tuple[int, dict]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, {"detail": raw}


def main() -> int:
    ap = argparse.ArgumentParser(description="Demo multidrug API (urllib, no extra deps).")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000", help="API root")
    ap.add_argument("--drug", default="warfarin", help="panel_drug slug")
    ap.add_argument("--dose-mg", type=float, default=10.0)
    ap.add_argument("--weight-kg", type=float, default=70.0)
    ap.add_argument("--age", type=float, default=55.0)
    ap.add_argument("--sex", type=float, default=0.0)
    args = ap.parse_args()
    base = args.base_url.rstrip("/")

    with urllib.request.urlopen(f"{base}/health", timeout=10) as r:
        health = json.loads(r.read().decode("utf-8"))
    print("GET /health\n", json.dumps(health, indent=2))

    pred_body = {
        "patient": {
            "weight_kg": args.weight_kg,
            "compound_name": args.drug,
            "age_years": args.age,
            "sex": args.sex,
        },
        "drug": {"name": args.drug, "panel_drug": args.drug},
        "regimen": [{"time_hr": 0.0, "dose_mg": args.dose_mg, "route": "oral"}],
        "horizon_hr": 48.0,
    }
    code, pred = post_json(f"{base}/predict/v2", pred_body)
    print(f"\nPOST /predict/v2 -> {code}\n", json.dumps(pred, indent=2)[:4000])

    explain_body = {**pred_body, "shap_seed": 7}
    code2, ex = post_json(f"{base}/explain/v2", explain_body)
    print(f"\nPOST /explain/v2 -> {code2}")
    if code2 == 200:
        print("shap.backend:", ex.get("shap", {}).get("attribution_backend"))
        print("shap.features:", ex.get("shap", {}).get("features"))
    else:
        print(json.dumps(ex, indent=2)[:2000])
    return 0 if code == 200 and code2 == 200 else 1


if __name__ == "__main__":
    sys.exit(main())
