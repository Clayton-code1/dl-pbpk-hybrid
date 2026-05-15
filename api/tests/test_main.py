"""API integration tests.

If model artifacts are present the hybrid-inference tests run;
otherwise the 503-guard tests run.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

_ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "models" / "hybrid_theoph_v1"
_HAS_MODEL = (_ARTIFACT_DIR / "model.pt").exists()


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert data.get("inference_ready") in (True, False)
    assert isinstance(data.get("panel_drugs_available", {}), dict)


# ---------------------------------------------------------------------------
# Legacy /predict (always works — fallback or hybrid)
# ---------------------------------------------------------------------------

def test_predict_legacy():
    resp = client.post(
        "/predict",
        json={"compound_name": "TestCompound", "dose_mg": 100, "weight_kg": 70},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["compound_name"] == "TestCompound"
    assert len(data["time_h"]) > 50
    assert len(data["concentration_ng_ml"]) == len(data["time_h"])
    assert data["pk_metrics"]["cmax_ng_ml"] > 0


# ---------------------------------------------------------------------------
# /predict/v2
# ---------------------------------------------------------------------------

def test_predict_v2_returns_result_or_503():
    payload = {
        "patient": {"weight_kg": 70, "compound_name": "Theophylline"},
        "regimen": [{"time_hr": 0, "dose_mg": 320, "route": "oral"}],
        "horizon_hr": 48,
    }
    resp = client.post("/predict/v2", json=payload)

    if _HAS_MODEL:
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["time_h"]) > 50
        assert "pk_metrics" in data
        assert "safety" in data
        assert data["safety"]["risk_score"] >= 0
        assert "pk_params" in data
        ver = data["model"]["version"]
        mu = data["model"].get("model_used")
        assert mu in ("mlp", "gnn", "multidrug_gnn", None)
        assert isinstance(ver, str) and ver.startswith("hybrid_")
    else:
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# /recommend
# ---------------------------------------------------------------------------

def test_recommend_returns_strategies_or_503():
    payload = {
        "patient": {"weight_kg": 70, "compound_name": "Theophylline"},
        "regimen": [{"time_hr": 0, "dose_mg": 320, "route": "oral"}],
        "horizon_hr": 48,
    }
    resp = client.post("/recommend", json=payload)

    if _HAS_MODEL:
        assert resp.status_code == 200
        data = resp.json()
        assert "baseline" in data
        assert "strategies" in data
        strats = data["strategies"]
        assert len(strats) == 3
        for s in strats:
            assert "pk_metrics" in s
            assert "delta_cmax_pct" in s
            assert "delta_auc_pct" in s
            assert "scale_factor" in s
            assert 0 < s["scale_factor"] <= 1.0
            assert len(s["time_h"]) > 50
            assert len(s["concentration_ng_ml"]) == len(s["time_h"])
    else:
        assert resp.status_code == 503


def test_recommend_at_least_one_strategy_safe_when_baseline_unsafe():
    """When the baseline prediction is unsafe, at least one strategy must be safe."""
    if not _HAS_MODEL:
        return  # skip when model not available

    payload = {
        "patient": {"weight_kg": 70, "compound_name": "Theophylline"},
        "regimen": [{"time_hr": 0, "dose_mg": 320, "route": "oral"}],
        "horizon_hr": 48,
    }
    resp = client.post("/recommend", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    if not data["baseline"]["safety"]["is_safe"]:
        safe_strategies = [s for s in data["strategies"] if s["safety"]["is_safe"]]
        assert len(safe_strategies) >= 1, (
            f"Baseline is unsafe but none of the {len(data['strategies'])} strategies are safe. "
            f"Scale factors: {[s['scale_factor'] for s in data['strategies']]}"
        )


# ---------------------------------------------------------------------------
# /model/state  &  /model/update
# ---------------------------------------------------------------------------

def test_model_state():
    resp = client.get("/model/state")
    assert resp.status_code == 200
    data = resp.json()
    assert "calibration" in data
    assert "update_flag" in data


def test_model_update_safe():
    resp = client.post("/model/update", json={"feedback_label": "safe"})
    assert resp.status_code == 200
    assert "calibration" in resp.json()


def test_model_update_invalid():
    resp = client.post("/model/update", json={"feedback_label": "maybe"})
    assert resp.status_code == 400
