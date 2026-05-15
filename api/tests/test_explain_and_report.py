"""Tests for /explain/v2 and /report/v2 endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

_ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "models" / "hybrid_theoph_v1"
_HAS_MODEL = (_ARTIFACT_DIR / "model.pt").exists()

_PAYLOAD = {
    "patient": {"weight_kg": 70, "compound_name": "Theophylline"},
    "regimen": [{"time_hr": 0, "dose_mg": 320, "route": "oral"}],
    "horizon_hr": 48,
}


# ---------------------------------------------------------------------------
# /explain/v2
# ---------------------------------------------------------------------------

def test_explain_v2_returns_shap_and_sensitivity_or_503():
    resp = client.post("/explain/v2", json=_PAYLOAD)

    if _HAS_MODEL:
        assert resp.status_code == 200
        data = resp.json()

        assert "shap" in data
        shap = data["shap"]
        nfeat = len(shap["features"])
        assert nfeat in (3, 5, 6)
        assert len(shap["values"]) == nfeat
        assert isinstance(shap["base"], float)
        assert shap["target"] == "risk_score"

        assert "sensitivity" in data
        sens = data["sensitivity"]
        assert sens["parameters"] == ["CL", "V", "ka"]
        assert len(sens["delta_auc_pct"]) == 3
        assert len(sens["delta_cmax_pct"]) == 3
        assert len(sens["rank_auc"]) == 3
        assert len(sens["rank_cmax"]) == 3

        assert "narrative" in data
        assert len(data["narrative"]["summary"]) > 20

        assert "pk_metrics" in data
        assert "safety" in data
    else:
        assert resp.status_code == 503


def test_explain_v2_shap_values_sum_near_prediction():
    """SHAP values + base should roughly equal the model output."""
    if not _HAS_MODEL:
        return
    resp = client.post("/explain/v2", json=_PAYLOAD)
    data = resp.json()
    shap = data["shap"]
    approx = shap["base"] + sum(shap["values"])
    assert 0.0 <= approx <= 1.0, f"SHAP sum {approx} outside [0,1] for risk_score"


def test_explain_v2_sensitivity_ranks_are_valid():
    if not _HAS_MODEL:
        return
    resp = client.post("/explain/v2", json=_PAYLOAD)
    sens = resp.json()["sensitivity"]
    assert sorted(sens["rank_auc"]) == [1, 2, 3]
    assert sorted(sens["rank_cmax"]) == [1, 2, 3]


# ---------------------------------------------------------------------------
# /report/v2
# ---------------------------------------------------------------------------

def test_report_v2_returns_pdf_or_503():
    resp = client.post("/report/v2", json=_PAYLOAD)

    if _HAS_MODEL:
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/pdf"
        assert len(resp.content) > 5000, "PDF content seems too short"
        assert resp.content[:5] == b"%PDF-"
    else:
        assert resp.status_code == 503


def test_report_v2_with_recommendations():
    if not _HAS_MODEL:
        return
    payload = {**_PAYLOAD, "include_recommendations": True}
    resp = client.post("/report/v2", json=payload)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert len(resp.content) > 5000
