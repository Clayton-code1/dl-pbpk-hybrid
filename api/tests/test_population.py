"""Tests for Layer 4: Population adapter endpoints."""

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
    "dt_min": 5.0,
}


# ---------------------------------------------------------------------------
# /predict/population
# ---------------------------------------------------------------------------

def test_predict_population_returns_bands_or_503():
    payload = {**_PAYLOAD, "population": {"n_samples": 100, "seed": 42}}
    resp = client.post("/predict/population", json=payload)

    if _HAS_MODEL:
        assert resp.status_code == 200
        data = resp.json()

        assert "deterministic" in data
        det = data["deterministic"]
        assert len(det["time_h"]) > 50

        pop = data["population"]
        assert pop["n_samples"] == 100
        assert len(pop["times_hr"]) > 50
        assert len(pop["bands"]["p05"]) == len(pop["times_hr"])
        assert len(pop["bands"]["p50"]) == len(pop["times_hr"])
        assert len(pop["bands"]["p95"]) == len(pop["times_hr"])
    else:
        assert resp.status_code == 503


def test_population_p05_le_p50_le_p95():
    if not _HAS_MODEL:
        return

    payload = {**_PAYLOAD, "population": {"n_samples": 100, "seed": 42}}
    resp = client.post("/predict/population", json=payload)
    assert resp.status_code == 200
    pop = resp.json()["population"]

    for i in range(0, len(pop["times_hr"]), 10):
        assert pop["bands"]["p05"][i] <= pop["bands"]["p50"][i] + 0.01, \
            f"p05 > p50 at index {i}"
        assert pop["bands"]["p50"][i] <= pop["bands"]["p95"][i] + 0.01, \
            f"p50 > p95 at index {i}"


def test_population_risk_fields():
    if not _HAS_MODEL:
        return

    payload = {**_PAYLOAD, "population": {"n_samples": 100, "seed": 42}}
    resp = client.post("/predict/population", json=payload)
    assert resp.status_code == 200
    risk = resp.json()["population"]["population_risk"]
    assert 0.0 <= risk["p_unsafe"] <= 1.0
    assert 0.0 <= risk["p_safe"] <= 1.0
    assert abs(risk["p_unsafe"] + risk["p_safe"] - 1.0) < 0.001


def test_population_metrics_dist():
    if not _HAS_MODEL:
        return

    payload = {**_PAYLOAD, "population": {"n_samples": 100, "seed": 42}}
    resp = client.post("/predict/population", json=payload)
    assert resp.status_code == 200
    md = resp.json()["population"]["metrics_dist"]

    for metric in ("cmax", "auc", "tmax"):
        d = md[metric]
        assert d["p05"] <= d["p50"] + 0.01
        assert d["p50"] <= d["p95"] + 0.01
        assert d["p05"] >= 0


def test_population_reproducible_with_seed():
    if not _HAS_MODEL:
        return

    payload = {**_PAYLOAD, "population": {"n_samples": 50, "seed": 123}}
    r1 = client.post("/predict/population", json=payload).json()["population"]
    r2 = client.post("/predict/population", json=payload).json()["population"]

    assert r1["bands"]["p50"][:5] == r2["bands"]["p50"][:5]
    assert r1["population_risk"] == r2["population_risk"]


# ---------------------------------------------------------------------------
# /population/config
# ---------------------------------------------------------------------------

def test_get_population_config():
    resp = client.get("/population/config")
    assert resp.status_code == 200
    data = resp.json()
    assert "omega_cl" in data
    assert "omega_v" in data
    assert "omega_ka" in data
    assert "n_samples" in data


def test_update_population_config_persists():
    new_omega = 0.35
    resp = client.post("/population/config", json={"omega_cl": new_omega})
    assert resp.status_code == 200
    assert resp.json()["omega_cl"] == new_omega

    resp2 = client.get("/population/config")
    assert resp2.json()["omega_cl"] == new_omega

    # Restore default
    client.post("/population/config", json={"omega_cl": 0.25})


def test_population_config_n_samples_capped():
    resp = client.post("/population/config", json={"n_samples": 999})
    assert resp.status_code == 422  # validation should reject >500


# ---------------------------------------------------------------------------
# /predict/v2 with include_population query param
# ---------------------------------------------------------------------------

def test_predict_v2_with_population_flag():
    if not _HAS_MODEL:
        return

    resp = client.post("/predict/v2?include_population=true", json=_PAYLOAD)
    assert resp.status_code == 200
    data = resp.json()
    assert data["population"] is not None
    assert "bands" in data["population"]
    assert "population_risk" in data["population"]


def test_predict_v2_without_population_flag():
    if not _HAS_MODEL:
        return

    resp = client.post("/predict/v2", json=_PAYLOAD)
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("population") is None
