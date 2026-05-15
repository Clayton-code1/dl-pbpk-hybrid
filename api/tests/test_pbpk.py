"""Tests for PBPK-lite multi-tissue ODE scaffold."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
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
# /predict/v2 with PBPK tissue output
# ---------------------------------------------------------------------------

def test_predict_v2_pbpk_lite_returns_pbpk_block():
    payload = {**_PAYLOAD, "include_tissues": False, "pbpk_mode": "pbpk_lite"}
    resp = client.post("/predict/v2", json=payload)
    if resp.status_code == 503:
        pytest.skip("model not loaded")
    assert resp.status_code == 200
    data = resp.json()
    assert "pbpk" in data
    pbpk = data["pbpk"]
    assert pbpk["enabled"] is True
    assert "params" in pbpk
    assert "CL_hep_l_h" in pbpk["params"]
    assert "CL_ren_l_h" in pbpk["params"]
    assert pbpk["tissues"] is None


def test_predict_v2_with_tissues():
    payload = {**_PAYLOAD, "include_tissues": True, "pbpk_mode": "pbpk_lite"}
    resp = client.post("/predict/v2", json=payload)
    if resp.status_code == 503:
        pytest.skip("model not loaded")
    assert resp.status_code == 200
    data = resp.json()
    pbpk = data["pbpk"]
    assert pbpk is not None
    assert pbpk["tissues"] is not None
    n_times = len(data["time_h"])
    for tissue in ["liver", "kidney", "muscle", "adipose", "lung"]:
        assert tissue in pbpk["tissues"]
        assert len(pbpk["tissues"][tissue]) == n_times


def test_predict_v2_pk1cpt_fallback():
    payload = {**_PAYLOAD, "pbpk_mode": "pk1cpt"}
    resp = client.post("/predict/v2", json=payload)
    if resp.status_code == 503:
        pytest.skip("model not loaded")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("pbpk") is None


# ---------------------------------------------------------------------------
# Mass balance / data quality
# ---------------------------------------------------------------------------

def test_pbpk_concentrations_non_negative():
    payload = {**_PAYLOAD, "include_tissues": True, "pbpk_mode": "pbpk_lite"}
    resp = client.post("/predict/v2", json=payload)
    if resp.status_code == 503:
        pytest.skip("model not loaded")
    data = resp.json()
    for c in data["concentration_ng_ml"]:
        assert c >= 0.0
    for tissue, vals in data["pbpk"]["tissues"].items():
        for v in vals:
            assert v >= 0.0, f"Negative concentration in {tissue}"


def test_pbpk_no_nans():
    payload = {**_PAYLOAD, "include_tissues": True, "pbpk_mode": "pbpk_lite"}
    resp = client.post("/predict/v2", json=payload)
    if resp.status_code == 503:
        pytest.skip("model not loaded")
    data = resp.json()
    import math
    for c in data["concentration_ng_ml"]:
        assert not math.isnan(c)
    for tissue, vals in data["pbpk"]["tissues"].items():
        for v in vals:
            assert not math.isnan(v), f"NaN in {tissue}"


# ---------------------------------------------------------------------------
# Population + PBPK
# ---------------------------------------------------------------------------

def test_population_still_works_with_pbpk():
    payload = {**_PAYLOAD, "population": {"n_samples": 50, "seed": 42}}
    resp = client.post("/predict/population", json=payload)
    if resp.status_code == 503:
        pytest.skip("model not loaded")
    assert resp.status_code == 200
    data = resp.json()
    bands = data["population"]["bands"]
    n = len(bands["p05"])
    assert n > 50
    for i in range(n):
        assert bands["p05"][i] <= bands["p50"][i] + 0.01
        assert bands["p50"][i] <= bands["p95"][i] + 0.01


# ---------------------------------------------------------------------------
# /pbpk/config endpoints
# ---------------------------------------------------------------------------

def test_get_pbpk_config():
    resp = client.get("/pbpk/config")
    assert resp.status_code == 200
    cfg = resp.json()
    assert "enabled" in cfg
    assert "Q_co_ref_L_per_h" in cfg
    assert "flow_fracs" in cfg
    assert "vol_fracs" in cfg
    assert "Kp" in cfg
    assert "f_hep" in cfg
    assert "fu" in cfg


def test_update_pbpk_config():
    original = client.get("/pbpk/config").json()
    new_f_hep = 0.6
    resp = client.post("/pbpk/config", json={"f_hep": new_f_hep})
    assert resp.status_code == 200
    assert resp.json()["f_hep"] == new_f_hep
    # Verify it persisted
    check = client.get("/pbpk/config").json()
    assert check["f_hep"] == new_f_hep
    # Restore original
    client.post("/pbpk/config", json={"f_hep": original["f_hep"]})


def test_update_pbpk_config_kp():
    resp = client.post("/pbpk/config", json={"Kp": {"liver": 2.5}})
    assert resp.status_code == 200
    assert resp.json()["Kp"]["liver"] == 2.5
    # Restore
    client.post("/pbpk/config", json={"Kp": {"liver": 2.0}})
