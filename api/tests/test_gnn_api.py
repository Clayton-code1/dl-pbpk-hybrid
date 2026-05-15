"""API tests for SMILES / GNN drug handling on /predict/v2."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

_ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "models" / "hybrid_theoph_v1"
_HAS_MODEL = (_ARTIFACT_DIR / "model.pt").exists()


def _base_payload(**overrides):
    payload = {
        "patient": {"weight_kg": 70, "compound_name": "Theophylline"},
        "regimen": [{"time_hr": 0, "dose_mg": 320, "route": "oral"}],
        "horizon_hr": 48,
    }
    payload.update(overrides)
    return payload


class TestTheophyllineWithoutSMILES:
    """Theophylline requests should work without explicit SMILES (backward compat)."""

    def test_no_drug_field(self):
        resp = client.post("/predict/v2", json=_base_payload())
        if _HAS_MODEL:
            assert resp.status_code == 200
        else:
            assert resp.status_code == 503

    def test_drug_name_theophylline_no_smiles(self):
        payload = _base_payload(drug={"name": "Theophylline"})
        resp = client.post("/predict/v2", json=payload)
        if _HAS_MODEL:
            assert resp.status_code == 200
            data = resp.json()
            assert data["model"]["model_used"] in ("mlp", "gnn", "multidrug_gnn")
        else:
            assert resp.status_code == 503

    def test_drug_with_smiles(self):
        payload = _base_payload(
            drug={"name": "Theophylline", "smiles": "Cn1c2c(c(=O)n(c1=O)C)[nH]cn2"}
        )
        resp = client.post("/predict/v2", json=payload)
        if _HAS_MODEL:
            assert resp.status_code == 200
        else:
            assert resp.status_code == 503


class TestUnknownDrugWithoutSMILES:
    """Non-panel drug without SMILES should return 400."""

    def test_unknown_drug_no_smiles_returns_400(self):
        if not _HAS_MODEL:
            return
        payload = _base_payload(
            patient={"weight_kg": 70, "compound_name": "FictionalDrugXYZ"},
            drug={"name": "FictionalDrugXYZ"},
        )
        resp = client.post("/predict/v2", json=payload)
        assert resp.status_code == 400
        assert "SMILES required" in resp.json()["detail"]

    def test_unknown_drug_with_smiles_passes(self):
        if not _HAS_MODEL:
            return
        payload = _base_payload(
            drug={"name": "Caffeine", "smiles": "Cn1c(=O)c2c(ncn2C)n(c1=O)C"}
        )
        resp = client.post("/predict/v2", json=payload)
        assert resp.status_code == 200


class TestModelMetadata:
    """Verify model_used field is present in response."""

    def test_model_used_in_response(self):
        if not _HAS_MODEL:
            return
        resp = client.post("/predict/v2", json=_base_payload())
        assert resp.status_code == 200
        data = resp.json()
        assert "model_used" in data["model"]
        assert data["model"]["model_used"] in ("mlp", "gnn", "multidrug_gnn", None)
