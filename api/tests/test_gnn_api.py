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


_ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"


class TestUnknownDrugWithoutSMILES:
    """Non-panel drugs: 400 without SMILES, zero-shot 200 with valid SMILES."""

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

    def test_no_drug_field_unknown_compound_returns_400(self):
        if not _HAS_MODEL:
            return
        payload = _base_payload(
            patient={"weight_kg": 70, "compound_name": "FictionalDrugXYZ"},
        )
        resp = client.post("/predict/v2", json=payload)
        assert resp.status_code == 400
        assert "SMILES required" in resp.json()["detail"]

    def test_invalid_smiles_returns_400(self):
        if not _HAS_MODEL:
            return
        payload = _base_payload(
            patient={"weight_kg": 70, "compound_name": "SomeDrug"},
            drug={"name": "SomeDrug", "smiles": "not-valid-smiles!!!"},
        )
        resp = client.post("/predict/v2", json=payload)
        assert resp.status_code == 400
        assert "Invalid SMILES" in resp.json()["detail"]

    def test_zeroshot_without_window_returns_400(self):
        if not _HAS_MODEL:
            return
        payload = _base_payload(
            patient={"weight_kg": 70, "compound_name": "Aspirin"},
            drug={"name": "Aspirin", "smiles": _ASPIRIN_SMILES},
        )
        resp = client.post("/predict/v2", json=payload)
        assert resp.status_code == 400
        assert "Therapeutic window" in resp.json()["detail"]

    def test_zeroshot_partial_window_min_only_returns_400(self):
        if not _HAS_MODEL:
            return
        payload = _base_payload(
            patient={"weight_kg": 70, "compound_name": "Aspirin"},
            drug={"name": "Aspirin", "smiles": _ASPIRIN_SMILES, "therapeutic_min_mg_L": 1.0},
        )
        resp = client.post("/predict/v2", json=payload)
        assert resp.status_code == 400
        assert "Therapeutic window" in resp.json()["detail"]

    def test_unknown_drug_with_smiles_and_window_passes(self):
        if not _HAS_MODEL:
            return
        payload = _base_payload(
            patient={"weight_kg": 70, "compound_name": "Aspirin"},
            drug={
                "name": "Aspirin",
                "smiles": _ASPIRIN_SMILES,
                "therapeutic_min_mg_L": 1.0,
                "therapeutic_max_mg_L": 50.0,
            },
        )
        resp = client.post("/predict/v2", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["model"]["model_used"] == "zeroshot_gnn"

    def test_zeroshot_returns_flagged_response(self):
        if not _HAS_MODEL:
            return
        payload = _base_payload(
            patient={"weight_kg": 70, "compound_name": "Aspirin"},
            drug={
                "name": "Aspirin",
                "smiles": _ASPIRIN_SMILES,
                "therapeutic_min_mg_L": 1.0,
                "therapeutic_max_mg_L": 50.0,
            },
        )
        resp = client.post("/predict/v2", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        zs = data.get("zeroshot")
        assert zs is not None, "Expected 'zeroshot' field in response"
        assert zs["smiles"] == _ASPIRIN_SMILES
        assert "cl_per_kg" in zs
        assert "vd_per_kg" in zs
        assert "ka" in zs
        assert "warning" in zs


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
