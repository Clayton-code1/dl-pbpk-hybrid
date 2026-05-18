"""Explainability service: SHAP feature attribution + mechanistic sensitivity.

Provides:
- explain_shap(): KernelSHAP on risk_score — MLP (3 features) or panel hybrid
  (training-aligned raw covariates + scaler) when *panel_drug* is set.
- sensitivity_analysis(): local +/-5 % perturbation of (CL, V, ka) and its effect
  on AUC and Cmax (baseline from MLP or ``predict_multidrug_pk``).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("uvicorn.error")

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DATA_PATH = _PROJECT_ROOT / "data" / "processed" / "theoph" / "theoph_subjects.json"

# Cached KernelSHAP backgrounds sampled from real training CSVs (panel path).
_PANEL_SHAP_BACKGROUND_CACHE: dict[tuple[str, int, int], np.ndarray] = {}


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def _build_background(n: int = 30, seed: int | None = None) -> np.ndarray:
    """Small background dataset for KernelSHAP from real Theoph subjects."""
    rng = np.random.RandomState(seed if seed is not None else 0)
    if _DATA_PATH.exists():
        with open(_DATA_PATH, "r") as f:
            subjects = json.load(f)
        rows = [[s["dose_mg"], s["weight_kg"], s["dose_mgkg"]] for s in subjects]
        bg = np.array(rows, dtype=np.float64)
        if len(bg) >= n:
            idx = rng.choice(len(bg), size=n, replace=False)
            return bg[idx]
        return bg
    # Fallback: synthetic samples around population mean
    dose_mg = rng.uniform(200, 400, n)
    wt = rng.uniform(55, 90, n)
    return np.column_stack([dose_mg, wt, dose_mg / wt])


def _build_panel_background(
    ref_raw: np.ndarray,
    feature_names: list[str],
    n: int,
    seed: int | None,
) -> np.ndarray:
    """Perturb raw training-scale features for KernelSHAP background (panel path).

    Kept as a fallback when training CSV sampling fails.
    """
    rng = np.random.default_rng(seed)
    ref = np.asarray(ref_raw, dtype=np.float64).reshape(-1)
    bg = np.zeros((n, len(ref)), dtype=np.float64)
    for i in range(n):
        row = ref.copy()
        for j, fname in enumerate(feature_names):
            if fname == "sex":
                row[j] = float(rng.choice([0.0, 1.0]))
            else:
                row[j] = max(ref[j] * float(rng.uniform(0.75, 1.25)), 1e-9)
        bg[i] = row
    return bg


def _ensure_experiments_on_path() -> Path:
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def load_panel_shap_background_training(
    drug: str,
    feature_names: list[str],
    *,
    n: int = 50,
    seed: int = 42,
) -> np.ndarray | None:
    """Sample ``n`` raw covariate rows from the drug's *training* CSV split (SEED logic aligned with training)."""
    cache_key = (drug, int(n), int(seed))
    if cache_key in _PANEL_SHAP_BACKGROUND_CACHE:
        return _PANEL_SHAP_BACKGROUND_CACHE[cache_key]

    _ensure_experiments_on_path()
    try:
        import pandas as pd

        from experiments.config import PROCESSED_DATA_DIR
        from experiments.training.multidrug_utils import (
            patient_feature_columns,
            split_patient_ids,
            split_rng_seed_for_drug,
        )
    except Exception as exc:
        logger.warning("Panel SHAP background import failed: %s", exc)
        return None

    csv_path = PROCESSED_DATA_DIR / f"{drug}_pk_dataset.csv"
    if not csv_path.exists():
        logger.warning("PK CSV missing for panel SHAP background: %s", csv_path)
        return None

    try:
        df = pd.read_csv(csv_path)
        if "dose_mgkg" not in df.columns:
            df["dose_mgkg"] = df["dose_mg"] / df["weight_kg"].replace(0, np.nan)
        df["dose_mg_per_kg"] = df["dose_mg"] / df["weight_kg"].replace(0, np.nan)
        df["log_dose_mg_per_kg"] = np.log(df["dose_mg_per_kg"] + 1e-8)

        n_patients = int(df["patient_id"].max()) + 1
        split_seed = split_rng_seed_for_drug(drug)
        train_ids, _, _ = split_patient_ids(n_patients, seed=split_seed)
        train_df = df[df["patient_id"].isin(train_ids)].copy()
        first = train_df.groupby("patient_id", sort=True).head(1)
        if len(first) == 0:
            return None

        need_cols = patient_feature_columns(drug)
        for c in need_cols:
            if c not in first.columns:
                logger.warning("Column %s missing for %s background", c, drug)
                return None

        rng = np.random.default_rng(seed)
        k = min(n, len(first))
        pick = rng.choice(len(first), size=k, replace=False)
        rows = first.iloc[pick]

        bg = np.zeros((k, len(feature_names)), dtype=np.float64)
        for i in range(k):
            row = rows.iloc[i]
            vec = np.array([float(row[c]) for c in feature_names], dtype=np.float64)
            bg[i] = vec

        _PANEL_SHAP_BACKGROUND_CACHE[cache_key] = bg
        logger.info("Panel SHAP: loaded %d-row training background for %s", k, drug)
        return bg
    except Exception:
        logger.exception("Failed building training SHAP background for %s", drug)
        return None


def _humanize_feature_names(names: list[str]) -> list[str]:
    labels = {
        "weight_kg": "Weight (kg)",
        "dose_mg": "Dose (mg)",
        "dose_mgkg": "Dose / weight (mg/kg)",
        "dose_mg_per_kg": "Dose / weight (mg/kg)",
        "log_dose_mg_per_kg": "log(dose/weight)",
        "age_years": "Age (y)",
        "sex": "Sex (0/1)",
    }
    return [labels.get(n, n) for n in names]


def explain_shap(
    dose_mg: float,
    weight_kg: float,
    *,
    panel_drug: str | None = None,
    age_years: float = 40.0,
    sex: float = 0.0,
    shap_seed: int | None = None,
) -> dict[str, Any]:
    """Compute SHAP values for risk_score; aligns with MLP or panel multi-drug predictor."""
    if panel_drug:
        return _explain_shap_panel(
            panel_drug, dose_mg, weight_kg, age_years, sex, shap_seed=shap_seed,
        )
    return _explain_shap_mlp(dose_mg, weight_kg, shap_seed=shap_seed)


def _explain_shap_mlp(dose_mg: float, weight_kg: float, *, shap_seed: int | None = None) -> dict[str, Any]:
    """Compute SHAP values for risk_score with respect to input features."""
    from app.services import hybrid_infer_service as infer
    from app.services import risk_service

    infer._ensure_loaded()
    if infer._model is None:
        return {
            "features": [],
            "values": [],
            "base": 0.0,
            "target": "risk_score",
            "attribution_backend": None,
        }

    def _model_fn(X: np.ndarray) -> np.ndarray:
        """Map raw features -> risk_score for each row."""
        scores = []
        for row in X:
            d, w, dmk = float(row[0]), float(row[1]), float(row[2])
            feat = infer.build_features(d, w)
            CL, V, ka = infer.predict_params(feat)
            ke = CL / V
            # Quick single-dose Euler to get Cmax + AUC
            n_steps = 200
            horizon = 48.0
            dt = horizon / n_steps
            A_gut, A_cent = d, 0.0
            cmax_mgl, auc_mgl = 0.0, 0.0
            for _ in range(n_steps):
                conc = max(A_cent / V, 0.0)
                cmax_mgl = max(cmax_mgl, conc)
                auc_mgl += conc * dt
                dg = -ka * A_gut
                dc = ka * A_gut - ke * A_cent
                A_gut += dg * dt
                A_cent += dc * dt
            risk = risk_service.assess_risk(cmax_mgl * 1000, auc_mgl * 1000)
            scores.append(risk["risk_score"])
        return np.array(scores)

    try:
        import shap
        bg = _build_background(20, seed=shap_seed)
        explainer = shap.KernelExplainer(_model_fn, bg, link="identity")
        x_input = np.array([[dose_mg, weight_kg, dose_mg / max(weight_kg, 1.0)]])
        sv = explainer.shap_values(x_input, nsamples=64, silent=True)
        if isinstance(sv, list):
            sv = sv[0]
        vals = sv[0].tolist() if sv.ndim > 1 else sv.tolist()
        base = float(explainer.expected_value)
        if isinstance(base, np.ndarray):
            base = float(base[0])
    except Exception as exc:
        logger.warning("SHAP computation failed, using finite-difference fallback: %s", exc)
        vals, base = _finite_diff_attribution(dose_mg, weight_kg, _model_fn)

    features = ["Dose (mg)", "Body Weight (kg)", "Dose/kg (mg/kg)"]
    return {
        "features": features,
        "values": [round(v, 6) for v in vals],
        "base": round(base, 6),
        "target": "risk_score",
        "attribution_backend": "mlp",
    }


def _explain_shap_panel(
    panel_drug: str,
    dose_mg: float,
    weight_kg: float,
    age_years: float,
    sex: float,
    *,
    shap_seed: int | None = None,
) -> dict[str, Any]:
    """SHAP on risk_score for the panel hybrid (same raw features + scaler as training)."""
    from app.services import hybrid_infer_service as infer
    from app.services import multidrug_bundle as mdb
    from app.services import risk_service

    bundle = mdb.load_multidrug_bundle(panel_drug)
    if bundle is None:
        return {
            "features": [],
            "values": [],
            "base": 0.0,
            "target": "risk_score",
            "attribution_backend": None,
        }

    full = mdb.build_raw_feature_row(panel_drug, float(dose_mg), float(weight_kg), float(age_years), float(sex))
    idx_map = {n: i for i, n in enumerate(mdb.patient_feature_column_names(panel_drug))}
    ref_raw = np.array([full[idx_map[n]] for n in bundle.feature_names], dtype=np.float64)
    w_idx = bundle.feature_names.index("weight_kg")

    f_bio = mdb.oral_bioavailability(panel_drug)

    def _model_fn(X: np.ndarray) -> np.ndarray:
        scores = []
        for row in X:
            w = float(row[w_idx])
            pk = infer.predict_multidrug_pk_from_raw(panel_drug, row.astype(np.float32), w)
            if pk is None:
                scores.append(0.0)
                continue
            CL, V, ka = pk
            tot_d = float(row[bundle.feature_names.index("dose_mg")])
            d_eff = tot_d * f_bio
            cmax_mgl, auc_mgl = _quick_sim(d_eff, CL, V, ka, 48.0, w)
            risk = risk_service.assess_risk(cmax_mgl * 1000, auc_mgl * 1000, drug=panel_drug)
            scores.append(risk["risk_score"])
        return np.array(scores)

    feat_labels = _humanize_feature_names(bundle.feature_names)

    try:
        import shap
        bg = load_panel_shap_background_training(
            panel_drug,
            bundle.feature_names,
            n=50,
            seed=shap_seed if shap_seed is not None else 42,
        )
        if bg is None:
            bg = _build_panel_background(ref_raw, bundle.feature_names, 24, shap_seed)
        explainer = shap.KernelExplainer(_model_fn, bg, link="identity")
        x_input = ref_raw.reshape(1, -1)
        sv = explainer.shap_values(x_input, nsamples=96, silent=True)
        if isinstance(sv, list):
            sv = sv[0]
        vals_arr = sv[0] if sv.ndim > 1 else sv
        vals = vals_arr.tolist()
        base = float(explainer.expected_value)
        if isinstance(base, np.ndarray):
            base = float(base[0])
    except Exception as exc:
        logger.warning("Panel SHAP failed, finite-difference fallback: %s", exc)
        vals, base = _finite_diff_attribution_array(ref_raw, _model_fn)

    return {
        "features": feat_labels,
        "values": [round(float(v), 6) for v in vals],
        "base": round(float(base), 6),
        "target": "risk_score",
        "attribution_backend": "panel_multidrug",
    }


def _finite_diff_attribution(dose_mg: float, weight_kg: float, model_fn) -> tuple[list[float], float]:
    """Fallback: central finite-difference attribution when SHAP is unavailable."""
    x0 = np.array([[dose_mg, weight_kg, dose_mg / max(weight_kg, 1.0)]])
    y0 = float(model_fn(x0)[0])
    eps_frac = 0.05
    vals = []
    for j in range(3):
        xp = x0.copy(); xp[0, j] *= (1 + eps_frac)
        xm = x0.copy(); xm[0, j] *= (1 - eps_frac)
        yp = float(model_fn(xp)[0])
        ym = float(model_fn(xm)[0])
        vals.append((yp - ym) / 2.0)
    return vals, y0


def _finite_diff_attribution_array(x0: np.ndarray, model_fn) -> tuple[list[float], float]:
    """Central finite-difference attribution for a generic feature vector."""
    x0r = np.asarray(x0, dtype=np.float64).reshape(1, -1)
    y0 = float(model_fn(x0r)[0])
    eps_frac = 0.05
    vals = []
    for j in range(x0r.shape[1]):
        h = max(abs(float(x0r[0, j])), 1e-6) * eps_frac
        xp = x0r.copy()
        xm = x0r.copy()
        xp[0, j] = float(xp[0, j]) + h
        xm[0, j] = float(xm[0, j]) - h
        yp = float(model_fn(xp)[0])
        ym = float(model_fn(xm)[0])
        vals.append((yp - ym) / 2.0)
    return vals, y0


# ---------------------------------------------------------------------------
# Sensitivity analysis  (mechanistic: CL, V, ka -> AUC, Cmax)
# ---------------------------------------------------------------------------

def sensitivity_analysis(
    dose_mg: float,
    weight_kg: float,
    route: str = "oral",
    horizon_hr: float = 48.0,
    *,
    panel_drug: str | None = None,
    age_years: float = 40.0,
    sex: float = 0.0,
) -> dict[str, Any]:
    """Perturb CL, V, ka by +/-5 % and measure delta AUC / Cmax."""
    from app.services import hybrid_infer_service as infer
    from app.services import multidrug_bundle as mdb

    d_sim = float(dose_mg)
    if route == "oral" and panel_drug:
        d_sim *= mdb.oral_bioavailability(panel_drug)

    if panel_drug:
        pk = infer.predict_multidrug_pk(panel_drug, float(dose_mg), weight_kg, age_years, sex)
        if pk is None:
            return _empty_sensitivity()
        CL0, V0, ka0 = pk
    else:
        infer._ensure_loaded()
        if infer._model is None:
            return _empty_sensitivity()
        features = infer.build_features(dose_mg, weight_kg)
        CL0, V0, ka0 = infer.predict_params(features)
        if route == "oral":
            d_sim = dose_mg

    base_cmax, base_auc = _quick_sim(d_sim, CL0, V0, ka0, horizon_hr, weight_kg)

    params = {"CL": CL0, "V": V0, "ka": ka0}
    delta_pct = 0.05
    delta_auc: list[float] = []
    delta_cmax: list[float] = []

    for pname in ["CL", "V", "ka"]:
        p_val = params[pname]
        up = {**params, pname: p_val * (1 + delta_pct)}
        dn = {**params, pname: p_val * (1 - delta_pct)}

        cmax_up, auc_up = _quick_sim(d_sim, up["CL"], up["V"], up["ka"], horizon_hr, weight_kg)
        cmax_dn, auc_dn = _quick_sim(d_sim, dn["CL"], dn["V"], dn["ka"], horizon_hr, weight_kg)

        d_auc = ((auc_up - auc_dn) / max(base_auc, 1e-9)) * 100 / (2 * delta_pct * 100)
        d_cmax = ((cmax_up - cmax_dn) / max(base_cmax, 1e-9)) * 100 / (2 * delta_pct * 100)
        delta_auc.append(round(d_auc, 4))
        delta_cmax.append(round(d_cmax, 4))

    rank_auc = _rank(delta_auc)
    rank_cmax = _rank(delta_cmax)

    return {
        "parameters": ["CL", "V", "ka"],
        "baseline_values": [round(CL0, 4), round(V0, 4), round(ka0, 4)],
        "delta_auc_pct": delta_auc,
        "delta_cmax_pct": delta_cmax,
        "rank_auc": rank_auc,
        "rank_cmax": rank_cmax,
    }


def _quick_sim(dose_mg: float, CL: float, V: float, ka: float, horizon: float, weight_kg: float = 70.0) -> tuple[float, float]:
    """Fast single-dose oral simulation via PBPK-lite -> (cmax_ng_ml, auc_ng_h_ml)."""
    from app.services import pbpk_service

    events = [{"time_hr": 0.0, "dose_mg": dose_mg, "route": "oral"}]
    times, conc_ng = pbpk_service.simulate_pbpk_with_params(
        events, weight_kg, CL, V, ka,
        horizon_hr=horizon, dt_min=5.0,
    )
    conc_arr = np.array(conc_ng)
    t_arr = np.array(times)
    cmax = float(np.max(conc_arr))
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc = float(_trapz(conc_arr, t_arr))
    return cmax, auc


def _rank(vals: list[float]) -> list[int]:
    """Rank by absolute magnitude (1 = most influential)."""
    indexed = sorted(enumerate(vals), key=lambda x: abs(x[1]), reverse=True)
    ranks = [0] * len(vals)
    for rank, (idx, _) in enumerate(indexed, 1):
        ranks[idx] = rank
    return ranks


def _empty_sensitivity() -> dict[str, Any]:
    return {
        "parameters": ["CL", "V", "ka"],
        "baseline_values": [0, 0, 0],
        "delta_auc_pct": [0, 0, 0],
        "delta_cmax_pct": [0, 0, 0],
        "rank_auc": [1, 2, 3],
        "rank_cmax": [1, 2, 3],
    }


# ---------------------------------------------------------------------------
# Narrative generation
# ---------------------------------------------------------------------------

def drug_structure_effect(
    smiles: str | None,
    dose_mg: float,
    weight_kg: float,
    horizon_hr: float = 48.0,
    *,
    panel_drug: str | None = None,
) -> dict[str, Any]:
    """Estimate a drug-structure contribution by comparing the GNN embedding
    against a zero-centred reference embedding.

    Returns a dict with ``drug_structure_delta_risk`` (float) and a short
    ``explanation`` string.  If no GNN model is loaded, returns zeroes.
    """
    from app.services import hybrid_infer_service as infer
    from app.services import risk_service

    if panel_drug:
        return {
            "drug_structure_delta_risk": 0.0,
            "explanation": (
                f"Panel hybrid ({panel_drug}) encodes structure via a fixed graph + "
                "drug-specific checkpoint; there is no alternate embedding to compare within this endpoint."
            ),
        }

    if smiles is None or not infer.is_gnn_loaded():
        return {
            "drug_structure_delta_risk": 0.0,
            "explanation": "Drug structure effect unavailable (MLP model in use).",
        }

    try:
        CL_gnn, V_gnn, ka_gnn = infer.predict_params_gnn(smiles, dose_mg, weight_kg)
    except Exception:
        return {
            "drug_structure_delta_risk": 0.0,
            "explanation": "Drug structure effect unavailable (SMILES error).",
        }

    cmax_gnn, auc_gnn = _quick_sim(dose_mg, CL_gnn, V_gnn, ka_gnn, horizon_hr, weight_kg)
    risk_gnn = risk_service.assess_risk(cmax_gnn, auc_gnn)["risk_score"]

    infer._ensure_loaded()
    if infer._model is not None:
        features = infer.build_features(dose_mg, weight_kg)
        CL_mlp, V_mlp, ka_mlp = infer.predict_params(features)
        cmax_mlp, auc_mlp = _quick_sim(dose_mg, CL_mlp, V_mlp, ka_mlp, horizon_hr, weight_kg)
        risk_mlp = risk_service.assess_risk(cmax_mlp, auc_mlp)["risk_score"]
    else:
        risk_mlp = risk_gnn

    delta = risk_gnn - risk_mlp
    if abs(delta) < 0.01:
        msg = "Drug structure has minimal effect on predicted risk."
    elif delta > 0:
        msg = f"Drug structure increases predicted risk by {delta:.3f} vs. the structure-agnostic baseline."
    else:
        msg = f"Drug structure decreases predicted risk by {abs(delta):.3f} vs. the structure-agnostic baseline."

    return {"drug_structure_delta_risk": round(delta, 6), "explanation": msg}


def generate_narrative(
    shap_result: dict, sensitivity_result: dict,
    dose_mg: float, weight_kg: float, risk_score: float, is_safe: bool,
) -> dict[str, Any]:
    """Build plain-language explanation from SHAP + sensitivity results."""
    features = shap_result.get("features", [])
    sv = shap_result.get("values", [])

    # Sort features by |SHAP value|
    if features and sv:
        pairs = sorted(zip(features, sv), key=lambda x: abs(x[1]), reverse=True)
        top_feature, top_val = pairs[0]
        direction = "increases" if top_val > 0 else "decreases"
    else:
        top_feature, direction = "dose", "affects"

    status = "safe" if is_safe else "unsafe"
    dose_mgkg = dose_mg / max(weight_kg, 1.0)

    summary = (
        f"For a {weight_kg:.0f} kg patient receiving {dose_mg:.0f} mg "
        f"({dose_mgkg:.1f} mg/kg), the predicted risk score is {risk_score:.3f} "
        f"({status}). "
        f"The most influential input feature is {top_feature}, which {direction} "
        f"the predicted risk. "
    )

    sens = sensitivity_result
    if sens.get("rank_cmax"):
        p_names = sens["parameters"]
        cmax_deltas = sens["delta_cmax_pct"]
        top_idx = sens["rank_cmax"].index(1)
        summary += (
            f"Among the mechanistic PK parameters, {p_names[top_idx]} has the "
            f"greatest influence on peak concentration (Cmax sensitivity: "
            f"{cmax_deltas[top_idx]:+.2f}%/%). "
        )

    # PBPK tissue-driver context
    from app.services import pbpk_service
    pbpk_cfg = pbpk_service.get_pbpk_config()
    f_hep = pbpk_cfg.get("f_hep", 0.7)
    if f_hep >= 0.5:
        summary += (
            f"Under the PBPK-lite model, hepatic clearance dominates "
            f"({f_hep*100:.0f}% of total CL), with renal clearance contributing "
            f"the remainder."
        )
    else:
        summary += (
            f"Under the PBPK-lite model, renal clearance dominates "
            f"({(1-f_hep)*100:.0f}% of total CL), with hepatic clearance contributing "
            f"the remainder."
        )

    key_drivers = []
    if features and sv:
        for f, v in sorted(zip(features, sv), key=lambda x: abs(x[1]), reverse=True):
            key_drivers.append({"feature": f, "shap_value": round(v, 6), "direction": "risk+" if v > 0 else "risk-"})

    return {"summary": summary, "key_drivers": key_drivers}
