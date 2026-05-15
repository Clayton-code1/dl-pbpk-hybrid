"""Population adapter: Bayesian-style random effects on PK parameters.

Produces uncertainty bands (5th-95th percentile) for concentration curves,
PK metric distributions, and population safety probability.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from app.services import risk_service

logger = logging.getLogger("uvicorn.error")

_STATE_PATH = Path(__file__).resolve().parent.parent / "state" / "model_state.json"

_OMEGA_DEFAULTS = {
    "omega_cl": 0.25,
    "omega_v": 0.20,
    "omega_ka": 0.30,
    "n_samples": 200,
}

_MAX_SAMPLES = 500


def _load_population_config() -> dict[str, float]:
    if _STATE_PATH.exists():
        with open(_STATE_PATH, "r") as f:
            state = json.load(f)
        return state.get("population", dict(_OMEGA_DEFAULTS))
    return dict(_OMEGA_DEFAULTS)


def _save_population_config(cfg: dict[str, float]) -> None:
    if _STATE_PATH.exists():
        with open(_STATE_PATH, "r") as f:
            state = json.load(f)
    else:
        state = {}
    state["population"] = cfg
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def get_population_config() -> dict[str, float]:
    return _load_population_config()


def update_population_config(updates: dict[str, float]) -> dict[str, float]:
    cfg = _load_population_config()
    for key in ("omega_cl", "omega_v", "omega_ka", "n_samples"):
        if key in updates:
            cfg[key] = updates[key]
    if cfg.get("n_samples", 200) > _MAX_SAMPLES:
        cfg["n_samples"] = _MAX_SAMPLES
    _save_population_config(cfg)
    return cfg


def sample_params(
    CL_base: float,
    V_base: float,
    ka_base: float,
    omega_cl: float,
    omega_v: float,
    omega_ka: float,
    n_samples: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw log-normal random-effect samples around base PK parameters.

    Returns arrays of shape (n_samples,) for CL, V, ka.
    """
    rng = np.random.default_rng(seed)
    CL_samples = CL_base * np.exp(rng.normal(0, omega_cl, n_samples))
    V_samples = V_base * np.exp(rng.normal(0, omega_v, n_samples))
    ka_samples = ka_base * np.exp(rng.normal(0, omega_ka, n_samples))
    return CL_samples, V_samples, ka_samples


def simulate_population(
    events: list[dict],
    weight_kg: float,
    horizon_hr: float = 48.0,
    dt_min: float = 5.0,
    n_samples: int | None = None,
    omega_cl: float | None = None,
    omega_v: float | None = None,
    omega_ka: float | None = None,
    seed: int | None = None,
    *,
    panel_drug: str | None = None,
    age_years: float = 40.0,
    sex: float = 0.0,
) -> dict[str, Any]:
    """Run population simulation with random-effect variability.

    Returns dict with curve bands, metric distributions, and population risk.
    """
    from app.services import hybrid_infer_service as infer
    from app.services import multidrug_bundle as mdb

    cfg = _load_population_config()
    omega_cl = omega_cl if omega_cl is not None else cfg.get("omega_cl", _OMEGA_DEFAULTS["omega_cl"])
    omega_v = omega_v if omega_v is not None else cfg.get("omega_v", _OMEGA_DEFAULTS["omega_v"])
    omega_ka = omega_ka if omega_ka is not None else cfg.get("omega_ka", _OMEGA_DEFAULTS["omega_ka"])
    n = n_samples if n_samples is not None else int(cfg.get("n_samples", _OMEGA_DEFAULTS["n_samples"]))
    n = min(n, _MAX_SAMPLES)
    if n > 200:
        logger.warning("Population simulation with n_samples=%d (>200), may be slow", n)

    total_dose = float(sum(e["dose_mg"] for e in events))
    sim_events = list(events)
    panel_resolved: str | None = None
    CL_base: float | None = None
    V_base: float | None = None
    ka_base: float | None = None

    if panel_drug:
        pk = infer.predict_multidrug_pk(panel_drug, total_dose, weight_kg, age_years, sex)
        if pk is not None:
            CL_base, V_base, ka_base = pk
            panel_resolved = panel_drug
            sim_events = infer._scale_oral_doses_for_f(events, mdb.oral_bioavailability(panel_drug))

    if CL_base is None or V_base is None or ka_base is None:
        infer._ensure_loaded()
        if infer._model is None:
            raise RuntimeError(
                "Population simulation requires either a loaded panel drug checkpoint "
                "or hybrid_theoph_v1 (MLP).",
            )
        features = infer.build_features(total_dose, weight_kg)
        CL_base, V_base, ka_base = infer.predict_params(features)
        sim_events = list(events)
        panel_resolved = None

    CL_samples, V_samples, ka_samples = sample_params(
        CL_base, V_base, ka_base, omega_cl, omega_v, omega_ka, n, seed,
    )

    from app.services import pbpk_service

    dt_hr = dt_min / 60.0
    out_times = np.arange(0.0, horizon_hr + dt_hr * 0.5, dt_hr)
    n_out = len(out_times)

    all_conc = np.zeros((n, n_out))
    all_cmax = np.zeros(n)
    all_auc = np.zeros(n)
    all_tmax = np.zeros(n)
    unsafe_count = 0

    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

    for si in range(n):
        CL_i = float(CL_samples[si])
        V_i = float(V_samples[si])
        ka_i = float(ka_samples[si])

        times_i, conc_ng_ml_list = pbpk_service.simulate_pbpk_with_params(
            sim_events, weight_kg, CL_i, V_i, ka_i,
            horizon_hr=horizon_hr, dt_min=dt_min,
        )

        conc_ng_ml = np.array(conc_ng_ml_list)
        all_conc[si] = conc_ng_ml
        cmax = float(np.max(conc_ng_ml))
        all_cmax[si] = cmax
        all_tmax[si] = float(out_times[int(np.argmax(conc_ng_ml))])
        all_auc[si] = float(_trapz(conc_ng_ml, out_times))

        risk_kw: dict[str, Any] = {}
        if panel_resolved:
            risk_kw["drug"] = panel_resolved
        risk = risk_service.assess_risk(cmax, all_auc[si], **risk_kw)
        if not risk["is_safe"]:
            unsafe_count += 1

    p05 = np.percentile(all_conc, 5, axis=0)
    p50 = np.percentile(all_conc, 50, axis=0)
    p95 = np.percentile(all_conc, 95, axis=0)

    p_unsafe = unsafe_count / n
    p_safe = 1.0 - p_unsafe

    return {
        "n_samples": n,
        "base_params": {
            "cl": round(CL_base, 4),
            "v": round(V_base, 4),
            "ka": round(ka_base, 4),
        },
        "omega": {
            "cl": round(omega_cl, 4),
            "v": round(omega_v, 4),
            "ka": round(omega_ka, 4),
        },
        "times_hr": [round(float(t), 4) for t in out_times],
        "bands": {
            "p05": [round(float(v), 2) for v in p05],
            "p50": [round(float(v), 2) for v in p50],
            "p95": [round(float(v), 2) for v in p95],
        },
        "metrics_dist": {
            "cmax": {
                "p05": round(float(np.percentile(all_cmax, 5)), 2),
                "p50": round(float(np.percentile(all_cmax, 50)), 2),
                "p95": round(float(np.percentile(all_cmax, 95)), 2),
            },
            "auc": {
                "p05": round(float(np.percentile(all_auc, 5)), 2),
                "p50": round(float(np.percentile(all_auc, 50)), 2),
                "p95": round(float(np.percentile(all_auc, 95)), 2),
            },
            "tmax": {
                "p05": round(float(np.percentile(all_tmax, 5)), 2),
                "p50": round(float(np.percentile(all_tmax, 50)), 2),
                "p95": round(float(np.percentile(all_tmax, 95)), 2),
            },
        },
        "population_risk": {
            "p_unsafe": round(p_unsafe, 4),
            "p_safe": round(p_safe, 4),
        },
    }
