"""PBPK-lite multi-tissue ODE solver.

6-tissue perfusion-limited model with mass-balance flows:
  gut -> central <-> liver, kidney, muscle, adipose, lung

Compartments (amounts in mg):
  A_gut, A_central, A_liver, A_kidney, A_muscle, A_adipose, A_lung

Physiology is scaled allometrically from body weight.
CL_total from the DL model is split into hepatic + renal elimination.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("uvicorn.error")

_STATE_PATH = Path(__file__).resolve().parent.parent / "state" / "model_state.json"

TISSUE_NAMES = ["liver", "kidney", "muscle", "adipose", "lung"]

_PBPK_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "Q_co_ref_L_per_h": 420.0,
    "flow_fracs": {
        "liver": 0.25,
        "kidney": 0.20,
        "muscle": 0.25,
        "adipose": 0.05,
        "lung": 0.25,
    },
    "vol_fracs": {
        "central": 0.07,
        "liver": 0.025,
        "kidney": 0.004,
        "muscle": 0.40,
        "adipose": 0.20,
        "lung": 0.01,
    },
    "Kp": {
        "liver": 2.0,
        "kidney": 1.5,
        "muscle": 1.0,
        "adipose": 3.0,
        "lung": 1.2,
    },
    "f_hep": 0.7,
    "fu": 1.0,
}


def _load_pbpk_config() -> dict[str, Any]:
    if _STATE_PATH.exists():
        with open(_STATE_PATH, "r") as f:
            state = json.load(f)
        cfg = state.get("pbpk")
        if cfg is not None:
            merged = dict(_PBPK_DEFAULTS)
            merged.update(cfg)
            return merged
    return dict(_PBPK_DEFAULTS)


def _save_pbpk_config(cfg: dict[str, Any]) -> None:
    if _STATE_PATH.exists():
        with open(_STATE_PATH, "r") as f:
            state = json.load(f)
    else:
        state = {}
    state["pbpk"] = cfg
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def get_pbpk_config() -> dict[str, Any]:
    return _load_pbpk_config()


def update_pbpk_config(updates: dict[str, Any]) -> dict[str, Any]:
    cfg = _load_pbpk_config()
    for key in ("Q_co_ref_L_per_h", "f_hep", "fu", "enabled"):
        if key in updates:
            cfg[key] = updates[key]
    for dkey in ("flow_fracs", "vol_fracs", "Kp"):
        if dkey in updates and isinstance(updates[dkey], dict):
            if dkey not in cfg or not isinstance(cfg[dkey], dict):
                cfg[dkey] = {}
            cfg[dkey].update(updates[dkey])
    _save_pbpk_config(cfg)
    return cfg


def derive_physiology(
    weight_kg: float,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute volumes, flows, and Kp from patient weight and config."""
    if cfg is None:
        cfg = _load_pbpk_config()

    vf = cfg.get("vol_fracs", _PBPK_DEFAULTS["vol_fracs"])
    V = {k: vf[k] * weight_kg for k in vf}

    Q_co_ref = cfg.get("Q_co_ref_L_per_h", 420.0)
    Q_co = Q_co_ref * (weight_kg / 70.0) ** 0.75

    ff = cfg.get("flow_fracs", _PBPK_DEFAULTS["flow_fracs"])
    Q = {k: ff[k] * Q_co for k in ff}

    Kp = dict(cfg.get("Kp", _PBPK_DEFAULTS["Kp"]))
    f_hep = cfg.get("f_hep", 0.7)
    fu = cfg.get("fu", 1.0)

    return {"V": V, "Q": Q, "Q_co": Q_co, "Kp": Kp, "f_hep": f_hep, "fu": fu}


def _required_euler_steps(phys: dict[str, Any], horizon_hr: float) -> int:
    """Compute minimum Euler steps for stability (dt * max_eigenvalue < 0.8)."""
    V = phys["V"]
    Q = phys["Q"]
    max_rate = 0.0
    for tissue in TISSUE_NAMES:
        vol = V.get(tissue, 1.0)
        flow = Q.get(tissue, 0.0)
        if vol > 0:
            max_rate = max(max_rate, flow / vol)
    central_rate = sum(Q.values()) / max(V.get("central", 1.0), 1e-9)
    max_rate = max(max_rate, central_rate)
    dt_max = 0.8 / max(max_rate, 1.0)
    return max(int(np.ceil(horizon_hr / dt_max)), 1000)


def simulate_pbpk(
    events: list[dict],
    weight_kg: float,
    CL_total: float,
    ka: float,
    horizon_hr: float = 48.0,
    dt_min: float = 5.0,
    return_tissues: bool = False,
    n_euler_steps: int | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the PBPK-lite multi-tissue ODE solver.

    Returns
    -------
    dict with keys:
        times_hr, conc_central_mg_l, conc_central_ng_ml,
        pk_params, pbpk_physiology,
        tissues (optional dict[tissue_name -> list[float]] in mg/L)
    """
    phys = derive_physiology(weight_kg, cfg)
    V = phys["V"]
    Q = phys["Q"]
    Kp = phys["Kp"]
    fu = phys["fu"]

    CL_hep = phys["f_hep"] * CL_total
    CL_ren = (1.0 - phys["f_hep"]) * CL_total

    if n_euler_steps is None:
        n_euler_steps = _required_euler_steps(phys, horizon_hr)

    dt = horizon_hr / n_euler_steps
    t_grid = np.linspace(0.0, horizon_hr, n_euler_steps + 1)

    sorted_events = sorted(events, key=lambda e: e["time_hr"])

    A_gut = 0.0
    A_cent = 0.0
    A_liver = 0.0
    A_kidney = 0.0
    A_muscle = 0.0
    A_adipose = 0.0
    A_lung = 0.0

    n_pts = n_euler_steps + 1
    conc_cent = np.zeros(n_pts)
    tissue_conc: dict[str, np.ndarray] = {}
    if return_tissues:
        for tn in TISSUE_NAMES:
            tissue_conc[tn] = np.zeros(n_pts)

    event_idx = 0

    for i in range(n_pts):
        t = t_grid[i]

        while event_idx < len(sorted_events) and sorted_events[event_idx]["time_hr"] <= t:
            ev = sorted_events[event_idx]
            if ev.get("route", "oral") == "iv":
                A_cent += ev["dose_mg"]
            else:
                A_gut += ev["dose_mg"]
            event_idx += 1

        C_cent = max(A_cent / max(V["central"], 1e-9), 0.0)
        C_liver = max(A_liver / max(V["liver"], 1e-9), 0.0)
        C_kidney = max(A_kidney / max(V["kidney"], 1e-9), 0.0)
        C_muscle = max(A_muscle / max(V["muscle"], 1e-9), 0.0)
        C_adipose = max(A_adipose / max(V["adipose"], 1e-9), 0.0)
        C_lung = max(A_lung / max(V["lung"], 1e-9), 0.0)

        conc_cent[i] = C_cent
        if return_tissues:
            tissue_conc["liver"][i] = C_liver
            tissue_conc["kidney"][i] = C_kidney
            tissue_conc["muscle"][i] = C_muscle
            tissue_conc["adipose"][i] = C_adipose
            tissue_conc["lung"][i] = C_lung

        if i < n_euler_steps:
            # Gut absorption
            dA_gut = -ka * A_gut

            # Tissue exchange (perfusion-limited)
            flux_liver = Q["liver"] * (C_cent - C_liver / Kp["liver"])
            flux_kidney = Q["kidney"] * (C_cent - C_kidney / Kp["kidney"])
            flux_muscle = Q["muscle"] * (C_cent - C_muscle / Kp["muscle"])
            flux_adipose = Q["adipose"] * (C_cent - C_adipose / Kp["adipose"])
            flux_lung = Q["lung"] * (C_cent - C_lung / Kp["lung"])

            # Elimination
            elim_hep = CL_hep * C_liver * fu
            elim_ren = CL_ren * C_kidney * fu

            # Central: gains from gut absorption and tissue return, loses to tissues
            dA_cent = (ka * A_gut
                       - flux_liver - flux_kidney - flux_muscle - flux_adipose - flux_lung)

            dA_liver = flux_liver - elim_hep
            dA_kidney = flux_kidney - elim_ren
            dA_muscle = flux_muscle
            dA_adipose = flux_adipose
            dA_lung = flux_lung

            A_gut += dA_gut * dt
            A_cent += dA_cent * dt
            A_liver += dA_liver * dt
            A_kidney += dA_kidney * dt
            A_muscle += dA_muscle * dt
            A_adipose += dA_adipose * dt
            A_lung += dA_lung * dt

            A_gut = max(A_gut, 0.0)
            A_cent = max(A_cent, 0.0)
            A_liver = max(A_liver, 0.0)
            A_kidney = max(A_kidney, 0.0)
            A_muscle = max(A_muscle, 0.0)
            A_adipose = max(A_adipose, 0.0)
            A_lung = max(A_lung, 0.0)

    # Downsample to output resolution
    dt_hr = dt_min / 60.0
    out_times = np.arange(0.0, horizon_hr + dt_hr * 0.5, dt_hr)
    out_conc = np.interp(out_times, t_grid, conc_cent)
    out_conc = np.maximum(out_conc, 0.0)

    conc_ng_ml = out_conc * 1000.0

    times_list = [round(float(t), 4) for t in out_times]
    conc_list = [round(float(c), 2) for c in conc_ng_ml]

    result: dict[str, Any] = {
        "times_hr": times_list,
        "conc_central_mg_l": [round(float(c), 6) for c in out_conc],
        "conc_central_ng_ml": conc_list,
        "pk_params": {
            "CL_l_h": round(CL_total, 4),
            "V_central_l": round(V["central"], 4),
            "ka_1_h": round(ka, 4),
            "CL_hep_l_h": round(CL_hep, 4),
            "CL_ren_l_h": round(CL_ren, 4),
        },
        "pbpk_physiology": {
            "Q_co_l_h": round(phys["Q_co"], 2),
            "Kp": {k: round(v, 2) for k, v in Kp.items()},
            "f_hep": phys["f_hep"],
            "fu": fu,
        },
    }

    if return_tissues:
        tissues_out: dict[str, list[float]] = {}
        for tn in TISSUE_NAMES:
            interp = np.interp(out_times, t_grid, tissue_conc[tn])
            interp = np.maximum(interp, 0.0)
            tissues_out[tn] = [round(float(c), 6) for c in interp]
        result["tissues"] = tissues_out

    return result


def simulate_pbpk_with_params(
    events: list[dict],
    weight_kg: float,
    CL: float,
    V_unused: float,
    ka: float,
    horizon_hr: float = 48.0,
    dt_min: float = 5.0,
    n_euler_steps: int | None = None,
    cfg: dict[str, Any] | None = None,
) -> tuple[list[float], list[float]]:
    """Convenience wrapper returning (times_hr, conc_ng_ml) for the plasma compartment.

    Used by population_adapter and xai_service which need the same
    signature as the old 1-cpt solver.
    """
    res = simulate_pbpk(
        events, weight_kg, CL, ka,
        horizon_hr=horizon_hr, dt_min=dt_min,
        return_tissues=False, n_euler_steps=n_euler_steps, cfg=cfg,
    )
    return res["times_hr"], res["conc_central_ng_ml"]
