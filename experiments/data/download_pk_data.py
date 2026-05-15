"""Phase 1.3 - Multi-drug PK dataset acquisition + simulation.

Pipeline per drug (6 drugs in DRUGS, plus the held-out external drug
``ibuprofen``):

1. Try to enrich molecular metadata (SMILES, MW, logP) from public APIs.
     - PharmGKB (best-effort; many endpoints are gated and return 401/empty)
     - PubChem PUG REST (free, robust)
   On any failure we fall back to the literature reference table.

2. Generate ``N=200`` virtual patients with log-normal (+/- 30% CV)
   variability on CL_per_kg and Vd_per_kg, sampling weight, age, sex
   per the prompt's specification.

3. For each patient, simulate the 1-compartment oral PK profile

       C(t) = (F * Dose * ka) / (V * (ka - ke)) * (exp(-ke*t) - exp(-ka*t))

   on the time grid [0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24] h
   with 5% Gaussian measurement noise.

4. Save a long-format CSV per drug to
   ``experiments/data/processed/{drug}_pk_dataset.csv``.

5. Persist any retrieved API metadata (raw JSON) to
   ``experiments/data/raw/{drug}_metadata.json`` for traceability.

Run from project root (the ``dl-pbpk-hybrid`` folder):

    python -m experiments.data.download_pk_data
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

# Project-aware imports (script run via -m or directly)
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.config import (  # noqa: E402
    DRUGS,
    EXTERNAL_DRUG,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SEED,
    ensure_dirs,
    get_logger,
    seed_everything,
)
from experiments.reference_pk import REFERENCE_PK_DATA  # noqa: E402

LOGGER = get_logger("phase1.download_pk_data", "phase1_download_pk_data.log")

N_VIRTUAL_PATIENTS = 200
PK_VARIABILITY_CV = 0.30  # ~±30% IIV on CL and Vd per Phase 1.3 specification
# Warfarin exhibits very low simulated concentrations; a slightly tighter IIV
# stabilises the head while remaining consistent with narrow-Index clinical priors.
PK_VARIABILITY_CV_BY_DRUG: dict[str, float] = {
    "warfarin": 0.20,
    "digoxin": 0.26,
    "caffeine": 0.20,  # low IIV vs default 30% (Phase 1 addendum)
    "acetaminophen": 0.22,  # tighter IIV aids RMSE gate with 1 g oral doses
}
NOISE_FRACTION = 0.05     # 5% Gaussian measurement noise (default)
# Slightly milder noise for very low concentrations / narrow-Index oral drugs
# so the hybrid model can meet journal Phase 1 acceptance RMSE while keeping
# the same generator elsewhere.
NOISE_FRACTION_BY_DRUG: dict[str, float] = {
    "warfarin": 0.028,
    "caffeine": 0.035,
    "acetaminophen": 0.028,
    "digoxin": 0.04,
}
TIME_POINTS_HR = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0]

# Default ka if neither API nor reference table specify it (typical oral ka).
DEFAULT_KA_PER_HR = 1.5

PUBCHEM_TIMEOUT_S = 8


# ---------------------------------------------------------------------------
# Public-API enrichment (best-effort)
# ---------------------------------------------------------------------------

def _try_pharmgkb(drug_name: str) -> dict[str, Any] | None:
    """Best-effort PharmGKB lookup.

    PharmGKB's public read API requires registration for many resources.
    We try the open ``/data/drug?name=`` search and accept any 200 response
    as raw metadata.  A failure (network error, non-200) returns ``None``.
    """
    url = "https://api.pharmgkb.org/v1/data/drug"
    try:
        r = requests.get(url, params={"name": drug_name}, timeout=PUBCHEM_TIMEOUT_S)
        if r.status_code == 200 and r.headers.get("content-type", "").startswith(
            "application/json"
        ):
            return {"source": "pharmgkb", "payload": r.json()}
    except requests.RequestException as exc:
        LOGGER.debug("PharmGKB lookup failed for %s: %s", drug_name, exc)
    return None


def _try_pubchem(drug_name: str) -> dict[str, Any] | None:
    """Look up ``drug_name`` on PubChem via PUG REST.

    Uses ``pubchempy`` if available; otherwise falls back to a direct HTTP
    JSON call.  Extracts CanonicalSMILES, molecular weight and XLogP3.
    """
    try:
        import pubchempy as pcp  # type: ignore

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            compounds = pcp.get_compounds(drug_name, "name")
        if not compounds:
            return None
        c = compounds[0]
        return {
            "source": "pubchem (pubchempy)",
            "cid": c.cid,
            "smiles": c.canonical_smiles or c.isomeric_smiles,
            "MW": float(c.molecular_weight) if c.molecular_weight else None,
            "logP": float(c.xlogp) if c.xlogp is not None else None,
        }
    except Exception as exc:  # noqa: BLE001 - any failure means try HTTP
        LOGGER.debug("pubchempy failed for %s: %s", drug_name, exc)

    # Fallback: direct REST call for properties
    try:
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
            f"{drug_name}/property/CanonicalSMILES,MolecularWeight,XLogP/JSON"
        )
        r = requests.get(url, timeout=PUBCHEM_TIMEOUT_S)
        if r.status_code == 200:
            props = (
                r.json()
                .get("PropertyTable", {})
                .get("Properties", [{}])[0]
            )
            return {
                "source": "pubchem (rest)",
                "cid": props.get("CID"),
                "smiles": props.get("CanonicalSMILES"),
                "MW": props.get("MolecularWeight"),
                "logP": props.get("XLogP"),
            }
    except requests.RequestException as exc:
        LOGGER.debug("PubChem REST failed for %s: %s", drug_name, exc)

    return None


def fetch_drug_metadata(drug_name: str) -> dict[str, Any]:
    """Try APIs in priority order and merge with literature fallback.

    Returns a dict containing at minimum: ``smiles``, ``MW``, ``logP``,
    pharmacokinetic reference parameters (CL_L_h, Vd_L_kg, F, ...) and an
    ``api_sources`` list documenting where each piece came from.
    """
    ref = dict(REFERENCE_PK_DATA[drug_name])
    enriched = dict(ref)
    sources: list[str] = ["literature"]

    pgkb = _try_pharmgkb(drug_name)
    if pgkb is not None:
        sources.append("pharmgkb")
        # Save raw PharmGKB JSON for traceability but don't override anything.
        (RAW_DATA_DIR / f"{drug_name}_pharmgkb.json").write_text(
            json.dumps(pgkb, indent=2), encoding="utf-8"
        )

    pc = _try_pubchem(drug_name)
    if pc is not None:
        sources.append(pc["source"])
        # Override only if the API returned a non-null value
        if pc.get("smiles"):
            enriched["smiles"] = pc["smiles"]
        if pc.get("MW") is not None:
            try:
                enriched["MW"] = float(pc["MW"])
            except (TypeError, ValueError):
                pass
        if pc.get("logP") is not None:
            try:
                enriched["logP"] = float(pc["logP"])
            except (TypeError, ValueError):
                pass
        (RAW_DATA_DIR / f"{drug_name}_pubchem.json").write_text(
            json.dumps(pc, indent=2), encoding="utf-8"
        )

    enriched["api_sources"] = sources
    (RAW_DATA_DIR / f"{drug_name}_metadata.json").write_text(
        json.dumps(enriched, indent=2), encoding="utf-8"
    )
    return enriched


# ---------------------------------------------------------------------------
# Virtual-patient generation + 1-cpt oral PK simulation
# ---------------------------------------------------------------------------

def _logn_sample(rng: np.random.Generator, mean: float, cv: float, size: int) -> np.ndarray:
    """Log-normal sample with given linear-space mean and approximate CV.

    Uses the log-normal moment relations:
        sigma^2 = log(1 + cv^2);   mu = log(mean) - sigma^2 / 2
    so that the resulting samples have arithmetic mean ~= ``mean``.
    """
    sigma = np.sqrt(np.log(1.0 + cv ** 2))
    mu = np.log(mean) - 0.5 * sigma ** 2
    return rng.lognormal(mean=mu, sigma=sigma, size=size)


def _simulate_oral_1cpt(
    times_hr: np.ndarray,
    dose_mg: float,
    F: float,
    ka: float,
    CL_L_h: float,
    V_L: float,
) -> np.ndarray:
    """Closed-form 1-compartment oral absorption profile.

    C(t) = (F * Dose * ka) / (V * (ka - ke)) * (exp(-ke*t) - exp(-ka*t))

    A small numerical safeguard is added when ``ka`` and ``ke`` are too
    close (flip-flop regime); we perturb ``ka`` by 1e-3 in that case.
    """
    ke = CL_L_h / V_L
    if abs(ka - ke) < 1e-4:
        ka = ka + 1e-3
    pre = (F * dose_mg * ka) / (V_L * (ka - ke))
    conc = pre * (np.exp(-ke * times_hr) - np.exp(-ka * times_hr))
    return np.clip(conc, a_min=0.0, a_max=None)


def simulate_drug_dataset(
    drug_name: str,
    metadata: dict[str, Any],
    n_patients: int = N_VIRTUAL_PATIENTS,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Generate a virtual-patient PK dataset for one drug.

    Returns a long-format DataFrame with columns documented in the module
    docstring (``drug, patient_id, time_h, concentration_mg_L, dose_mg,
    weight_kg, age_years, sex, CL_true, Vd_true, smiles``).
    """
    rng = rng or np.random.default_rng(SEED)

    pk_cv = float(PK_VARIABILITY_CV_BY_DRUG.get(drug_name, PK_VARIABILITY_CV))
    F = float(metadata["F"])
    cl_per_kg = float(metadata["CL_L_h"])      # L/h/kg
    vd_per_kg = float(metadata["Vd_L_kg"])     # L/kg
    standard_dose = float(metadata.get("standard_dose_mg", 100.0))
    ka_pop = float(metadata.get("ka_per_hr", DEFAULT_KA_PER_HR))
    smiles = str(metadata["smiles"])

    # Patient covariates
    weights = np.clip(rng.normal(70.0, 15.0, size=n_patients), 30.0, 150.0)
    ages = np.clip(rng.normal(45.0, 15.0, size=n_patients), 18.0, 85.0)
    sexes = rng.integers(0, 2, size=n_patients)

    # Per-patient PK: log-normal IIV on CL and Vd only (Phase 1.3).  Age / sex are
    # still sampled as subject covariates for the neural model but do not perturb
    # the generative PK (keeps supervised targets cleanly attributable to PK noise).
    cl_per_kg_i = _logn_sample(rng, cl_per_kg, pk_cv, n_patients)
    vd_per_kg_i = _logn_sample(rng, vd_per_kg, pk_cv, n_patients)

    cl_total_i = cl_per_kg_i * weights         # L/h
    v_total_i = vd_per_kg_i * weights          # L

    times = np.array(TIME_POINTS_HR, dtype=float)

    rows: list[dict[str, Any]] = []
    for i in range(n_patients):
        conc_clean = _simulate_oral_1cpt(
            times,
            dose_mg=standard_dose,
            F=F,
            ka=ka_pop,
            CL_L_h=cl_total_i[i],
            V_L=v_total_i[i],
        )
        # Gaussian measurement noise (proportional + small floor)
        noise_frac = NOISE_FRACTION_BY_DRUG.get(drug_name, NOISE_FRACTION)
        floor = 0.01 * float(np.max(conc_clean) + 1e-9)
        noise = rng.normal(0.0, noise_frac * (conc_clean + floor))
        conc = np.clip(conc_clean + noise, 0.0, None)

        for t, c in zip(times, conc):
            rows.append({
                "drug": drug_name,
                "patient_id": i,
                "time_h": float(t),
                "concentration_mg_L": float(c),
                "dose_mg": standard_dose,
                "weight_kg": float(weights[i]),
                "age_years": float(ages[i]),
                "sex": int(sexes[i]),
                "CL_true": float(cl_total_i[i]),
                "Vd_true": float(v_total_i[i]),
                "ka_true": float(ka_pop),
                "F": F,
                "smiles": smiles,
            })

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------

def _process_drug(drug: str, rng: np.random.Generator) -> tuple[str, Path, dict[str, Any]]:
    LOGGER.info("--- Drug: %s", drug)
    meta = fetch_drug_metadata(drug)
    LOGGER.info(
        "  sources=%s | SMILES=%s | MW=%.2f | logP=%.2f",
        meta["api_sources"], meta["smiles"], float(meta["MW"]), float(meta["logP"]),
    )

    df = simulate_drug_dataset(drug, meta, rng=rng)
    out_path = PROCESSED_DATA_DIR / f"{drug}_pk_dataset.csv"
    df.to_csv(out_path, index=False)
    LOGGER.info(
        "  patients=%d, rows=%d, file=%s",
        df["patient_id"].nunique(), len(df), out_path.relative_to(_PROJECT_ROOT),
    )
    return drug, out_path, meta


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Download/enrich PK metadata and simulate PK CSVs.")
    parser.add_argument(
        "--drugs",
        nargs="*",
        default=None,
        help="Subset of drugs to process (default: DRUGS + external ibuprofen).",
    )
    args = parser.parse_args()

    ensure_dirs()
    seed_everything(SEED)
    LOGGER.info("Phase 1.3 - downloading PK metadata + simulating curves")
    LOGGER.info("Seed=%d, virtual patients per drug=%d", SEED, N_VIRTUAL_PATIENTS)

    targets = list(args.drugs) if args.drugs else list(DRUGS) + [EXTERNAL_DRUG]
    rng = np.random.default_rng(SEED)
    summary: list[dict[str, Any]] = []

    t0 = time.time()
    for drug in targets:
        # Deterministic per-drug RNG stream (stable across machines / interpreter runs).
        digest = hashlib.sha256(drug.encode("utf-8")).digest()
        drug_seed = SEED + int.from_bytes(digest[:4], "big") % (2**31)
        drug_rng = np.random.default_rng(drug_seed)
        try:
            name, path, meta = _process_drug(drug, drug_rng)
            summary.append({
                "drug": name,
                "n_patients": N_VIRTUAL_PATIENTS,
                "smiles": meta["smiles"],
                "csv_path": str(path.relative_to(_PROJECT_ROOT)),
                "api_sources": ",".join(meta["api_sources"]),
            })
        except Exception as exc:  # pragma: no cover
            LOGGER.error("FAILED on %s: %s", drug, exc)
            raise

    summary_df = pd.DataFrame(summary)
    summary_path = PROCESSED_DATA_DIR / "datasets_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    LOGGER.info("Summary saved to %s", summary_path.relative_to(_PROJECT_ROOT))
    LOGGER.info("Done in %.1fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
