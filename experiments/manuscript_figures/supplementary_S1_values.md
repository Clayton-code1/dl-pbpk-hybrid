# Supplementary Table S1 — Drug-Specific Simulation Parameters

All values extracted verbatim from the project source files listed below.
No values have been estimated, inferred, or supplied from external knowledge.
Units are reported exactly as defined in the code.

**Source files:**
- `experiments/reference_pk.py` — population PK parameters and literature sources
- `experiments/data/download_pk_data.py` — simulation constants (IIV, noise, time grid)
- `experiments/config.py` — global seed and drug list
- `experiments/training/multidrug_utils.py` — patient split logic

---

## Per-drug parameter table

| Drug | Role | Dose (mg) | CL (L/h/kg) | Vd (L/kg) | ka (1/h) | IIV CV | Noise fraction | Literature source |
|------|------|----------:|------------:|----------:|----------:|-------:|---------------:|-------------------|
| Theophylline | Training | 200.0 | 0.048 | 0.50 | 1.5 | 0.30 | 0.05 | Hendeles & Weinberger, 1982 |
| Warfarin | Training | 5.0 | 0.0022 | 0.14 | 1.5 | 0.20 | 0.028 | Holford, 1986 |
| Midazolam | Training | 7.5 | 0.42 | 1.10 | 1.5 | 0.30 | 0.05 | Smith et al., 1981 |
| Caffeine | Training | 200.0 | 0.078 | 0.60 | 1.5 | 0.20 | 0.035 | Arnaud, 1993 |
| Acetaminophen | Training | 1000.0 | 0.30 | 0.95 | 1.5 | 0.22 | 0.028 | Prescott, 1980 |
| Digoxin | Training | 0.25 | 0.12 | 7.30 | 1.5 | 0.26 | 0.04 | Reuning et al., 1973 |
| Ibuprofen | Withheld (zero-shot) | 400.0 | 0.035 | 0.15 | 1.5 | 0.30 | 0.05 | Greenblatt & Koch-Weser, 1975 |

---

## File:line citations for every cell

### Dose (`standard_dose_mg`) — `experiments/reference_pk.py`

| Drug | Value | File:line |
|------|------:|-----------|
| Theophylline | 200.0 mg | `reference_pk.py:45` |
| Warfarin | 5.0 mg | `reference_pk.py:59` |
| Midazolam | 7.5 mg | `reference_pk.py:73` |
| Caffeine | 200.0 mg | `reference_pk.py:87` |
| Acetaminophen | 1000.0 mg | `reference_pk.py:101` |
| Digoxin | 0.25 mg | `reference_pk.py:118` |
| Ibuprofen | 400.0 mg | `reference_pk.py:133` |

### Clearance (`CL_L_h`) — `experiments/reference_pk.py`

Field docstring (line 21): `CL_L_h: float  # L/h/kg (population mean)`
All values are **per-kilogram** (L/h/kg). Absolute CL is computed as `CL_total = CL_L_h × weight_kg` in `download_pk_data.py:287`.

| Drug | Value | File:line |
|------|------:|-----------|
| Theophylline | 0.048 L/h/kg | `reference_pk.py:37` |
| Warfarin | 0.0022 L/h/kg | `reference_pk.py:51` |
| Midazolam | 0.42 L/h/kg | `reference_pk.py:65` |
| Caffeine | 0.078 L/h/kg | `reference_pk.py:79` |
| Acetaminophen | 0.30 L/h/kg | `reference_pk.py:93` |
| Digoxin | 0.12 L/h/kg | `reference_pk.py:110` |
| Ibuprofen | 0.035 L/h/kg | `reference_pk.py:125` |

### Volume of distribution (`Vd_L_kg`) — `experiments/reference_pk.py`

Field docstring (line 22): `Vd_L_kg: float  # L/kg`
All values are **per-kilogram** (L/kg). Absolute Vd computed as `V_total = Vd_L_kg × weight_kg` in `download_pk_data.py:288`.

| Drug | Value | File:line |
|------|------:|-----------|
| Theophylline | 0.50 L/kg | `reference_pk.py:38` |
| Warfarin | 0.14 L/kg | `reference_pk.py:52` |
| Midazolam | 1.10 L/kg | `reference_pk.py:66` |
| Caffeine | 0.60 L/kg | `reference_pk.py:80` |
| Acetaminophen | 0.95 L/kg | `reference_pk.py:94` |
| Digoxin | 7.30 L/kg | `reference_pk.py:111` |
| Ibuprofen | 0.15 L/kg | `reference_pk.py:126` |

### Absorption rate constant (ka) — `experiments/data/download_pk_data.py`

**No per-drug `ka` values are defined in `reference_pk.py`.** The `ReferencePK` TypedDict (lines 17–30) does not include a `ka_per_hr` field, and no drug entry in `REFERENCE_PK_DATA` contains a `ka_per_hr` key. All seven drugs therefore use the module-level default:

```
download_pk_data.py:90    DEFAULT_KA_PER_HR = 1.5
download_pk_data.py:273   ka_pop = float(metadata.get("ka_per_hr", DEFAULT_KA_PER_HR))
```

| Drug | ka value | Source |
|------|----------:|--------|
| All 7 drugs | 1.5 /h | `download_pk_data.py:90` (default) |

Ka is applied as a **fixed population value with no inter-individual variability** — only CL and Vd receive log-normal IIV (see `download_pk_data.py:284–285`).

### Inter-individual variability (IIV) — `experiments/data/download_pk_data.py`

**Distribution:** log-normal, implemented via `_logn_sample()` (`download_pk_data.py:219–228`).
Parameterisation: `σ² = log(1 + CV²)`, `μ = log(mean) − σ²/2`, ensuring arithmetic mean ≈ population value.
IIV applies to CL and Vd only; ka has no IIV.

**Default CV** (`download_pk_data.py:68`):
```python
PK_VARIABILITY_CV = 0.30  # ~±30% IIV on CL and Vd per Phase 1.3 specification
```

**Per-drug overrides** (`download_pk_data.py:71–76`):
```python
PK_VARIABILITY_CV_BY_DRUG: dict[str, float] = {
    "warfarin": 0.20,       # line 72
    "digoxin": 0.26,        # line 73
    "caffeine": 0.20,       # line 74 — "low IIV vs default 30% (Phase 1 addendum)"
    "acetaminophen": 0.22,  # line 75 — "tighter IIV aids RMSE gate with 1 g oral doses"
}
```

| Drug | CV | Source |
|------|---:|--------|
| Theophylline | 0.30 | `download_pk_data.py:68` (default) |
| Warfarin | 0.20 | `download_pk_data.py:72` |
| Midazolam | 0.30 | `download_pk_data.py:68` (default) |
| Caffeine | 0.20 | `download_pk_data.py:74` |
| Acetaminophen | 0.22 | `download_pk_data.py:75` |
| Digoxin | 0.26 | `download_pk_data.py:73` |
| Ibuprofen | 0.30 | `download_pk_data.py:68` (default; ibuprofen not in `PK_VARIABILITY_CV_BY_DRUG`) |

### Observation noise — `experiments/data/download_pk_data.py`

**Type:** Additive Gaussian, proportional to instantaneous concentration plus a small floor term.
Implementation (`download_pk_data.py:303–306`):
```python
noise_frac = NOISE_FRACTION_BY_DRUG.get(drug_name, NOISE_FRACTION)
floor = 0.01 * float(np.max(conc_clean) + 1e-9)
noise = rng.normal(0.0, noise_frac * (conc_clean + floor))
```

**Default noise fraction** (`download_pk_data.py:77`):
```python
NOISE_FRACTION = 0.05  # 5% Gaussian measurement noise (default)
```

**Per-drug overrides** (`download_pk_data.py:81–86`):
```python
NOISE_FRACTION_BY_DRUG: dict[str, float] = {
    "warfarin": 0.028,       # line 82
    "caffeine": 0.035,       # line 83
    "acetaminophen": 0.028,  # line 84
    "digoxin": 0.04,         # line 85
}
```

| Drug | Noise fraction | Source |
|------|---------------:|--------|
| Theophylline | 0.05 | `download_pk_data.py:77` (default) |
| Warfarin | 0.028 | `download_pk_data.py:82` |
| Midazolam | 0.05 | `download_pk_data.py:77` (default) |
| Caffeine | 0.035 | `download_pk_data.py:83` |
| Acetaminophen | 0.028 | `download_pk_data.py:84` |
| Digoxin | 0.04 | `download_pk_data.py:85` |
| Ibuprofen | 0.05 | `download_pk_data.py:77` (default; ibuprofen not in `NOISE_FRACTION_BY_DRUG`) |

### Literature source (`reference`) — `experiments/reference_pk.py`

| Drug | Source string in code | File:line |
|------|----------------------|-----------|
| Theophylline | `"Hendeles & Weinberger, 1982"` | `reference_pk.py:46` |
| Warfarin | `"Holford, 1986"` | `reference_pk.py:60` |
| Midazolam | `"Smith et al., 1981"` | `reference_pk.py:74` |
| Caffeine | `"Arnaud, 1993"` | `reference_pk.py:88` |
| Acetaminophen | `"Prescott, 1980"` | `reference_pk.py:102` |
| Digoxin | `"Reuning et al., 1973"` | `reference_pk.py:119` |
| Ibuprofen | `"Greenblatt & Koch-Weser, 1975"` | `reference_pk.py:134` |

Note: these are the literal strings in the code. They are author–year strings, not full bibliographic entries. Full citations must be supplied by the authors; see "Items to resolve manually" below.

---

## Global simulation constants (same for all drugs)

| Parameter | Value | File:line |
|-----------|-------|-----------|
| Virtual patients per drug | 200 | `download_pk_data.py:67` — `N_VIRTUAL_PATIENTS = 200` |
| Sampling time points | 13 | `download_pk_data.py:87` — `TIME_POINTS_HR` list |
| Time point values (h) | [0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0] | `download_pk_data.py:87` |
| Train / val / test split | 80 / 10 / 10 (patients) | `multidrug_utils.py:169–170` — `train_frac=0.8, val_frac=0.1` |
| Train patients per drug | 160 | derived: round(0.8 × 200) |
| Val patients per drug | 20 | derived: round(0.1 × 200) |
| Test patients per drug | 20 | derived: 200 − 160 − 20 |
| Global random seed | 42 | `config.py:20` — `SEED = 42` |
| Per-drug split seed | SEED + SHA256(drug_name)[:4] % 2³¹ | `multidrug_utils.py:183–187` |
| Per-drug simulation seed | SEED + SHA256(drug_name)[:4] % 2³¹ | `download_pk_data.py:375–377` |
| Simulated PK formula | Closed-form 1-cpt oral: C(t) = (F·D·ka)/(V·(ka−ke))·(e^{−ke·t} − e^{−ka·t}) | `download_pk_data.py:241–251` |
| IIV applies to | CL and Vd only (ka fixed) | `download_pk_data.py:284–285` |

**Bioavailability (F)** is used in the simulation formula but was not requested. Reported here for completeness as it affects simulated Cmax and is drug-specific (theophylline=1.0, warfarin=0.99, midazolam=0.44, caffeine=1.0, acetaminophen=0.85, digoxin=0.70, ibuprofen=1.0; `reference_pk.py:40/54/68/82/96/113/129`). F=1.0 is used during model inference for all drugs.

---

## Patient covariate generation (same for all drugs unless noted)

| Covariate | Distribution | Parameters | File:line |
|-----------|-------------|------------|-----------|
| Weight (kg) | Normal, clipped | N(70, 15²), clipped [30, 150] | `download_pk_data.py:277` |
| Age (years) | Normal, clipped | N(45, 15²), clipped [18, 85] | `download_pk_data.py:278` |
| Sex | Uniform integer | {0, 1} | `download_pk_data.py:279` |

Note: Age and sex are sampled as covariates for the neural model input but do **not** perturb the generative PK parameters (comment at `download_pk_data.py:281–283`).

**Acetaminophen patient feature columns differ** from the other five drugs: six features are used (weight_kg, dose_mg, dose_mg_per_kg, log_dose_mg_per_kg, age_years, sex) versus five for the remaining drugs (weight_kg, dose_mg, dose_mgkg, age_years, sex). Source: `multidrug_utils.py:45–61`.

---

## Items marked NOT FOUND or NO SOURCE IN CODE

| Item | Status | Notes |
|------|--------|-------|
| Per-drug ka values | **NOT FOUND in reference_pk.py** | `ka_per_hr` is not a field in `ReferencePK` TypedDict and no drug entry contains it. All 7 drugs use `DEFAULT_KA_PER_HR = 1.5` /h (`download_pk_data.py:90`). This single value must be disclosed in the paper. |
| Full bibliographic entries for literature sources | **NO SOURCE IN CODE** | The `reference` field holds author–year strings only (e.g., `"Hendeles & Weinberger, 1982"`). Full journal, volume, pages, and DOI must be supplied by the authors from their library. |
| Ibuprofen IIV CV and noise overrides | **NOT FOUND** | Ibuprofen is absent from `PK_VARIABILITY_CV_BY_DRUG` and `NOISE_FRACTION_BY_DRUG`; it falls through to the defaults (CV=0.30, noise=0.05). Confirm this is intentional. |
| Warfarin F in inference | **NOT IN SCOPE but noteworthy** | Training used F=0.99 (reference_pk.py:54); the model sets F=1.0 at inference time for all drugs (`multidrug_utils.py:141`). This discrepancy is present in the code and may be worth a footnote. |
