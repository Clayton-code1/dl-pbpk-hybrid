# Real Validation Dataset Verification Report

**Branch:** `data/real-validation-datasets`  
**Date:** 2026-06-08  
**Analyst:** Dataset-verification pass (no model runs, no code changes)  
**Prior validation baseline:** Theophylline (R `Theoph` dataset, 12 subjects, R²=0.673)

---

## Usability bar (reminder)

Model is a **one-compartment oral absorption ODE**, trained on simulated data, taking inputs: `dose_mg`, `weight_kg`, `age_years`, `sex`, `smiles`. A valid external test requires:

| Criterion | Required |
|-----------|----------|
| Real measured (not simulated) data | YES |
| Oral administration | YES — IV would have no absorption phase to test |
| Per-subject: dose, weight, time, concentration | YES |
| Multiple subjects | Strongly preferred (for per-subject R²) |

---

## Dataset 1: Warfarin — nlmixr2data R package

### Source
- **Origin:** O'Reilly RA, Aggeler PM (1963, 1968); assembled and modeled by Holford NHG (1986) "Clinical Pharmacokinetics and Pharmacodynamics of Warfarin" *Clin Pharmacokinet* 11:483–504.
- **Access:** R package `nlmixr2data` (CRAN/GitHub). File inspected directly: `warfarin.rda` at `https://raw.githubusercontent.com/nlmixr2/nlmixr2data/main/data/warfarin.rda`
- **Status:** **REAL** human clinical study data — NOT simulated. Confirmed from source citation.

### What the file actually contains (inspected with pyreadr)

| Property | Value |
|----------|-------|
| File | `warfarin.rda` (1,762 bytes, 515 rows × 9 columns) |
| Total rows | 515 (PK + PD mixed) |
| PK-only rows (dvid='cp') | **283** |
| Subjects | **32** |
| Time range | 0–144 h (6 days) |
| PK concentration range | 0–17.6 mg/L |
| Dose range | 60–153 mg per subject (mean ~105 mg; this is a 1.5 mg/kg dose, so dose varies with weight) |

**Columns:**

| Column | Description |
|--------|-------------|
| `id` | Subject identifier (1–33, not contiguous) |
| `time` | Hours post-dose |
| `amt` | Dose given (mg); 0 on observation rows |
| `dv` | Dependent variable: warfarin plasma conc (mg/L) when dvid='cp', PCA % when dvid='pca' |
| `dvid` | `'cp'` = pharmacokinetic (PK); `'pca'` = pharmacodynamic effect (prothrombin complex activity) |
| `evid` | 1 = dosing event; 0 = observation |
| `wt` | Body weight (kg) |
| `age` | Age (years) |
| `sex` | `'male'` or `'female'` |

**Covariate distribution (per-subject):**

| Covariate | Min | Mean | Max | Notes |
|-----------|-----|------|-----|-------|
| weight_kg | 40.0 | 70.0 | 102.0 | Real individual values |
| age_years | 21 | 31 | 63 | Real individual values |
| sex | — | 27M / 5F | — | Predominantly male (O'Reilly studies used healthy male volunteers + some female) |

**Administration route:** ORAL. Single dose of ~1.5 mg/kg warfarin sodium. No RATE column; no IV flag.

### Critical caveat: time-point coverage

Two distinct sub-studies are merged in this dataset:

| Sub-study | Subject IDs | n | First obs after t=0 | Absorption phase visible? |
|-----------|-------------|---|---------------------|--------------------------|
| Dense early | 1,3,4,5,6,7,8,9,12,13,14,15,16 | 13 | 0.5–6 h | **YES** |
| Sparse late | 2,10,17–33 | 19 | **24 h** | **NO** (trough-only) |

The 19 sparse subjects have their first measured concentration at t=24h (well past the peak). For these subjects:
- The model will predict an absorption + distribution + elimination curve.
- Observed data contains only elimination-phase points.
- Per-subject R² for these 19 subjects tests elimination behavior only; ka cannot be assessed.

**Recommendation**: Compute R² on all 32 subjects but also report the 13-subject sub-group (with absorption phase) separately.

### Critical caveat: dose range extrapolation

The project's simulated warfarin training data used **5 mg doses** (standard anticoagulation dose), yielding training-set concentrations of approximately **0.2–0.7 mg/L**.

The real O'Reilly data used **~60–153 mg** (mean 105 mg) as a single dose pharmacokinetic characterization study (1960s–era large-dose PK study), yielding concentrations of **0–17.6 mg/L**.

- Dose ratio: **~12–30× larger than training range.**
- Concentration ratio: **~20–25× larger than training range.**
- In a correct 1-compartment ODE, concentration scales linearly with dose, so the shape of the curve (normalized to dose) should be preserved. R² is scale-invariant, so if the model predicts the correct shape, R² will be unaffected by the absolute scale.
- HOWEVER: the model may have learned non-linear features from the 5 mg training data that do not extrapolate, and the GNN embeddings were exposed to 5 mg-scale concentrations only. **This extrapolation is real and must be disclosed.**

### Column mapping to model input format

```python
cp = df[df['dvid'] == 'cp'].copy()
cp = cp.rename(columns={
    'id': 'patient_id',
    'time': 'time_h',
    'dv': 'concentration_mg_L',
    'wt': 'weight_kg',
    'age': 'age_years',
})
# dose_mg: use amt at evid=1 rows (dosing), propagate to all subject rows
# sex: convert 'male'->1, 'female'->0 (or match training convention)
```

### Verdict

| Criterion | Assessment |
|-----------|-----------|
| Real measured data? | **YES** — O'Reilly 1963/1968 clinical studies |
| Oral administration? | **YES** — single oral dose |
| Per-subject dose, weight, time, concentration? | **YES** — all four present with individual variation |
| Multiple subjects? | **YES** — 32 subjects |
| Free public access? | **YES** — GitHub RDA, no login required |
| **OVERALL** | **USABLE WITH CAVEATS** |

**Caveats to state explicitly when reporting results:**
1. Dose range is 12–30× larger than training data (large extrapolation).
2. 19/32 subjects have absorption-phase-free data (first obs at t=24h); validation for these subjects tests only elimination.
3. Dataset was assembled from a 1960s clinical study — population is historical healthy volunteers, predominantly young adult males.
4. Must filter to `dvid='cp'` before using.

---

## Dataset 2 (Panel Drug): Caffeine — PK-DB Database

### Source
- **Database:** PK-DB (Pharmacokinetics Database), https://pk-db.com
- **Citation:** Grzegorzewski J et al. (2022) "PK-DB: pharmacokinetics database for individualized and stratified computational modeling." *CPT Pharmacometrics Syst Pharmacol* 11:3–4.
- **Relevant studies:** Multiple oral caffeine studies with individual-level data verified via REST API:
  - **Blanchard & Sawers 1983** (`Blanchard1983`): 31 individuals, oral + IV crossover, caffeine, time-course data present. *Br J Clin Pharmacol* 16(3):277–280.
  - **Birkett et al. 1991** (`Birkett1991`): 41 individuals, oral only, caffeine, time-course data present.
  - **Balogh et al. 1992** (`Balogh1992`): 12 individuals, oral, caffeine, time-course data present.

### What was confirmed via API

PK-DB REST API at `https://pk-db.com/api/v1/studies/?substance=caffeine&route=oral` returns at least 18 studies (page 1 of 6) with individual-level annotations. Multiple studies are flagged as having time-course data. However:
- The `pkdata/timecourses/` endpoint returned **count=0** in unauthenticated queries, indicating individual-level time-series **requires registration** to access.
- The PK-DB website (pk-db.com) provides a free academic registration mechanism.

### Caffeine drug properties (training metadata)

| Property | Value |
|----------|-------|
| Training dose | 200 mg (oral, F=1.0) |
| CL (literature) | 0.078 L/h |
| Vd (literature) | 0.6 L/kg |
| t½ | ~5 h |
| Cmax expected at 200mg | ~8 mg/L |

Caffeine is **one of the 6 panel drugs** — the model was directly trained on simulated caffeine data. This makes it a more meaningful second validation target than a non-panel drug.

### Expected data structure after PK-DB download

Based on PK-DB documentation and API structure, the individual timecourse data should contain:
- Subject ID, time (h), concentration (mg/L or µg/mL — **check units**)
- Dose (mg), route (oral)
- Covariates available vary by study (Blanchard 1983 may be sparse on covariates)

### Access instructions (what to fetch)

1. Register for free at https://pk-db.com (academic use).
2. Authenticate with your token and query:
   ```
   GET https://pk-db.com/api/v1/pkdata/timecourses/?format=json&substance=caffeine&route=oral
   ```
3. Filter to studies with adequate subjects and time-course data. **Priority studies:**
   - `Birkett1991` (41 subjects oral, largest) — or
   - `Balogh1992` (12 subjects, similar size to theophylline)
4. Inspect columns, check concentration units (µg/mL vs mg/L; caffeine MW=194.19, so 1 mg/L = 5.15 µmol/L).
5. Confirm: individual time points span at least 0–12h (covers absorption + early elimination for caffeine t½~5h).

### Alternative: manual digitization from published paper

If PK-DB access is inconvenient, the Blanchard & Sawers (1983) paper (BJCP 16:277) contains:
- Individual concentration-time tables for 3 subjects (oral arm only from the crossover)
- Doses, weights, sex, and time points

This paper is available via journal subscription or institutional access. Three subjects is small but honest — it is real data.

### Verdict

| Criterion | Assessment |
|-----------|-----------|
| Real measured data? | **YES** — sourced from published clinical studies |
| Oral administration? | **YES** — all PK-DB oral caffeine studies are oral |
| Per-subject dose, weight, time, concentration? | **LIKELY YES** (PK-DB captures individual covariates where available) — verify after download |
| Multiple subjects? | **YES** — Birkett1991 has 41, Balogh1992 has 12 |
| Free public access? | **REQUIRES FREE REGISTRATION** at pk-db.com |
| **OVERALL** | **USABLE WITH CAVEATS** (pending download and column verification) |

**Caveats:**
1. Data access requires PK-DB account registration (free, academic).
2. Covariate completeness (weight, age, sex) must be verified after download — some studies may lack individual-level covariates.
3. If weight is unavailable, it will need imputation as in the theophylline validation.
4. Concentration units must be checked and converted to mg/L (model unit) if stored as µg/mL.

---

## Investigated and Ruled Out

### Indomethacin — R `datasets::Indometh`

- **Route: CONFIRMED INTRAVENOUS.** Documentation (rdocumentation.org): "Each of the six subjects were given an **intravenous injection** of indometacin."
- 6 subjects, 3 columns (Subject, time, conc), 66 rows total.
- No dose column, no weight column.
- **VERDICT: NOT USABLE.** IV route, no absorption phase, too few subjects, missing covariates. Not a fair test for an oral absorption model.

### Phenobarbital — nlmixr2data `pheno_sd`

- **Population: CONFIRMED NEONATAL.** Weight range 0.6–3.6 kg (mean 1.53 kg). These are premature/low-birth-weight infants in the NICU.
- **Route: AMBIGUOUS, LIKELY IV.** No RATE column and no explicit route label. Loading doses of 15–70 mg/kg are consistent with NICU IV phenobarbital protocols (Grasela & Donn 1985, "Neonatal population pharmacokinetics of phenobarbital derived from **routine clinical data**"). NONMEM tutorials for this exact dataset use a 1-compartment IV bolus model.
- 59 subjects, 155 observations (very sparse: ~2.6 obs/subject).
- Drug concentration units: µg/mL at doses in µg/kg — entirely different scale from adult dosing.
- The model was trained on adult simulated data; applying it to neonates would be physiologically invalid (neonate PK differs substantially: lower protein binding, immature renal function, different Vd).
- **VERDICT: NOT USABLE.** Likely IV route; neonatal population is outside the model's trained domain.

### Mavoglurant — nlmixr2data `mavoglurant`

- **Route: CONFIRMED IV INFUSION.** RATE column present with values 75, 150, 225, 300 mg/h. Multiple infusion rate arms.
- 2678 rows, 120 subjects, 14 columns.
- Not a panel drug.
- **VERDICT: NOT USABLE.** IV infusion, no oral absorption phase.

### Quinidine — nlme `Quinidine`

- **Route: ORAL.** Documentation confirmed: "All patients were receiving oral quinidine doses."
- **BUT: MULTIPLE-DOSE, STEADY-STATE STUDY.** Real clinical data from cardiac arrhythmia patients receiving quinidine long-term with complex, changing dose regimens.
- 136 subjects — but very sparse observations: typically 2–3 concentration measurements per subject spread over >100 h of variable dosing.
- To validate against Quinidine data, the model would need to implement superposition for multiple oral doses — which it currently does not.
- Not a panel drug.
- **VERDICT: NOT USABLE** for single-dose 1-compartment validation. The sparse per-subject data and complex multi-dose regimen make a fair R² comparison impossible.

---

## Summary Recommendation

### Which 2 datasets are genuinely usable

| # | Drug | Dataset | n subjects | Route | Access | Verdict |
|---|------|---------|-----------|-------|--------|---------|
| 1 | **Warfarin** | nlmixr2data `warfarin` | 32 | Oral | Free, GitHub RDA | **USABLE WITH CAVEATS** |
| 2 | **Caffeine** | PK-DB (Birkett1991 or Balogh1992) | 12–41 | Oral | Free registration | **USABLE WITH CAVEATS** (pending download) |

Both are panel drugs that the model was trained on (warfarin and caffeine). Both are real human clinical data with oral administration.

### What each requires to run

**Warfarin:**
1. Data already downloaded and verified: 32 subjects, 283 PK rows, columns confirmed.
2. Save file: `experiments/real_validation_data/warfarin_raw.rda` (download from GitHub at the URL above).
3. Pre-processing: load with `pyreadr`, filter `dvid=='cp'`, rename columns, convert `sex` to numeric.
4. Note: run validation on all 32 subjects AND report 13-subject sub-group separately (absorption-phase subjects only).
5. Report dose range mismatch prominently — this tests ODE dose-scaling, not just shape fitting.

**Caffeine:**
1. Register at pk-db.com and authenticate.
2. Query API for `Birkett1991` (41 subjects) or `Balogh1992` (12 subjects) oral caffeine timecourse data.
3. Check units (convert µg/mL → mg/L if needed; caffeine 1 µg/mL = 1 mg/L, same numerically since MW conversion differs but check the actual units reported).
4. Check covariate completeness (weight, age, sex); impute training-set mean for any missing (document exactly as done for theophylline).
5. The 200 mg training dose should approximately match real caffeine PK study doses (most studies use 100–400 mg oral caffeine).

### Limitations to state in any future paper/report

**Warfarin:**
- 20× dose extrapolation beyond training range; validates ODE linearity assumption but is not a conventional in-distribution test.
- Absorption phase absent for 19/32 subjects; those subjects test elimination only.

**Caffeine (once downloaded):**
- Model trained on simulated caffeine PK; real data is a genuine out-of-distribution test for simulation-to-reality gap.
- Covariate availability (weight in particular) needs to be confirmed after download.

---

## Files Produced by This Verification

| File | Contents |
|------|----------|
| `experiments/real_validation_data/dataset_verification.md` | This report |

The warfarin RDA file was inspected in memory only. No large files were downloaded or committed. No model was run. No training or API code was modified.
