"""Literature-validated reference pharmacokinetic parameters per drug.

These values are used as fallback population means when public APIs do not
return enough information, and as the *ground truth* generators for the
simulated PK datasets used in Phase 1 training.

All clearance and volume values are expressed *per kg* so virtual patients of
different weights produce realistic absolute parameters
(``CL_total = CL_L_h_per_kg * weight_kg``).
"""

from __future__ import annotations

from typing import TypedDict


class ReferencePK(TypedDict):
    smiles: str
    MW: float
    logP: float
    CL_L_h: float          # L/h/kg (population mean)
    Vd_L_kg: float         # L/kg
    t_half_h: float        # informational only
    F: float               # bioavailability fraction
    Cmax_mg_L: float       # informational reference
    AUC_mg_h_L: float      # informational reference
    therapeutic_min_mg_L: float
    therapeutic_max_mg_L: float
    standard_dose_mg: float  # clinically-typical adult oral dose
    reference: str


REFERENCE_PK_DATA: dict[str, ReferencePK] = {
    "theophylline": {
        "smiles": "Cn1cnc2c1c(=O)[nH]c(=O)n2C",
        "MW": 180.16, "logP": -0.02,
        "CL_L_h": 0.048,
        "Vd_L_kg": 0.5,
        "t_half_h": 8.0,
        "F": 1.0,
        "Cmax_mg_L": 10.0,
        "AUC_mg_h_L": 80.0,
        "therapeutic_min_mg_L": 5.0,
        "therapeutic_max_mg_L": 15.0,
        "standard_dose_mg": 200.0,
        "reference": "Hendeles & Weinberger, 1982",
    },
    "warfarin": {
        "smiles": "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
        "MW": 308.33, "logP": 2.7,
        "CL_L_h": 0.0022,
        "Vd_L_kg": 0.14,
        "t_half_h": 40.0,
        "F": 0.99,
        "Cmax_mg_L": 2.5,
        "AUC_mg_h_L": 100.0,
        "therapeutic_min_mg_L": 0.5,
        "therapeutic_max_mg_L": 3.0,
        "standard_dose_mg": 5.0,
        "reference": "Holford, 1986",
    },
    "midazolam": {
        "smiles": "Clc1ccc2c(c1)C(c1cccnc1F)=NCC(=O)N2C",
        "MW": 325.77, "logP": 3.89,
        "CL_L_h": 0.42,
        "Vd_L_kg": 1.1,
        "t_half_h": 2.5,
        "F": 0.44,
        "Cmax_mg_L": 0.12,
        "AUC_mg_h_L": 0.3,
        "therapeutic_min_mg_L": 0.04,
        "therapeutic_max_mg_L": 0.3,
        "standard_dose_mg": 7.5,
        "reference": "Smith et al., 1981",
    },
    "caffeine": {
        "smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
        "MW": 194.19, "logP": -0.07,
        "CL_L_h": 0.078,
        "Vd_L_kg": 0.6,
        "t_half_h": 5.0,
        "F": 1.0,
        "Cmax_mg_L": 8.0,
        "AUC_mg_h_L": 40.0,
        "therapeutic_min_mg_L": 5.0,
        "therapeutic_max_mg_L": 20.0,
        "standard_dose_mg": 200.0,
        "reference": "Arnaud, 1993",
    },
    "acetaminophen": {
        "smiles": "CC(=O)Nc1ccc(O)cc1",
        "MW": 151.16, "logP": 0.49,
        "CL_L_h": 0.3,
        "Vd_L_kg": 0.95,
        "t_half_h": 2.0,
        "F": 0.85,
        "Cmax_mg_L": 15.0,
        "AUC_mg_h_L": 30.0,
        "therapeutic_min_mg_L": 5.0,
        "therapeutic_max_mg_L": 20.0,
        "standard_dose_mg": 1000.0,
        "reference": "Prescott, 1980",
    },
    "digoxin": {
        "smiles": (
            "O=C1OC(CC1)(C1CCC2(O)C3(CCC4C3(C)CCC4(O)C=C)CCC12)"
            "C1OC(OC2C(O)COC2OC2CC(O)CC(O2)C2=CC(=O)OC2)CC1O"
        ),
        "MW": 780.94, "logP": 1.26,
        "CL_L_h": 0.12,
        "Vd_L_kg": 7.3,
        "t_half_h": 39.0,
        "F": 0.7,
        "Cmax_mg_L": 0.002,
        "AUC_mg_h_L": 0.015,
        "therapeutic_min_mg_L": 0.0005,
        "therapeutic_max_mg_L": 0.002,
        "standard_dose_mg": 0.25,
        "reference": "Reuning et al., 1973",
    },
    # External validation drug (Phase 2.4) - never used in training.
    "ibuprofen": {
        "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "MW": 206.28, "logP": 3.97,
        "CL_L_h": 0.035,
        "Vd_L_kg": 0.15,
        "t_half_h": 2.0,
        "F": 1.0,
        "Cmax_mg_L": 45.0,
        "AUC_mg_h_L": 90.0,
        "therapeutic_min_mg_L": 10.0,
        "therapeutic_max_mg_L": 50.0,
        "standard_dose_mg": 400.0,
        "reference": "Greenblatt & Koch-Weser, 1975",
    },
}
