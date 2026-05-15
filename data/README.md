# Data Directory

This directory contains the data needed to reproduce the experiments in
the accompanying manuscript. Most files are committed directly. Three
categories of data are too large for GitHub and are hosted externally
(see "Large external data" below).

## Directory layout

```
data/
├── raw/
│   ├── adme_pretrain/         # ESOL, Lipophilicity, ChEMBL train CSVs
│   ├── reference/             # DrugBank vocabulary (small);
│   │                          # CID-SMILES NOT INCLUDED (see below)
│   └── theoph/                # Theophylline-specific raw data
└── processed/
    ├── adme_pretrain/         # Featurised ADME data (CSVs committed;
    │                          # large .pt graph caches NOT INCLUDED)
    └── (other processed subdirs)
```

## Committed data (in this repository)

The following are included directly in the git repository:

- ESOL and Lipophilicity CSVs (`data/raw/adme_pretrain/`)
- ChEMBL training CSV (`data/raw/adme_pretrain/train.csv`, ~79 MB) —
  this is a derivative under CC-BY-SA 3.0; see NOTICE for attribution.
- Per-drug PK datasets (`experiments/data/processed/<drug>_pk_dataset.csv`)
- Per-drug molecular graphs (`experiments/data/processed/graphs/*.pt`)

## Large external data (NOT in this repository)

The following files are too large for GitHub and are hosted externally:

### 1. PubChem CID-SMILES reference (~8 GB)

**Status:** Not redistributed by this project.
**Source:** Download directly from PubChem FTP:
  https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz

After download, place at: `data/raw/reference/CID-SMILES`

**Note:** The full CID-SMILES dump is NOT required to reproduce the
manuscript's reported results — it was used only for early exploratory
analysis. If you only want to reproduce manuscript results, you can skip
this download.

### 2. Unsupervised ADME training corpus (~108 MB CSV + ~110 MB graph cache)

**Status:** Hosted on Zenodo (DOI to be added upon manuscript acceptance).
**Files:**
  - `data/processed/adme_pretrain/adme_unsupervised.csv`
  - `data/processed/adme_pretrain/adme_unsupervised_sample_10k_graphs.pt`

These files can also be regenerated from `data/raw/adme_pretrain/train.csv`
by running:

```bash
python -m experiments.data.featurize_drugs
# (or the appropriate ADME pretraining preparation script;
#  see paper/reproducibility_checklist.md)
```

## Regenerating all data from scratch

If you prefer to regenerate everything rather than download large files:

```bash
# 1. Per-drug PK datasets (requires internet for PubChem API enrichment)
python -m experiments.data.download_pk_data

# 2. Molecular graph featurisation
python -m experiments.data.featurize_drugs
```

This should complete in approximately 10-20 minutes on commodity hardware.

## License notes

See the project NOTICE file for full attribution and licensing of
third-party data sources. In brief:
- ChEMBL-derived data: CC-BY-SA 3.0 — any redistribution must preserve
  the same license.
- ESOL, Lipophilicity: publicly available, academic use unrestricted.
- DrugBank vocabulary: subject to DrugBank Academic License — verify
  current terms before any redistribution.
- PubChem: public domain.

The source code in this repository is under MIT License (see LICENSE).
