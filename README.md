# DL-PBPK Hybrid

A hybrid Graph Neural Network and Physiologically Based Pharmacokinetic 
(PBPK) framework for multi-drug concentration prediction with 
explainability and uncertainty quantification.

**Manuscript:** Takayidza C, Musara M. *A Hybrid Graph Neural Network 
and Physiologically Based Pharmacokinetic Framework for Multi-Drug 
Concentration Prediction with Explainability and Uncertainty 
Quantification.* Journal of Cheminformatics (submitted 2026).

**Authors:** Clayton Takayidza, Maronge Musara  
**Institution:** Harare Institute of Technology, Harare, Zimbabwe  
**License:** MIT (see [LICENSE](LICENSE))  
**Data licensing:** See [NOTICE](NOTICE) for third-party data attribution  

---

## Installation

Requires Python 3.10 or later. Tested on Python 3.11.

```bash
git clone https://github.com/<USERNAME>/dl-pbpk-hybrid.git
cd dl-pbpk-hybrid
pip install -r requirements.txt
```

For the API and frontend stack, see Docker/Make commands below and 
`api/requirements.txt`.

## Quick reproduction of manuscript results

All experiments use `SEED = 42` for deterministic reproduction.

```bash
# 1. Generate simulated PK datasets for the 6 training drugs + ibuprofen
python -m experiments.data.download_pk_data

# 2. Featurise molecular graphs
python -m experiments.data.featurize_drugs

# 3. Train hybrid models per drug
python -m experiments.training.train_multidrug_hybrid

# 4. Evaluate (use --force to overwrite existing results)
python -m experiments.evaluation.evaluate_multidrug --help
python -m experiments.evaluation.evaluate_multidrug --force

# 5. Phase 2 baselines, ablations, statistics
python -m experiments.baselines.train_baselines
python -m experiments.baselines.correct_pbpk_realistic
python -m experiments.ablation.ablation_study
python -m experiments.statistics.significance_tests

# 6. Phase 3 safety, uncertainty, explainability
python -m experiments.safety.safety_thresholds
python -m experiments.uncertainty.monte_carlo_calibration
python -m experiments.explainability.shap_interpretation
```

Total reproduction time: approximately 6–8 hours on CPU-only hardware 
(Intel Core i7 with 16 GB RAM).

For full details, see [paper/reproducibility_checklist.md](paper/reproducibility_checklist.md).

## Data

Small data files (per-drug PK datasets, molecular graphs, ESOL, 
Lipophilicity) are committed directly. Large data files (ChEMBL-derived 
training corpus, PubChem CID-SMILES dump) are hosted externally or 
regenerated from scripts. See [data/README.md](data/README.md) for full 
details.

## Citation

If you use this code in your research, please cite the manuscript and 
optionally the software directly. See [CITATION.cff](CITATION.cff) for 
machine-readable citation metadata.

## License

- **Source code:** MIT License (see [LICENSE](LICENSE))
- **ChEMBL-derived data:** CC-BY-SA 3.0 (see [NOTICE](NOTICE))
- **ESOL, Lipophilicity:** publicly available; see NOTICE for citations

## Contact

Clayton Takayidza — takayidzaclayton@gmail.com  
ORCID: [0009-0001-8146-5875](https://orcid.org/0009-0001-8146-5875)

---

# DL-PBPK Hybrid Model

Deep-learning augmented physiologically based pharmacokinetic (PBPK) modelling platform.

## Repository Structure

```
dl-pbpk-hybrid/
├── api/                 # FastAPI backend
│   ├── app/
│   │   ├── main.py      # Endpoints (/health, /predict)
│   │   └── config.py    # Environment configuration
│   ├── tests/
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/            # Next.js 14 App Router + Tailwind + Recharts
│   ├── src/
│   │   ├── app/         # Pages: Home, Predict, Explain, Compare, Reports
│   │   ├── components/  # Sidebar, shared UI
│   │   └── lib/         # API client
│   ├── Dockerfile
│   └── package.json
├── src/                 # Model training code
│   ├── datasets/
│   │   └── theoph_loader.py      # Theophylline CSV loader
│   ├── models/
│   │   ├── ode/pk_1cpt_torch.py  # Differentiable 1-cpt ODE solver
│   │   ├── gnn/molecule_gnn.py   # Message-passing GNN encoder
│   │   └── hybrid_dl_pk.py       # MLP + ODE hybrid model
│   ├── molecules/
│   │   └── rdkit_graph.py        # SMILES -> graph (RDKit featurisation)
│   ├── training/
│   │   ├── train_hybrid_theoph.py # End-to-end hybrid training
│   │   ├── pretrain_gnn_adme.py   # GNN ADME pretraining
│   │   └── plot_predictions.py    # Observed vs predicted plots
│   └── requirements.txt           # Pinned: numpy<2, torch>=2.2, rdkit-pypi
├── data/
│   ├── raw/theoph/      # Raw Theophylline CSV
│   └── processed/theoph/# Preprocessed JSON outputs
├── scripts/
│   ├── preprocess_theoph.py       # CSV -> JSON preprocessing
│   ├── check_theoph.py            # Data verification
│   ├── prepare_adme_csv.py        # Prepare ADME pretraining CSV
│   ├── cache_adme_graphs.py       # Pre-compute molecular graphs for GNN
│   ├── train_hybrid_theoph.ps1    # One-command hybrid model training
│   ├── setup_gnn_training.ps1     # Create & validate GNN training venv
│   └── train_gnn_pretrain.ps1     # One-command GNN pretraining
├── artifacts/models/              # Saved model checkpoints and metrics
├── docs/                # Documentation
├── notebooks/           # Jupyter notebooks
├── docker-compose.yml
├── Makefile
├── requirements.txt           # Unified manuscript reproduction (Python)
└── .env.example
```

## Quick Start

### Using Docker (recommended)

```bash
# 1. Copy environment variables
cp .env.example .env

# 2. Start all services
docker-compose up --build

# 3. Open in your browser
#    Frontend:  http://localhost:3000
#    API docs:  http://localhost:8000/docs
#    API health: http://localhost:8000/health
```

### Local Development

```bash
# 1. Install dependencies
make setup

# 2. Start both API and frontend
make dev

# 3. Or start them individually
make api       # FastAPI on :8000
make frontend  # Next.js on :3000
```

## Available Make Commands

| Command          | Description                                  |
| ---------------- | -------------------------------------------- |
| `make setup`     | Install Python and Node dependencies         |
| `make api`       | Start FastAPI dev server on port 8000        |
| `make frontend`  | Start Next.js dev server on port 3000        |
| `make dev`       | Start API and frontend in parallel           |
| `make lint`      | Run ruff (Python) and next lint (TypeScript) |
| `make test`      | Run pytest for the API                       |
| `make clean`     | Remove containers, volumes, build artifacts  |

## API Endpoints

| Method | Path       | Description                                |
| ------ | ---------- | ------------------------------------------ |
| GET    | `/health`  | Health check, returns status and version    |
| POST   | `/predict` | Generate PK curve and metrics for a compound |
| GET    | `/docs`    | Interactive OpenAPI (Swagger) documentation |

### Example Predict Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "compound_name": "Compound-A",
    "dose_mg": 100,
    "weight_kg": 70,
    "route": "oral"
  }'
```

## Frontend Pages

- **Home** – Dashboard with API status and platform overview
- **Predict** – Input compound parameters and visualise PK curves
- **Explain** – Model interpretability and feature importance
- **Compare** – Side-by-side dosing scenario comparison
- **Reports** – View and download generated PK reports

## Data Preprocessing

The Theophylline PK dataset (`data/raw/theoph/theoph.csv`) is preprocessed into
JSON for downstream use by the model and API.

```bash
# Preprocess raw CSV into structured JSON
python scripts/preprocess_theoph.py

# Verify the processed output
python scripts/check_theoph.py
```

**Outputs:**

| File | Description |
| ---- | ----------- |
| `data/processed/theoph/theoph_subjects.json` | Per-subject PK records (12 subjects, 132 data points) |
| `data/processed/theoph/theoph_summary.json` | Aggregate statistics (n_subjects, time/conc ranges, means) |

> If using the project venv: `api\.venv\Scripts\python.exe scripts/preprocess_theoph.py`

## Model Training

The hybrid DL+ODE model combines an MLP (predicts CL, V, ka from subject features)
with a differentiable 1-compartment oral PK simulator (Euler ODE solver), trained
end-to-end against observed concentration-time curves.

### One-command training (PowerShell)

```powershell
.\scripts\train_hybrid_theoph.ps1
```

This creates `src\.venv`, installs dependencies, and runs training.

### Manual training

```bash
# Create venv and install deps
py -3.11 -m venv src\.venv
src\.venv\Scripts\pip install -r src\requirements.txt

# Run training
src\.venv\Scripts\python -u src/training/train_hybrid_theoph.py

# Re-generate plots from saved model
src\.venv\Scripts\python src/training/plot_predictions.py
```

### Architecture

```
Input [dose_mg, weight_kg, dose_mgkg]
  --> MLP (2x hidden layers, Tanh, 32 units)
  --> exp() mapping --> CL, V, ka  (positive, physiologically plausible)
  --> 1-compartment oral ODE (Euler, 300 steps)
  --> predicted concentration-time curve
  --> MSE loss in log-space against observed data
```

### Artifacts

Saved under `artifacts/models/hybrid_theoph_v1/`:

| File | Description |
| ---- | ----------- |
| `model.pt` | PyTorch model state dict |
| `scaler.json` | Feature normalisation parameters |
| `config.json` | Model and training hyperparameters |
| `metrics.json` | Train/val/all RMSE, MAE, loss |
| `subject_*.png` | Per-subject observed vs predicted plots |
| `predictions_overview.png` | Multi-panel comparison plot |

## GNN ADME Pretraining (Windows)

The GNN encoder is pretrained on the cleaned supervised ADME corpus
(ESOL + Lipophilicity, ~5 300 molecules) so it learns molecular
representations that transfer to the downstream PBPK task.

### One-command setup

```powershell
# 1. Create a clean training venv (Python 3.11 required)
.\scripts\setup_gnn_training.ps1 -Recreate

# 2. (Optional) Prepare the supervised corpus from raw datasets
src\.venv\Scripts\python scripts\prepare_large_adme_corpus.py

# 3. Pre-compute molecular graphs (optional, speeds up training)
src\.venv\Scripts\python scripts\cache_adme_graphs.py

# 4. Run pretraining on the processed supervised corpus
.\scripts\train_gnn_pretrain.ps1
```

### Useful flags

```powershell
# Quick test with 500 molecules and CPU-friendly defaults
.\scripts\train_gnn_pretrain.ps1 --max-samples 500 --cpu-friendly

# Rebuild the graph cache before training
.\scripts\train_gnn_pretrain.ps1 --rebuild-cache

# Point to a custom CSV (must have smiles,label columns)
.\scripts\train_gnn_pretrain.ps1 --data-csv data\raw\adme_pretrain\adme.csv

# Custom hyper-parameters
.\scripts\train_gnn_pretrain.ps1 --batch-size 32 --max-epochs 100 --patience 20
```

### Artifacts

Saved under `artifacts/models/gnn_pretrain_v1/`:

| File | Description |
| ---- | ----------- |
| `model_gnn.pt` | Supervised-pretrained GNN encoder weights |
| `scaler.json` | Label normalisation (mean, std) |
| `metrics.json` | Train/val metrics, config, epoch log |

Combined **unsupervised → supervised** transfer run saves to\n`artifacts/models/gnn_pretrain_combined_v1/` and is started with:

```powershell
src\.venv\Scripts\python src\training\pretrain_gnn_adme.py --cpu-friendly --init-weights
```

## Unsupervised GNN Pretraining (ChEMBL, sampled)

An additional self-supervised pretraining stage that trains the same GNN
encoder on unlabelled ChEMBL SMILES via **masked node feature reconstruction**
(analogous to masked language modelling in NLP).

### Step 1 — Sample the unsupervised corpus

```powershell
src\.venv\Scripts\python scripts\sample_unsupervised_corpus.py
# default: 10 000 molecules, random_state=42
# custom:  --n-samples 5000
```

Produces `data/processed/adme_pretrain/adme_unsupervised_sample_10k.csv`.

### Step 2 — Cache molecular graphs

```powershell
src\.venv\Scripts\python scripts\cache_adme_graphs.py `
    --data-csv  data\processed\adme_pretrain\adme_unsupervised_sample_10k.csv `
    --output    data\processed\adme_pretrain\adme_unsupervised_sample_10k_graphs.pt
```

### Step 3 — Run unsupervised pretraining

Smoke-test (200 molecules, CPU-friendly):

```powershell
src\.venv\Scripts\python src\training\pretrain_gnn_unsupervised.py `
    --max-samples 200 --cpu-friendly
```

Full 10k run (CPU-friendly):

```powershell
src\.venv\Scripts\python src\training\pretrain_gnn_unsupervised.py --cpu-friendly
```

### Unsupervised artifacts

Saved under `artifacts/models/gnn_pretrain_unsup_v1/`:

| File | Description |
| ---- | ----------- |
| `model_gnn.pt` | Unsupervised-pretrained GNN encoder weights |
| `config.json` | GNN architecture + training configuration |
| `metrics.json` | Train/val reconstruction loss, epoch log |

### Troubleshooting

**`_ARRAY_API not found`** or **NumPy binary incompatibility**

The RDKit wheel was compiled against NumPy 1.x. If NumPy 2.x is installed the
C extension will fail to load. Fix:

```powershell
src\.venv\Scripts\pip install "numpy>=1.24,<2"
# or recreate the venv:
.\scripts\setup_gnn_training.ps1 -Recreate
```

**`ModuleNotFoundError: No module named 'rdkit'`**

```powershell
src\.venv\Scripts\pip install rdkit-pypi
```

**`ModuleNotFoundError: No module named 'torch'`**

```powershell
src\.venv\Scripts\pip install "torch>=2.2,<3"
```

**`py -3.11` not found**

Install Python 3.11 from <https://www.python.org/downloads/> and ensure the
Windows `py` launcher is on your PATH.

## GNN Training-to-Activation Pipeline (Windows)

Complete end-to-end instructions to train the GNN encoder, fine-tune on
Theophylline, and activate the GNN model in the live API.

### Step 1 — Set up the training environment

```powershell
.\scripts\setup_gnn_training.ps1 -Recreate
```

Creates `src\.venv` with Python 3.11 and installs all pinned dependencies
(numpy<2, torch, rdkit-pypi, pandas, scikit-learn). Validates the environment.

### Step 2 — Prepare the supervised corpus (if not already done)

```powershell
src\.venv\Scripts\python scripts\prepare_large_adme_corpus.py
```

Produces `data/processed/adme_pretrain/adme_supervised.csv` (ESOL + Lipophilicity,
~5 300 rows with columns `smiles`, `label`, `task`).

### Step 3 — Pre-compute molecular graphs (optional, speeds up training)

```powershell
src\.venv\Scripts\python scripts\cache_adme_graphs.py
```

Produces `data/processed/adme_pretrain/adme_graphs.pt`.

### Step 4 — Pretrain the GNN encoder

Quick smoke-test on CPU:

```powershell
.\scripts\train_gnn_pretrain.ps1 --max-samples 500 --cpu-friendly
```

Full pretraining:

```powershell
.\scripts\train_gnn_pretrain.ps1
```

Produces `artifacts/models/gnn_pretrain_v1/` containing `model_gnn.pt`,
`scaler.json`, and `metrics.json`.

### Step 5 — Fine-tune the hybrid GNN-PBPK model on Theophylline

```powershell
src\.venv\Scripts\python src\training\finetune_gnn_pbpk_theoph.py
```

Loads pretrained GNN weights automatically if present. Produces
`artifacts/models/hybrid_gnn_pbpk_theoph_v1/` with `model.pt`, `config.json`,
`metrics.json`, and `scaler.json`.

### Step 6 — Verify all artifacts

```powershell
.\scripts\verify_gnn_pipeline.ps1
```

Reports whether both pretrained and fine-tuned models are present and whether
the API is ready to switch to `model_used="gnn"`.

### Step 7 — Activate in the API

Restart the API to pick up the new artifacts:

```powershell
docker-compose restart api
# or, for local dev:
cd api && .venv\Scripts\uvicorn app.main:app --reload
```

The API automatically detects `artifacts/models/hybrid_gnn_pbpk_theoph_v1/`
on startup. Confirm by checking the response metadata:

```json
{ "model": { "model_used": "gnn", "version": "hybrid_gnn_pbpk_theoph_v1" } }
```

If the GNN artifacts are missing, the API falls back to the MLP model
(`model_used: "mlp"`) with no behaviour change required.

## Tech Stack

- **Backend**: Python 3.11, FastAPI, SQLAlchemy, PostgreSQL
- **Frontend**: Next.js 14, TypeScript, Tailwind CSS, Recharts
- **Infrastructure**: Docker Compose, Make
