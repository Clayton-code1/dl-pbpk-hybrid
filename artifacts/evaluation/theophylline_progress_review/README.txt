================================================================================
Theophylline evaluation – progress review outputs
================================================================================

What the script does
--------------------
The evaluation script (run_theophylline_eval.py) calls the live DL-PBPK API
/predict/v2 with five fixed Theophylline dose scenarios, collects PK metrics
and safety for each run, and writes the results into CSV, Markdown, and JSON
in this folder. It is for progress review only and does not change the API
or any backend/frontend logic.

Which API endpoint it uses
--------------------------
  http://127.0.0.1:8000/predict/v2

The backend must be running (e.g. uvicorn app.main:app --host 0.0.0.0 --port 8000)
before running the script.

Which 5 doses were tested
--------------------------
  - Theophylline, 70 kg,  50 mg, oral
  - Theophylline, 70 kg, 100 mg, oral
  - Theophylline, 70 kg, 150 mg, oral
  - Theophylline, 70 kg, 200 mg, oral
  - Theophylline, 70 kg, 300 mg, oral

Where outputs are saved
-----------------------
  - theophylline_results.csv   (table for spreadsheets)
  - theophylline_results.md    (Markdown table for reports)
  - theophylline_results.json  (raw rows for tooling)

All files are written into this directory:
  artifacts/evaluation/theophylline_progress_review/

Exact command to run the script from the repo root
--------------------------------------------------
From the repository root (dl-pbpk-hybrid):

  python scripts/evaluation/run_theophylline_eval.py

On Windows, from the same root:

  py scripts/evaluation/run_theophylline_eval.py

Or using the API venv:

  api\.venv\Scripts\python.exe scripts/evaluation/run_theophylline_eval.py

Requirements: requests, pandas (pip install requests pandas)

================================================================================
