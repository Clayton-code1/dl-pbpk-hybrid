"""GraphSHAP-lite: node-mask explainability for the multi-drug hybrid GNN encoder.

Perturbs each atom's feature vector (replaces with corpus mean node features),
recomputes predicted oral AUC (0--24 h), and reports mean |ΔAUC| over reference
patients. No model retraining; uses checkpoints under ``artifacts/models/``.

Outputs
-------
- ``experiments/results/phase3b_graph_explainability.csv``
- ``experiments/plots/graph_explainability_<drug>.png`` (RDKit 2D heatmap)
- ``experiments/plots/graph_explainability_grid.png``

Examples
--------
    python experiments/explainability/graph_shap_lite.py --dry-run
    python experiments/explainability/graph_shap_lite.py --drugs theophylline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.config import DRUGS, PLOTS_DIR, RESULTS_DIR, SEED, ensure_dirs, get_logger, seed_everything  # noqa: E402
from experiments.evaluation.evaluate_multidrug import _load_model  # noqa: E402
from experiments.reference_pk import REFERENCE_PK_DATA  # noqa: E402
from experiments.training.multidrug_utils import load_drug_dataset, load_drug_graph  # noqa: E402
from src.molecules.rdkit_graph import smiles_to_graph  # noqa: E402

LOGGER = get_logger("phase3b.graph_shap", "phase3b_graph_shap.log")
N_REF = 8


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GraphSHAP-lite node masking")
    p.add_argument("--dry-run", action="store_true", help="Parse and exit without I/O")
    p.add_argument(
        "--drugs",
        type=str,
        default="",
        help="Comma-separated drug slugs (default: all DRUGS)",
    )
    return p.parse_args()


def _auc_0_24(conc: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    mask = times <= 24.0 + 1e-6
    if mask.sum() < 2:
        return torch.tensor(0.0, device=conc.device)
    t = times[mask]
    c = conc[mask]
    dt = t[1:] - t[:-1]
    return torch.sum(0.5 * (c[1:] + c[:-1]) * dt)


def _corpus_mean_node_feature(project_root: Path) -> torch.Tensor:
    """Mean feature vector [F] across all atoms in panel graphs."""
    xs = []
    for drug in DRUGS:
        path = project_root / "experiments" / "data" / "processed" / "graphs" / f"{drug}.pt"
        if not path.exists():
            continue
        blob = torch.load(path, map_location="cpu", weights_only=True)
        xs.append(blob["x"].reshape(-1, blob["x"].shape[-1]).float())
    if not xs:
        raise FileNotFoundError("No panel graphs found under processed/graphs")
    return torch.cat(xs, dim=0).mean(dim=0)


def _load_graph_tensors(drug: str, project_root: Path) -> dict[str, torch.Tensor]:
    path = project_root / "experiments" / "data" / "processed" / "graphs" / f"{drug}.pt"
    if path.exists():
        return load_drug_graph(drug)
    LOGGER.warning("Missing %s — building graph from SMILES", path)
    smiles = REFERENCE_PK_DATA[drug]["smiles"]
    g = smiles_to_graph(smiles)
    return {"x": g["x"], "edge_index": g["edge_index"], "edge_attr": g["edge_attr"]}


def _patient_baselines(
    model: torch.nn.Module,
    graph: dict[str, torch.Tensor],
    test_recs: list,
    n_ref: int,
) -> list[float]:
    out: list[float] = []
    with torch.no_grad():
        for rec in test_recs[:n_ref]:
            conc, *_ = model(
                graph["x"],
                graph["edge_index"],
                graph["edge_attr"],
                rec.features,
                rec.times_hr,
                rec.dose_mg,
                rec.weight_kg,
                rec.f_bio,
            )
            out.append(float(_auc_0_24(conc, rec.times_hr).item()))
    return out


def _masked_auc_for_atom(
    model: torch.nn.Module,
    graph: dict[str, torch.Tensor],
    atom_idx: int,
    mean_vec: torch.Tensor,
    test_recs: list,
    n_ref: int,
) -> list[float]:
    x = graph["x"].clone().float()
    x[atom_idx] = mean_vec.to(dtype=x.dtype, device=x.device)
    out: list[float] = []
    with torch.no_grad():
        for rec in test_recs[:n_ref]:
            conc, *_ = model(
                x,
                graph["edge_index"],
                graph["edge_attr"],
                rec.features,
                rec.times_hr,
                rec.dose_mg,
                rec.weight_kg,
                rec.f_bio,
            )
            out.append(float(_auc_0_24(conc, rec.times_hr).item()))
    return out


def _draw_molecule_heatmap(
    drug: str,
    importances: np.ndarray,
    out_png: Path,
) -> None:
    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D
    except Exception as exc:  # pragma: no cover
        LOGGER.error("RDKit draw unavailable: %s", exc)
        raise

    smiles = REFERENCE_PK_DATA[drug]["smiles"]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Bad SMILES for {drug}")
    mol = Chem.AddHs(mol)
    n = mol.GetNumAtoms()
    if importances.shape[0] != n:
        LOGGER.warning("Atom count mismatch draw (%d vs %d) — skipping RDKit plot", n, importances.shape[0])
        _bar_fallback(drug, importances, out_png)
        return

    # Normalise colours
    imp = importances.astype(float)
    if imp.max() > imp.min() + 1e-12:
        imp_n = (imp - imp.min()) / (imp.max() - imp.min())
    else:
        imp_n = np.zeros_like(imp)

    # Map to RGB (white -> red)
    colours: dict[int, tuple[float, float, float]] = {}
    for i in range(n):
        t = float(imp_n[i])
        colours[i] = (1.0, 1.0 - 0.85 * t, 1.0 - 0.85 * t)

    d2d = rdMolDraw2D.MolDraw2DCairo(420, 320)
    d2d.DrawMolecule(mol, highlightAtoms=list(range(n)), highlightAtomColors=colours)
    d2d.FinishDrawing()
    out_png.write_bytes(d2d.GetDrawingText())


def _bar_fallback(drug: str, importances: np.ndarray, out_png: Path) -> None:
    idx = np.argsort(-importances)[:15]
    plt.figure(figsize=(6, 4))
    plt.bar(range(len(idx)), importances[idx])
    plt.xticks(range(len(idx)), [str(int(i)) for i in idx], rotation=45)
    plt.ylabel("|ΔAUC| mean")
    plt.title(f"Top atoms — {drug}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()


def run(drugs: list[str], project_root: Path) -> None:
    ensure_dirs()
    seed_everything(SEED)
    mean_vec = _corpus_mean_node_feature(project_root)
    csv_rows: list[str] = []

    single_imgs: list[tuple[str, np.ndarray]] = []

    for drug in drugs:
        LOGGER.info("GraphSHAP-lite: %s", drug)
        train_recs, _, test_recs, _ = load_drug_dataset(drug)
        graph = _load_graph_tensors(drug, project_root)
        model, _ = _load_model(drug)
        model.eval()

        n_ref = min(N_REF, len(test_recs))
        baselines = np.array(_patient_baselines(model, graph, test_recs, n_ref), dtype=np.float64)
        n_atoms = int(graph["x"].shape[0])
        importance = np.zeros(n_atoms, dtype=np.float64)

        for a in range(n_atoms):
            masked = np.array(
                _masked_auc_for_atom(model, graph, a, mean_vec, test_recs, n_ref),
                dtype=np.float64,
            )
            importance[a] = float(np.mean(np.abs(masked - baselines)))

        order = np.argsort(-importance)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(order) + 1)

        try:
            from rdkit import Chem

            mol = Chem.AddHs(Chem.MolFromSmiles(REFERENCE_PK_DATA[drug]["smiles"]))
            symbols = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(n_atoms)]
        except Exception:
            symbols = ["?"] * n_atoms

        for i in range(n_atoms):
            csv_rows.append(
                f"{drug},{i},{symbols[i]},{importance[i]:.8f},{int(ranks[i])}",
            )

        out_drug = PLOTS_DIR / f"graph_explainability_{drug}.png"
        try:
            _draw_molecule_heatmap(drug, importance, out_drug)
        except Exception as exc:
            LOGGER.warning("Heatmap failed for %s: %s — bar fallback", drug, exc)
            _bar_fallback(drug, importance, out_drug)
        single_imgs.append((drug, importance))
        LOGGER.info("Wrote %s", out_drug)

    (RESULTS_DIR / "phase3b_graph_explainability.csv").write_text(
        "drug,atom_index,atom_symbol,importance_score,rank\n" + "\n".join(csv_rows) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Wrote %s", RESULTS_DIR / "phase3b_graph_explainability.csv")

    # Grid of top atom importances (2x3 when six drugs; fewer columns if subset)
    n_plots = len(single_imgs)
    ncols = min(3, n_plots)
    nrows = int(np.ceil(n_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 3.2 * nrows))
    ax_list = list(np.atleast_1d(axes).ravel())
    for idx, (drug, imp) in enumerate(single_imgs):
        ax = ax_list[idx]
        top = np.argsort(-imp)[:10]
        ax.barh(range(len(top)), imp[top][::-1])
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels([str(int(t)) for t in top[::-1]], fontsize=7)
        ax.set_title(drug)
        ax.set_xlabel("mean |ΔAUC|")
    for idx in range(n_plots, len(ax_list)):
        ax_list[idx].axis("off")
    plt.tight_layout()
    grid_path = PLOTS_DIR / "graph_explainability_grid.png"
    fig.savefig(grid_path, dpi=160)
    plt.close(fig)
    LOGGER.info("Wrote %s", grid_path)


def main() -> int:
    args = _parse_args()
    drugs_arg = [d.strip() for d in args.drugs.split(",") if d.strip()]
    drugs = drugs_arg if drugs_arg else list(DRUGS)
    if args.dry_run:
        print(f"graph_shap_lite dry-run OK — would run drugs={drugs}", flush=True)
        return 0
    try:
        run(drugs, _PROJECT_ROOT)
    except Exception as exc:
        print(f"FAILED graph_shap_lite: {exc}", flush=True)
        LOGGER.exception("graph_shap_lite failed")
        raise
    print(
        f"graph_shap_lite OK — CSV={RESULTS_DIR / 'phase3b_graph_explainability.csv'} plots={PLOTS_DIR}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
