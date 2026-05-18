"""Generate Journal of Cheminformatics graphical abstract (920×300 px; PNG ≤150 KB).

V2: five-box horizontal pipeline, 15 px outer margins, pixel coordinates.
Does not modify Figure_1_architecture.*.

Window preview (optional): set GRAPHICAL_ABSTRACT_PREVIEW=1 before running the script
(matplotlib uses TkAgg; window stays ~12 s before PNG export).

Default: Agg backend (headless); PDF, SVG, and PNG are written in all cases.
"""

from __future__ import annotations

import os
import shutil
from io import BytesIO
from pathlib import Path

import matplotlib

_PREVIEW = os.environ.get("GRAPHICAL_ABSTRACT_PREVIEW", "").lower() in ("1", "true", "yes")
matplotlib.use("TkAgg" if _PREVIEW else "Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

WIDTH_PX = 920
HEIGHT_PX = 300
MARGIN_PX = 15
GAP_BOX_PX = 20

DPI = 200
MAX_PNG_BYTES = 153_600
FIGSIZE_IN = (WIDTH_PX / DPI, HEIGHT_PX / DPI)

TITLE_STR = "DL-PBPK Hybrid: Multi-Drug PK Prediction"
FOOTER_STR = "Validated across 6 drugs · External zero-shot on ibuprofen"

TITLE_FS = 12
BOX_LABEL_FS = 9
BOX_SUB_FS = 7
FOOTER_FS = 7

_INNER_LEFT = MARGIN_PX
_INNER_RIGHT = WIDTH_PX - MARGIN_PX
_INNER_BOTTOM = MARGIN_PX
_INNER_TOP = HEIGHT_PX - MARGIN_PX
_INNER_W = _INNER_RIGHT - _INNER_LEFT

TITLE_BAND_H_PX = 38
GAP_TITLE_TO_BOXES_PX = 14
BOX_W_PX = 162  # (~150 px target; fills width with four 20 px gaps)
BOX_H_PX = 180

_SCRIPT_DIR = Path(__file__).resolve().parent
_PLOTS_DIR = _SCRIPT_DIR.parent
_REPO_ROOT = _PLOTS_DIR.parent.parent

PNG_PATH = _PLOTS_DIR / "graphical_abstract.png"
PDF_PATH = _PLOTS_DIR / "graphical_abstract.pdf"
SVG_PATH = _PLOTS_DIR / "graphical_abstract.svg"
ROOT_PNG_COPY = _REPO_ROOT / "graphical_abstract.png"

COLORS = {
    "title_bg": "#E8F0FE",
    "b1_structure": "#E8F2DC",
    "b2_gnn": "#BBDEFB",
    "b3_pat": "#E1BEE7",
    "b4_fusion": "#CFD8DC",
    "b5_ode": "#FFF9C4",
    "border": "#424242",
}


def _rounded_rect(
    ax: plt.Axes,
    x_ll: float,
    y_ll: float,
    w: float,
    h: float,
    *,
    facecolor: str,
    zorder: int = 2,
) -> FancyBboxPatch:
    r_corner = float(min(w, h) * 0.09)
    p = FancyBboxPatch(
        (x_ll, y_ll),
        w,
        h,
        boxstyle=f"round,pad=3,rounding_size={max(5.5, r_corner)}",
        linewidth=0.9,
        edgecolor=COLORS["border"],
        facecolor=facecolor,
        zorder=zorder,
        clip_on=False,
    )
    ax.add_patch(p)
    return p


def _arrow_h(ax: plt.Axes, x0: float, x1: float, y: float) -> None:
    patch = FancyArrowPatch(
        (x0, y),
        (x1, y),
        arrowstyle="-|>",
        mutation_scale=8,
        linewidth=1.2,
        shrinkA=0,
        shrinkB=3.8,
        color="#212121",
        zorder=6,
        clip_on=False,
    )
    ax.add_patch(patch)


def _draw_curve_in_box(ax: plt.Axes, x_ll: float, y_ll: float, w_box: float, h_box: float) -> None:
    t = np.linspace(0.0, 1.08, 60)
    y = np.exp(-2.5 * np.maximum(0.0, t - 0.13)) * (1.0 - np.exp(-17 * np.maximum(0.015, t)))
    xmin = x_ll + 0.06 * w_box
    xmax = x_ll + 0.94 * w_box
    t_span = float(t.max() - t.min())
    x_phys = xmin + (t - t.min()) / (t_span + 1e-12) * (xmax - xmin)
    y_norm = np.asarray(y / np.max(y), dtype=float)
    y_base = y_ll + h_box * 0.10
    y_top = y_ll + h_box * 0.70
    ax.plot(
        x_phys,
        y_base + y_norm * (y_top - y_base),
        color="#0277BD",
        lw=2.0,
        clip_on=False,
        solid_capstyle="round",
        zorder=5,
    )


def build_figure() -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
        }
    )

    fig = plt.figure(figsize=FIGSIZE_IN, dpi=DPI)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    ax = fig.add_axes((0, 0, 1, 1))

    ax.set_xlim(0, WIDTH_PX)
    ax.set_ylim(0, HEIGHT_PX)
    ax.axis("off")

    # Title band inside top margin
    title_y_bottom = _INNER_TOP - TITLE_BAND_H_PX
    _rounded_rect(
        ax,
        _INNER_LEFT,
        title_y_bottom,
        _INNER_W,
        TITLE_BAND_H_PX,
        facecolor=COLORS["title_bg"],
        zorder=1,
    )
    ax.text(
        _INNER_LEFT + _INNER_W / 2,
        title_y_bottom + TITLE_BAND_H_PX / 2,
        TITLE_STR,
        ha="center",
        va="center",
        fontsize=TITLE_FS,
        weight="bold",
        color="#1565C0",
        zorder=9,
        clip_on=False,
    )

    usable_w_boxes = BOX_W_PX * 5 + GAP_BOX_PX * 4
    x0_boxes = _INNER_LEFT + (_INNER_W - usable_w_boxes) / 2
    boxes_x_ll = [x0_boxes + i * (BOX_W_PX + GAP_BOX_PX) for i in range(5)]

    boxes_top_y = title_y_bottom - GAP_TITLE_TO_BOXES_PX
    boxes_y_ll = boxes_top_y - BOX_H_PX
    cy = boxes_y_ll + BOX_H_PX / 2
    arrow_y = cy

    _rounded_rect(ax, boxes_x_ll[0], boxes_y_ll, BOX_W_PX, BOX_H_PX, facecolor=COLORS["b1_structure"])
    ax.text(
        boxes_x_ll[0] + BOX_W_PX / 2,
        cy,
        "SMILES → Molecular graph",
        ha="center",
        va="center",
        fontsize=BOX_LABEL_FS,
        weight="bold",
        color="#212121",
        zorder=8,
        clip_on=False,
    )

    _rounded_rect(ax, boxes_x_ll[1], boxes_y_ll, BOX_W_PX, BOX_H_PX, facecolor=COLORS["b2_gnn"])
    cx2 = boxes_x_ll[1] + BOX_W_PX / 2
    ax.text(
        cx2,
        cy + 24,
        "GNN encoder",
        ha="center",
        va="center",
        fontsize=BOX_LABEL_FS,
        weight="bold",
        color="#0D47A1",
        zorder=8,
        clip_on=False,
    )
    ax.text(
        cx2,
        cy - 8,
        "2 MP layers + mean/max pool",
        ha="center",
        va="center",
        fontsize=BOX_SUB_FS,
        color="#363636",
        zorder=8,
        clip_on=False,
    )

    _rounded_rect(ax, boxes_x_ll[2], boxes_y_ll, BOX_W_PX, BOX_H_PX, facecolor=COLORS["b3_pat"])
    cx3 = boxes_x_ll[2] + BOX_W_PX / 2
    ax.text(
        cx3,
        cy + 24,
        "Patient covariates",
        ha="center",
        va="center",
        fontsize=BOX_LABEL_FS,
        weight="bold",
        color="#6A1B9A",
        zorder=8,
        clip_on=False,
    )
    ax.text(
        cx3,
        cy - 8,
        "weight, dose, age, sex",
        ha="center",
        va="center",
        fontsize=BOX_SUB_FS,
        color="#363636",
        zorder=8,
        clip_on=False,
    )

    _rounded_rect(ax, boxes_x_ll[3], boxes_y_ll, BOX_W_PX, BOX_H_PX, facecolor=COLORS["b4_fusion"])
    ax.text(
        boxes_x_ll[3] + BOX_W_PX / 2,
        cy,
        "Fusion MLP → CL, V, ka",
        ha="center",
        va="center",
        fontsize=BOX_LABEL_FS,
        weight="bold",
        color="#37474F",
        zorder=8,
        clip_on=False,
    )

    _rounded_rect(ax, boxes_x_ll[4], boxes_y_ll, BOX_W_PX, BOX_H_PX, facecolor=COLORS["b5_ode"])
    cx5 = boxes_x_ll[4] + BOX_W_PX / 2
    ax.text(
        cx5,
        cy + 46,
        "1-compartment ODE → C(t)",
        ha="center",
        va="center",
        fontsize=BOX_LABEL_FS,
        weight="bold",
        color="#F57F17",
        zorder=8,
        clip_on=False,
    )
    _draw_curve_in_box(ax, boxes_x_ll[4], boxes_y_ll, BOX_W_PX, BOX_H_PX)

    for i in range(4):
        x_from = boxes_x_ll[i] + BOX_W_PX + 3
        x_to = boxes_x_ll[i + 1] - 3
        _arrow_h(ax, x_from, x_to, arrow_y)

    ax.text(
        _INNER_RIGHT,
        _INNER_BOTTOM + 10,
        FOOTER_STR,
        ha="right",
        va="bottom",
        fontsize=FOOTER_FS,
        style="italic",
        color="#424242",
        zorder=9,
        clip_on=False,
    )

    return fig


def save_png_under_budget(path: Path, fig: plt.Figure) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        format="png",
        dpi=DPI,
        bbox_inches=None,
        pad_inches=0,
        facecolor="white",
        edgecolor="none",
        pil_kwargs={"optimize": True},
    )
    plt.close(fig)

    im = Image.open(path)
    if im.size != (WIDTH_PX, HEIGHT_PX):
        resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS)
        im = im.resize((WIDTH_PX, HEIGHT_PX), resample)
        im.save(path, format="PNG", optimize=True)
        im = Image.open(path)
    if im.size != (WIDTH_PX, HEIGHT_PX):
        raise RuntimeError(f"Dimension check failed: {im.size} != {(WIDTH_PX, HEIGHT_PX)}")

    rgb_im = im.convert("RGB")

    def write_png_payload(img_write: Image.Image) -> None:
        buf = BytesIO()
        img_write.save(buf, format="PNG", optimize=True)
        path.write_bytes(buf.getvalue())

    write_png_payload(rgb_im)
    sz = path.stat().st_size

    for n_colors in (192, 160, 144, 128, 112, 96, 84, 72, 64, 52, 40):
        if sz <= MAX_PNG_BYTES:
            break
        q = rgb_im.quantize(colors=n_colors)
        write_png_payload(q)
        sz = path.stat().st_size

    im2 = Image.open(path)
    if im2.size != (WIDTH_PX, HEIGHT_PX):
        raise RuntimeError("PNG dimensions changed after Pillow processing.")
    return sz


def main() -> None:
    _SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    backend = matplotlib.get_backend()
    print(f"[preview] matplotlib backend={backend!r}")

    fig = build_figure()

    fig.savefig(PDF_PATH, format="pdf", dpi=DPI, bbox_inches=None, pad_inches=0)
    fig.savefig(SVG_PATH, format="svg", dpi=DPI, bbox_inches=None, pad_inches=0)

    if _PREVIEW:
        print("[preview] plt.show(block=False): close window within ~12 s.")
        plt.show(block=False)
        plt.pause(12.0)

    png_sz = save_png_under_budget(PNG_PATH, fig)

    pil_img = Image.open(PNG_PATH)
    if pil_img.size != (WIDTH_PX, HEIGHT_PX):
        raise RuntimeError(f"PIL size {pil_img.size} != {(WIDTH_PX, HEIGHT_PX)}")
    if png_sz > MAX_PNG_BYTES:
        raise RuntimeError(f"PNG {png_sz} B exceeds limit {MAX_PNG_BYTES}")

    pdf_sz = PDF_PATH.stat().st_size if PDF_PATH.exists() else -1
    svg_sz = SVG_PATH.stat().st_size if SVG_PATH.exists() else -1
    if pdf_sz <= 0 or svg_sz <= 0:
        raise RuntimeError("PDF or SVG missing or empty.")

    shutil.copyfile(PNG_PATH, ROOT_PNG_COPY)

    verdict = png_sz <= MAX_PNG_BYTES and pil_img.size == (WIDTH_PX, HEIGHT_PX)
    verdict_s = "PASS" if verdict else "FAIL"

    def kb(b: float) -> float:
        return b / 1024.0

    print("\n=== GRAPHICAL ABSTRACT GENERATED ===")
    print(
        f"PNG: {PNG_PATH} "
        f"({pil_img.width}x{pil_img.height} pixels, {kb(png_sz):.1f} KB)"
    )
    print(f"PDF: {PDF_PATH} ({kb(pdf_sz):.1f} KB)")
    print(f"SVG: {SVG_PATH} ({kb(svg_sz):.1f} KB)")
    print(f"All journal spec requirements: {verdict_s}")
    print("=== GRAPHICAL ABSTRACT V2 READY FOR REVIEW ===")


if __name__ == "__main__":
    main()
