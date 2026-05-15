"""PDF report generation service.

Produces an FDA-style PK analysis report with:
- Patient & dosing inputs
- Safety decision
- PK metrics table
- Concentration-time curve (matplotlib PNG)
- SHAP feature attribution chart
- Sensitivity tornado chart
- Optional recommendation summary
"""

from __future__ import annotations

import io
import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np

logger = logging.getLogger("uvicorn.error")


def generate_pdf(
    patient: dict,
    regimen: list[dict],
    pk_metrics: dict,
    safety: dict,
    pk_params: dict,
    times: list[float],
    conc: list[float],
    shap_data: dict | None = None,
    sensitivity_data: dict | None = None,
    narrative: dict | None = None,
    strategies: list[dict] | None = None,
    population_data: dict | None = None,
) -> bytes:
    """Return PDF bytes for a regulatory-ready PK report."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
        PageBreak, HRFlowable,
    )

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=20*mm, bottomMargin=20*mm,
                            leftMargin=18*mm, rightMargin=18*mm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title2", parent=styles["Title"], fontSize=18,
                                  spaceAfter=4*mm, textColor=colors.HexColor("#312e81"))
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13,
                         spaceBefore=6*mm, spaceAfter=3*mm, textColor=colors.HexColor("#4338ca"))
    body = styles["BodyText"]
    small = ParagraphStyle("Small", parent=body, fontSize=8, textColor=colors.grey)

    elements: list = []

    # ---- Title ----
    elements.append(Paragraph("DL-PBPK Pharmacokinetic Analysis Report", title_style))
    elements.append(Paragraph(
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", small))
    elements.append(HRFlowable(width="100%", color=colors.HexColor("#c7d2fe"), thickness=1))
    elements.append(Spacer(1, 4*mm))

    # ---- Patient & Dosing ----
    elements.append(Paragraph("Patient & Dosing Information", h2))
    weight = patient.get("weight_kg", 70)
    compound = patient.get("compound_name", "Unknown")
    total_dose = sum(e.get("dose_mg", 0) for e in regimen)
    info_data = [
        ["Compound", compound],
        ["Weight (kg)", f"{weight}"],
        ["Total Dose (mg)", f"{total_dose:.1f}"],
        ["Dose/kg (mg/kg)", f"{total_dose / max(weight, 1):.2f}"],
        ["Regimen Events", str(len(regimen))],
    ]
    info_table = Table(info_data, colWidths=[45*mm, 80*mm])
    info_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
    ]))
    elements.append(info_table)

    # ---- Safety ----
    elements.append(Paragraph("Safety Assessment", h2))
    safe_text = "SAFE" if safety.get("is_safe") else "UNSAFE"
    safe_color = "#059669" if safety.get("is_safe") else "#dc2626"
    elements.append(Paragraph(
        f'<font color="{safe_color}" size="14"><b>{safe_text}</b></font>'
        f'&nbsp;&nbsp;Risk score: {safety.get("risk_score", 0):.4f}', body))
    elements.append(Paragraph(safety.get("reason", ""), body))
    elements.append(Spacer(1, 2*mm))

    # ---- PK Metrics ----
    elements.append(Paragraph("Pharmacokinetic Metrics", h2))
    pk_data = [["Metric", "Value"]]
    labels = {"cmax_ng_ml": "Cmax (ng/mL)", "tmax_h": "Tmax (h)", "auc_0_inf": "AUC0-inf (ng*h/mL)",
              "half_life_h": "t1/2 (h)", "clearance_l_h": "CL (L/h)", "vd_l": "Vd (L)"}
    for k, lbl in labels.items():
        pk_data.append([lbl, f"{pk_metrics.get(k, 0):.2f}"])
    pk_table = Table(pk_data, colWidths=[55*mm, 45*mm])
    pk_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2ff")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c7d2fe")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
    ]))
    elements.append(pk_table)

    # ---- Curve Plot ----
    elements.append(Paragraph("Concentration-Time Profile", h2))
    curve_img = _plot_curve(times, conc)
    elements.append(Image(curve_img, width=160*mm, height=80*mm))

    # ---- Population uncertainty bands ----
    if population_data and population_data.get("bands"):
        elements.append(Paragraph("Population Uncertainty Analysis", h2))
        pop_img = _plot_population_bands(population_data)
        elements.append(Image(pop_img, width=160*mm, height=80*mm))
        elements.append(Spacer(1, 3*mm))

        pop_risk = population_data.get("population_risk", {})
        elements.append(Paragraph(
            f'Population Safety: <b>P(safe) = {pop_risk.get("p_safe", 0):.1%}</b>, '
            f'P(unsafe) = {pop_risk.get("p_unsafe", 0):.1%} '
            f'(n={population_data.get("n_samples", 0)} samples)', body))
        elements.append(Spacer(1, 2*mm))

        md = population_data.get("metrics_dist", {})
        pop_table_data = [["Metric", "5th %ile", "Median", "95th %ile"]]
        for mname, mlabel in [("cmax", "Cmax (ng/mL)"), ("auc", "AUC (ng*h/mL)"), ("tmax", "Tmax (h)")]:
            d = md.get(mname, {})
            pop_table_data.append([
                mlabel, f"{d.get('p05', 0):.1f}", f"{d.get('p50', 0):.1f}", f"{d.get('p95', 0):.1f}",
            ])
        pop_tbl = Table(pop_table_data, colWidths=[45*mm, 30*mm, 30*mm, 30*mm])
        pop_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2ff")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c7d2fe")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
        ]))
        elements.append(pop_tbl)

        omega = population_data.get("omega", {})
        elements.append(Spacer(1, 2*mm))
        elements.append(Paragraph(
            f"Random-effect SDs: omega_CL={omega.get('cl', 0):.2f}, "
            f"omega_V={omega.get('v', 0):.2f}, omega_ka={omega.get('ka', 0):.2f}", small))

    # ---- SHAP ----
    if shap_data and shap_data.get("values"):
        elements.append(PageBreak())
        elements.append(Paragraph("Feature Attribution (SHAP)", h2))
        shap_img = _plot_shap(shap_data)
        elements.append(Image(shap_img, width=150*mm, height=70*mm))
        elements.append(Spacer(1, 2*mm))
        elements.append(Paragraph(
            f"Target: {shap_data.get('target', 'risk_score')} | "
            f"Base value: {shap_data.get('base', 0):.4f}", small))

    # ---- Sensitivity ----
    if sensitivity_data and sensitivity_data.get("delta_cmax_pct"):
        elements.append(Paragraph("Mechanistic Sensitivity Analysis", h2))
        sens_img = _plot_sensitivity(sensitivity_data)
        elements.append(Image(sens_img, width=150*mm, height=75*mm))

    # ---- Narrative ----
    if narrative:
        elements.append(Paragraph("Interpretation", h2))
        elements.append(Paragraph(narrative.get("summary", ""), body))

    # ---- Strategies ----
    if strategies:
        elements.append(Paragraph("Dosing Recommendations", h2))
        for s in strategies:
            elements.append(Paragraph(
                f"<b>{s.get('title', '')}</b>: {s.get('description', '')}", body))
            safe_s = "SAFE" if s.get("safety", {}).get("is_safe") else "UNSAFE"
            elements.append(Paragraph(
                f"&nbsp;&nbsp;Result: {safe_s} | "
                f"Delta Cmax: {s.get('delta_cmax_pct', 0):+.1f}% | "
                f"Delta AUC: {s.get('delta_auc_pct', 0):+.1f}%", small))
            elements.append(Spacer(1, 2*mm))

    # ---- Footer ----
    elements.append(Spacer(1, 8*mm))
    elements.append(HRFlowable(width="100%", color=colors.HexColor("#c7d2fe"), thickness=0.5))
    elements.append(Paragraph(
        "This report was generated by the DL-PBPK Hybrid Model platform. "
        "For investigational use only.", small))

    doc.build(elements)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Matplotlib helpers  (render to in-memory PNG wrapped in BytesIO)
# ---------------------------------------------------------------------------

def _plot_curve(times: list[float], conc: list[float]) -> io.BytesIO:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(times, conc, color="#4f46e5", linewidth=1.5)
    ax.set_xlabel("Time (h)", fontsize=9)
    ax.set_ylabel("Concentration (ng/mL)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def _plot_shap(shap_data: dict) -> io.BytesIO:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    features = shap_data["features"]
    values = shap_data["values"]
    y = np.arange(len(features))
    clrs = ["#ef4444" if v > 0 else "#10b981" for v in values]

    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    ax.barh(y, values, color=clrs, height=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(features, fontsize=9)
    ax.set_xlabel("SHAP value (impact on risk score)", fontsize=9)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def _plot_population_bands(pop: dict) -> io.BytesIO:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = pop["times_hr"]
    bands = pop["bands"]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.fill_between(times, bands["p05"], bands["p95"], alpha=0.2, color="#4f46e5", label="5th-95th %ile")
    ax.plot(times, bands["p50"], color="#4f46e5", linewidth=1.5, label="Median")
    ax.set_xlabel("Time (h)", fontsize=9)
    ax.set_ylabel("Concentration (ng/mL)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def _plot_sensitivity(sens: dict) -> io.BytesIO:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    params = sens["parameters"]
    d_cmax = sens["delta_cmax_pct"]
    d_auc = sens["delta_auc_pct"]
    y = np.arange(len(params))
    h = 0.35

    fig, ax = plt.subplots(figsize=(6.5, 3))
    ax.barh(y - h / 2, d_cmax, h, label="Cmax sensitivity", color="#4f46e5")
    ax.barh(y + h / 2, d_auc, h, label="AUC sensitivity", color="#f59e0b")
    ax.set_yticks(y)
    ax.set_yticklabels(params, fontsize=9)
    ax.set_xlabel("Sensitivity (%change / %perturbation)", fontsize=9)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf
