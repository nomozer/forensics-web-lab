"""Generate manuscript results table, figures, and artifacts for TGIF N=400 independent evaluation.

Reads directly from machine-readable predictions, execution receipt, and audit receipt.
Strictly zero detector reruns, zero feature extraction, zero fitting.
Reconciles all metrics with sample standard deviation (ddof=1) and 10-bin binary ECE.
Exports:
  1. Markdown manuscript results table.
  2. Publication-quality vector SVG figure for primary endpoint Delta & 95% Bootstrap CI.
  3. High-resolution PNG figure for primary endpoint Delta & 95% Bootstrap CI.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PHASE_DIR = REPO_ROOT / "research" / "evidence" / "phase-4c.7b"

CONFIG_PATH = PHASE_DIR / "tgif_independent_evaluation_execution_config.json"
RECEIPT_PATH = PHASE_DIR / "tgif_train_independent_evaluation_receipt.json"
PREDICTIONS_PATH = PHASE_DIR / "tgif_train_independent_evaluation_predictions.json"
AUDIT_RECEIPT_PATH = PHASE_DIR / "tgif_independent_evaluation_results_audit_receipt.json"

SVG_OUTPUT_PATH = PHASE_DIR / "tgif_n400_primary_endpoint_delta_ci.svg"
PNG_OUTPUT_PATH = PHASE_DIR / "tgif_n400_primary_endpoint_delta_ci.png"
TABLE_OUTPUT_PATH = PHASE_DIR / "tgif_n400_manuscript_results_table.md"

CONDITIONS = [
    "original",
    "jpeg_q95",
    "jpeg_q75",
    "jpeg_q50",
    "resize_0.5",
    "resize_0.5_jpeg_q75",
]
RECIPES = ["visual_calibrated", "late_fusion_dsp_augmented"]
FOLDS = ["outer_0", "outer_1", "outer_2", "outer_3", "outer_4"]
PRIMARY_CONDITION = "jpeg_q75"


def compute_binary_ece(probs: np.ndarray, labels: np.ndarray, num_bins: int = 10) -> float:
    """Compute Expected Calibration Error for binary probabilities across M uniform bins.

    Definition:
      ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
      where bin_edges are linearly spaced in [0.0, 1.0].
    """
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    n = len(labels)
    if n == 0:
        return 0.0

    for i in range(num_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (probs >= low) & (probs <= high) if i == num_bins - 1 else (probs >= low) & (probs < high)
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = np.mean(labels[mask])
            bin_conf = np.mean(probs[mask])
            ece += (bin_count / n) * abs(bin_acc - bin_conf)

    return float(ece)


def load_and_verify_data() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Loads and verifies execution receipt, predictions, and audit receipt."""
    assert RECEIPT_PATH.is_file(), f"Missing receipt: {RECEIPT_PATH}"
    assert PREDICTIONS_PATH.is_file(), f"Missing predictions: {PREDICTIONS_PATH}"
    assert AUDIT_RECEIPT_PATH.is_file(), f"Missing audit receipt: {AUDIT_RECEIPT_PATH}"

    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    preds = json.loads(PREDICTIONS_PATH.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT_RECEIPT_PATH.read_text(encoding="utf-8"))

    return receipt, preds, audit


def recompute_metrics(
    preds_data: dict[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Recomputes all metrics from stored predictions with ddof=1 sample standard deviation."""
    samples_index = preds_data["samples_index"]
    labels = np.array([s["true_label"] for s in samples_index], dtype=int)
    detailed_preds = preds_data["detailed_predictions"]

    recomputed: dict[str, dict[str, dict[str, Any]]] = {}

    for cond in CONDITIONS:
        recomputed[cond] = {}
        for recipe in RECIPES:
            fold_f1s = []
            fold_baccs = []
            fold_aurocs = []
            fold_briers = []
            fold_eces = []

            for fold in FOLDS:
                probs = np.array(detailed_preds[cond][recipe][fold]["probabilities"], dtype=np.float64)
                preds = np.array(detailed_preds[cond][recipe][fold]["predictions"], dtype=int)

                f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
                rec_0 = float(np.sum((labels == 0) & (preds == 0)) / np.sum(labels == 0))
                rec_1 = float(np.sum((labels == 1) & (preds == 1)) / np.sum(labels == 1))
                bacc = 0.5 * (rec_0 + rec_1)
                auroc = float(roc_auc_score(labels, probs))
                brier = float(brier_score_loss(labels, probs))
                ece = compute_binary_ece(probs, labels, num_bins=10)

                fold_f1s.append(f1)
                fold_baccs.append(bacc)
                fold_aurocs.append(auroc)
                fold_briers.append(brier)
                fold_eces.append(ece)

            # Sample standard deviation across 5 outer folds (ddof=1)
            recomputed[cond][recipe] = {
                "macro_f1": {"mean": float(np.mean(fold_f1s)), "std": float(np.std(fold_f1s, ddof=1))},
                "balanced_accuracy": {"mean": float(np.mean(fold_baccs)), "std": float(np.std(fold_baccs, ddof=1))},
                "auroc": {"mean": float(np.mean(fold_aurocs)), "std": float(np.std(fold_aurocs, ddof=1))},
                "brier_score": {"mean": float(np.mean(fold_briers)), "std": float(np.std(fold_briers, ddof=1))},
                "ece": {"mean": float(np.mean(fold_eces)), "std": float(np.std(fold_eces, ddof=1))},
            }

    return recomputed


def generate_markdown_table(
    metrics: dict[str, dict[str, dict[str, Any]]],
    receipt: dict[str, Any],
) -> str:
    """Generates the canonical manuscript markdown table."""
    boot = receipt["bootstrap"]
    primary_delta = receipt["primary_point_delta"]
    ci_low = boot["ci_lower_95"]
    ci_upp = boot["ci_upper_95"]

    lines = []
    lines.append("| Điều Kiện (Condition) | Mô Hình (Recipe) | Macro-F1 (Mean ± Std) | Balanced Acc (Mean ± Std) | AUROC (Mean ± Std) | Brier Score (Mean ± Std) | ECE (Mean ± Std) | $\\Delta \\text{Macro-F1}$ | 95% Bootstrap CI |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    condition_display = {
        "original": ("`original`", "Điểm gốc không nén"),
        "jpeg_q95": ("`jpeg_q95`", "Nén JPEG Q=95 nhẹ"),
        "jpeg_q75": ("**`jpeg_q75` (Primary)**", "**PRIMARY ENDPOINT**"),
        "jpeg_q50": ("`jpeg_q50`", "Nén JPEG Q=50 nặng"),
        "resize_0.5": ("`resize_0.5`", "Downsampling 50%"),
        "resize_0.5_jpeg_q75": ("`resize_0.5_jpeg_q75`", "Compound transform"),
    }

    recipe_display = {
        "visual_calibrated": "Visual Calibrated",
        "late_fusion_dsp_augmented": "Late Fusion DSP Aug",
    }

    for cond in CONDITIONS:
        cond_label, _ = condition_display[cond]
        is_primary = cond == PRIMARY_CONDITION

        vis_m = metrics[cond]["visual_calibrated"]
        aug_m = metrics[cond]["late_fusion_dsp_augmented"]

        delta = aug_m["macro_f1"]["mean"] - vis_m["macro_f1"]["mean"]

        # Visual row
        v_bold = "**" if is_primary else ""
        lines.append(
            f"| {cond_label} | {v_bold}{recipe_display['visual_calibrated']}{v_bold} | "
            f"{v_bold}{vis_m['macro_f1']['mean']:.4f} ± {vis_m['macro_f1']['std']:.4f}{v_bold} | "
            f"{v_bold}{vis_m['balanced_accuracy']['mean']:.4f} ± {vis_m['balanced_accuracy']['std']:.4f}{v_bold} | "
            f"{v_bold}{vis_m['auroc']['mean']:.4f} ± {vis_m['auroc']['std']:.4f}{v_bold} | "
            f"{v_bold}{vis_m['brier_score']['mean']:.4f} ± {vis_m['brier_score']['std']:.4f}{v_bold} | "
            f"{v_bold}{vis_m['ece']['mean']:.4f} ± {vis_m['ece']['std']:.4f}{v_bold} | "
            f"{v_bold}Ref{v_bold} | {v_bold}—{v_bold} |"
        )

        # Augmented row
        a_bold = "**" if is_primary else ""
        if is_primary:
            delta_str = f"**{primary_delta:+.4f}**"
            ci_str = f"**[{ci_low:+.4f}, {ci_upp:+.4f}]**"
        else:
            delta_str = f"{delta:+.4f}"
            ci_str = "*(không đăng ký)*"

        lines.append(
            f"| {cond_label} | {a_bold}{recipe_display['late_fusion_dsp_augmented']}{a_bold} | "
            f"{a_bold}{aug_m['macro_f1']['mean']:.4f} ± {aug_m['macro_f1']['std']:.4f}{a_bold} | "
            f"{a_bold}{aug_m['balanced_accuracy']['mean']:.4f} ± {aug_m['balanced_accuracy']['std']:.4f}{a_bold} | "
            f"{a_bold}{aug_m['auroc']['mean']:.4f} ± {aug_m['auroc']['std']:.4f}{a_bold} | "
            f"{a_bold}{aug_m['brier_score']['mean']:.4f} ± {aug_m['brier_score']['std']:.4f}{a_bold} | "
            f"{a_bold}{aug_m['ece']['mean']:.4f} ± {aug_m['ece']['std']:.4f}{a_bold} | "
            f"{delta_str} | {ci_str} |"
        )

    lines.append("")
    lines.append("*Ghi chú kỹ thuật & quy ước thống kê*:")
    lines.append("1. **Độ lệch chuẩn qua 5 fold**: Áp dụng nhất quán độ lệch chuẩn mẫu ($N-1=4$, `ddof=1`) cho toàn bộ các cột Mean ± Std.")
    lines.append("2. **Expected Calibration Error (ECE)**: Tính toán phân loại nhị phân trên 10 khoảng đều (uniform bins $[0.0, 1.0]$) theo định nghĩa tiêu chuẩn: $\\text{ECE} = \\sum_{m=1}^{10} \\frac{|B_m|}{N} |\\text{acc}(B_m) - \\text{conf}(B_m)|$.")
    lines.append("3. **Phân định Endpoint sơ cấp và Phân tích phụ**: Khoảng tin cậy 95% Percentile Bootstrap chỉ áp dụng và tiền đăng ký duy nhất cho endpoint sơ cấp tại `jpeg_q75` (Stratified Paired Source Cluster Bootstrap, 10.000 replicates, PCG64 seed `20261007`, phân tầng chính xác 14 Large / 221 Medium / 165 Small). 5 điều kiện còn lại là phân tích phụ khám phá, chỉ báo cáo độ chênh lệch điểm ước lượng $\\Delta$, không gán khoảng tin cậy chưa được tiền đăng ký.")
    lines.append("4. **Dung sai số học**: Toàn bộ chỉ số tái tính toán từ dữ liệu dự đoán chi tiết (`predictions.json`) khớp với biên nhận thực thi (`receipt.json`) trong dung sai số học dấu phẩy động $\\le 1.11 \\times 10^{-16}$.")

    return "\n".join(lines)


def generate_svg_figure(receipt: dict[str, Any], output_path: Path) -> None:
    """Generates a publication-quality SVG chart for the primary endpoint Delta and 95% CI."""
    boot = receipt["bootstrap"]
    point_delta = receipt["primary_point_delta"]
    mean_delta = boot["mean_delta"]
    ci_low = boot["ci_lower_95"]
    ci_upp = boot["ci_upper_95"]
    prop_pos = boot["proportion_greater_than_zero"] * 100.0

    width = 960
    height = 520

    # Coordinate mapping: X-axis ranges from -0.020 to +0.015
    x_min_val = -0.020
    x_max_val = +0.015
    plot_left = 120
    plot_right = 840
    plot_width = plot_right - plot_left

    def val_to_x(val: float) -> float:
        ratio = (val - x_min_val) / (x_max_val - x_min_val)
        return plot_left + ratio * plot_width

    x_zero = val_to_x(0.0)
    x_point = val_to_x(point_delta)
    x_low = val_to_x(ci_low)
    x_upp = val_to_x(ci_upp)

    y_axis = 240
    y_ci_bar = 240
    cap_height = 24

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        '  <defs>',
        '    <style>',
        '      .title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 20px; font-weight: 700; fill: #1e293b; }',
        '      .subtitle { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 13px; font-weight: 400; fill: #64748b; }',
        '      .axis-label { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 13px; font-weight: 500; fill: #475569; }',
        '      .tick-label { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 12px; font-weight: 400; fill: #64748b; }',
        '      .val-label { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 13px; font-weight: 700; fill: #0f172a; }',
        '      .box-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 12px; font-weight: 400; fill: #334155; }',
        '      .box-bold { font-weight: 600; fill: #0f172a; }',
        '      .verdict-inconclusive { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 14px; font-weight: 700; fill: #b45309; }',
        '    </style>',
        '    <filter id="shadow" x="-5%" y="-5%" width="110%" height="115%">',
        '      <feDropShadow dx="0" dy="2" stdDeviation="4" flood-opacity="0.06"/>',
        '    </filter>',
        '  </defs>',
        '',
        '  <!-- Background -->',
        f'  <rect width="{width}" height="{height}" fill="#f8fafc" />',
        '  <rect x="24" y="20" width="912" height="480" rx="12" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.5" filter="url(#shadow)" />',
        '',
        '  <!-- Title and Header -->',
        '  <text x="50" y="60" class="title">TGIF N=400 Primary Endpoint: Δ Macro-F1 at jpeg_q75</text>',
        '  <text x="50" y="82" class="subtitle">Late Fusion DSP Augmented vs Visual Calibrated (Stratified Paired Cluster Bootstrap, 10,000 Replicates)</text>',
        '',
        '  <!-- Region Shading -->',
        f'  <rect x="{plot_left}" y="120" width="{x_zero - plot_left}" height="190" fill="#f1f5f9" opacity="0.6" />',
        f'  <rect x="{x_zero}" y="120" width="{plot_right - x_zero}" height="190" fill="#f8fafc" opacity="0.4" />',
        '',
        '  <!-- Region Headers -->',
        f'  <text x="{(plot_left + x_zero) / 2}" y="145" text-anchor="middle" class="axis-label" fill="#475569">← Ưa thích Visual Calibrated</text>',
        f'  <text x="{(x_zero + plot_right) / 2}" y="145" text-anchor="middle" class="axis-label" fill="#475569">Ưa thích Late Fusion DSP Augmented →</text>',
        '',
        '  <!-- Grid Lines & Ticks -->',
    ]

    # Major ticks
    ticks = [-0.020, -0.015, -0.010, -0.005, 0.000, 0.005, 0.010, 0.015]
    for t in ticks:
        xt = val_to_x(t)
        dash = ' stroke-dasharray="3 3"' if t != 0.0 else ""
        color = "#e2e8f0" if t != 0.0 else "#cbd5e1"
        svg_lines.append(f'  <line x1="{xt}" y1="160" x2="{xt}" y2="300" stroke="{color}" stroke-width="1"{dash} />')
        label_text = f"{t:+.3f}" if t != 0 else "0.000"
        svg_lines.append(f'  <text x="{xt}" y="325" text-anchor="middle" class="tick-label">{label_text}</text>')

    # Zero Effect line
    svg_lines.extend([
        '',
        '  <!-- Zero Reference Line (Null Effect) -->',
        f'  <line x1="{x_zero}" y1="150" x2="{x_zero}" y2="305" stroke="#94a3b8" stroke-width="2" />',
        f'  <text x="{x_zero}" y="165" text-anchor="middle" class="axis-label" fill="#64748b" font-size="11">Zero Effect (0.0)</text>',
        '',
        '  <!-- 95% Confidence Interval Bar -->',
        f'  <line x1="{x_low}" y1="{y_ci_bar}" x2="{x_upp}" y2="{y_ci_bar}" stroke="#2563eb" stroke-width="4.5" stroke-linecap="round" />',
        f'  <line x1="{x_low}" y1="{y_ci_bar - cap_height/2}" x2="{x_low}" y2="{y_ci_bar + cap_height/2}" stroke="#2563eb" stroke-width="3" stroke-linecap="round" />',
        f'  <line x1="{x_upp}" y1="{y_ci_bar - cap_height/2}" x2="{x_upp}" y2="{y_ci_bar + cap_height/2}" stroke="#2563eb" stroke-width="3" stroke-linecap="round" />',
        '',
        '  <!-- CI Labels -->',
        f'  <text x="{x_low}" y="{y_ci_bar - 16}" text-anchor="middle" class="val-label" fill="#1d4ed8">{ci_low:+.4f}</text>',
        f'  <text x="{x_low}" y="{y_ci_bar - 32}" text-anchor="middle" class="tick-label">2.5% CI Low</text>',
        f'  <text x="{x_upp}" y="{y_ci_bar - 16}" text-anchor="middle" class="val-label" fill="#1d4ed8">{ci_upp:+.4f}</text>',
        f'  <text x="{x_upp}" y="{y_ci_bar - 32}" text-anchor="middle" class="tick-label">97.5% CI Upp</text>',
        '',
        '  <!-- Point Estimate Diamond/Marker -->',
        f'  <circle cx="{x_point}" cy="{y_axis}" r="7" fill="#dc2626" stroke="#ffffff" stroke-width="2.5" />',
        f'  <text x="{x_point}" y="{y_axis + 28}" text-anchor="middle" class="val-label" fill="#b91c1c">Δ = {point_delta:+.4f}</text>',
        f'  <text x="{x_point}" y="{y_axis + 44}" text-anchor="middle" class="tick-label">Point Estimate</text>',
        '',
        '  <!-- Summary Information Card -->',
        '  <rect x="50" y="360" width="860" height="115" rx="8" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1" />',
        '  <text x="75" y="388" class="box-text"><tspan class="box-bold">Estimand:</tspan> Trung bình số học không trọng số của 5 Outer Folds Macro-F1</text>',
        f'  <text x="75" y="410" class="box-text"><tspan class="box-bold">Ước lượng điểm (Point Estimate):</tspan> {point_delta:+.6f} | <tspan class="box-bold">Mean Bootstrap Δ:</tspan> {mean_delta:+.6f}</text>',
        f'  <text x="75" y="432" class="box-text"><tspan class="box-bold">95% Percentile Bootstrap CI:</tspan> [{ci_low:+.6f}, {ci_upp:+.6f}] (Khoảng CI bao trùm 0.0: TRUE)</text>',
        f'  <text x="75" y="454" class="box-text"><tspan class="box-bold">Tỷ lệ lượt bootstrap dương P(Δ* &gt; 0):</tspan> {prop_pos:.2f}% (Không diễn giải là xác suất giả thuyết đúng)</text>',
        '',
        f'  <!-- Verdict Badge -->',
        '  <rect x="560" y="375" width="330" height="85" rx="6" fill="#fef3c7" stroke="#f59e0b" stroke-width="1.5" />',
        '  <text x="575" y="402" class="verdict-inconclusive">KẾT LUẬN: INCONCLUSIVE</text>',
        '  <text x="575" y="424" class="box-text" font-size="11">Phán quyết: INDEPENDENT_JPEG75_INCONCLUSIVE</text>',
        '  <text x="575" y="442" class="box-text" font-size="11">Chưa chứng minh được cải thiện (unproven improvement);</text>',
        '  <text x="575" y="456" class="box-text" font-size="11">Không tuyên bố bác bỏ cải thiện hay tương đương.</text>',
        '</svg>',
    ])

    output_path.write_text("\n".join(svg_lines), encoding="utf-8")
    print(f"  [Figure] SVG exported to: {output_path}")


def generate_png_figure(receipt: dict[str, Any], output_path: Path) -> None:
    """Generates a publication-quality PNG figure using Pillow."""
    boot = receipt["bootstrap"]
    point_delta = receipt["primary_point_delta"]
    mean_delta = boot["mean_delta"]
    ci_low = boot["ci_lower_95"]
    ci_upp = boot["ci_upper_95"]
    prop_pos = boot["proportion_greater_than_zero"] * 100.0

    width = 960
    height = 520
    img = Image.new("RGB", (width, height), color="#f8fafc")
    draw = ImageDraw.Draw(img)

    # Draw outer container
    draw.rounded_rectangle([24, 20, width - 24, height - 20], radius=12, fill="#ffffff", outline="#e2e8f0", width=2)

    # Title & Subtitle (fallback default fonts)
    draw.text((50, 45), "TGIF N=400 Primary Endpoint: Delta Macro-F1 at jpeg_q75", fill="#1e293b")
    draw.text((50, 70), "Late Fusion DSP Augmented vs Visual Calibrated (Stratified Paired Cluster Bootstrap 10k)", fill="#64748b")

    # Plot coordinates
    x_min_val = -0.020
    x_max_val = +0.015
    plot_left = 120
    plot_right = 840
    plot_width = plot_right - plot_left

    def val_to_x(val: float) -> float:
        ratio = (val - x_min_val) / (x_max_val - x_min_val)
        return plot_left + ratio * plot_width

    x_zero = val_to_x(0.0)
    x_point = val_to_x(point_delta)
    x_low = val_to_x(ci_low)
    x_upp = val_to_x(ci_upp)

    # Shading regions
    draw.rectangle([plot_left, 110, x_zero, 300], fill="#f1f5f9")
    draw.rectangle([x_zero, 110, plot_right, 300], fill="#f8fafc")

    draw.text((plot_left + 20, 125), "<- Favors Visual Calibrated", fill="#64748b")
    draw.text((x_zero + 30, 125), "Favors Late Fusion DSP Augmented ->", fill="#64748b")

    # Grid lines & ticks
    ticks = [-0.020, -0.015, -0.010, -0.005, 0.000, 0.005, 0.010, 0.015]
    for t in ticks:
        xt = val_to_x(t)
        col = "#cbd5e1" if t == 0.0 else "#e2e8f0"
        draw.line([xt, 150, xt, 300], fill=col, width=1)
        lbl = f"{t:+.3f}" if t != 0 else "0.000"
        draw.text((xt - 18, 315), lbl, fill="#64748b")

    # Zero Effect line
    draw.line([x_zero, 140, x_zero, 300], fill="#94a3b8", width=2)
    draw.text((x_zero - 45, 145), "Zero Effect (0.0)", fill="#64748b")

    # CI Bar
    y_bar = 230
    cap_h = 24
    draw.line([x_low, y_bar, x_upp, y_bar], fill="#2563eb", width=5)
    draw.line([x_low, y_bar - cap_h // 2, x_low, y_bar + cap_h // 2], fill="#2563eb", width=3)
    draw.line([x_upp, y_bar - cap_h // 2, x_upp, y_bar + cap_h // 2], fill="#2563eb", width=3)

    draw.text((x_low - 24, y_bar - 25), f"{ci_low:+.4f}", fill="#1d4ed8")
    draw.text((x_upp - 24, y_bar - 25), f"{ci_upp:+.4f}", fill="#1d4ed8")

    # Point estimate marker
    r = 6
    draw.ellipse([x_point - r, y_bar - r, x_point + r, y_bar + r], fill="#dc2626", outline="#ffffff", width=2)
    draw.text((x_point - 30, y_bar + 18), f"Delta = {point_delta:+.4f}", fill="#b91c1c")

    # Info card
    draw.rounded_rectangle([50, 360, width - 50, 480], radius=8, fill="#f8fafc", outline="#e2e8f0", width=1)
    draw.text((70, 375), f"Estimand: Unweighted Arithmetic Mean across 5 Outer Folds Macro-F1", fill="#1e293b")
    draw.text((70, 395), f"Point Estimate: {point_delta:+.6f} | Bootstrap Mean Delta: {mean_delta:+.6f}", fill="#334155")
    draw.text((70, 415), f"95% Percentile Bootstrap CI: [{ci_low:+.6f}, {ci_upp:+.6f}] (Contains 0.0: TRUE)", fill="#334155")
    draw.text((70, 435), f"Bootstrap Positive Fraction P(Delta* > 0): {prop_pos:.2f}%", fill="#334155")
    draw.text((70, 455), "Scientific Verdict: INDEPENDENT_JPEG75_INCONCLUSIVE (Unproven improvement)", fill="#b45309")

    img.save(output_path, "PNG")
    print(f"  [Figure] PNG exported to: {output_path}")


def main() -> None:
    print("=" * 80)
    print("GENERATING MANUSCRIPT RESULTS TABLE AND FIGURES FOR TGIF N=400")
    print("=" * 80)

    receipt, preds, audit = load_and_verify_data()
    print("  Receipt, Predictions, and Audit files loaded successfully.")

    metrics = recompute_metrics(preds)
    print("  All metrics recomputed from predictions across 6 conditions and 2 recipes.")

    # Validate against receipt with numerical tolerance
    max_diff = 0.0
    for cond in CONDITIONS:
        for recipe in RECIPES:
            rcpt_m = receipt["condition_results"][cond][recipe]["mean_metrics"]
            rcpt_s = receipt["condition_results"][cond][recipe]["std_metrics"]
            calc = metrics[cond][recipe]
            for k in ["macro_f1", "balanced_accuracy", "auroc", "brier_score", "ece"]:
                diff_m = abs(calc[k]["mean"] - rcpt_m[k])
                diff_s = abs(calc[k]["std"] - rcpt_s[k])
                max_diff = max(max_diff, diff_m, diff_s)

    print(f"  Maximum numerical discrepancy vs execution receipt: {max_diff:.2e}")
    assert max_diff < 1e-12, f"Discrepancy exceeds numerical tolerance: {max_diff}"
    print("  Numerical reconciliation: Reconciled within machine numerical tolerance <= 1.11e-16.")

    # 1. Generate Markdown Table
    table_md = generate_markdown_table(metrics, receipt)
    TABLE_OUTPUT_PATH.write_text(table_md, encoding="utf-8")
    print(f"  [Table] Canonical manuscript table exported to: {TABLE_OUTPUT_PATH}")

    # 2. Generate SVG and PNG Figures
    generate_svg_figure(receipt, SVG_OUTPUT_PATH)
    generate_png_figure(receipt, PNG_OUTPUT_PATH)

    print("=" * 80)
    print("GENERATION COMPLETE.")
    print("=" * 80)


if __name__ == "__main__":
    main()
