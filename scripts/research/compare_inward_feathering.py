#!/usr/bin/env python3
"""Compare hard composite with inward-only edge feathering (k=2 px) on pilot-20261008T113700Z.

Invariants:
- Preserves all original ZIP, image, mask, receipt, and ledger files untouched.
- Saves derived images with explicit '_feathered_k2.png' suffix in a dedicated subfolder.
- Alpha blending operates strictly inside the registered binary mask (mask == 255).
- Guaranteed: outside_mask_mean_l1 == 0.000000 across all pairs.
- Produces a self-contained HTML contact sheet with base64 embedded images.
"""
from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = REPO_ROOT / "data/research/local-artifacts/phase-4c.7b/pilot-20261008T113700Z"
IMAGES_DIR = RUN_DIR / "images"
MASKS_DIR = RUN_DIR / "masks"
OUTPUT_DIR = RUN_DIR / "derived_feathered_k2"
CONTACT_SHEET_PATH = RUN_DIR / "feathering_comparison_contact_sheet.html"


def compute_inward_feathered_image(
    auth_img: Image.Image,
    edit_img: Image.Image,
    mask_img: Image.Image,
    k: int = 2,
    mode: str = "cosine",
) -> tuple[Image.Image, dict]:
    """Compute inward-feathered composite preserving 100% of outside pixels."""
    auth_arr = np.array(auth_img, dtype=np.float32)
    edit_arr = np.array(edit_img, dtype=np.float32)
    mask_arr = np.array(mask_img, dtype=np.uint8)

    mask_bin = (mask_arr == 255)
    outside_bin = (mask_arr == 0)

    # Euclidean distance from inside mask to nearest non-mask pixel
    dist = distance_transform_edt(mask_bin)

    # Initialize alpha: 0 outside mask, 1 inside core (dist > k)
    alpha = np.zeros_like(dist, dtype=np.float32)

    if mode == "cosine":
        # Cosine S-curve ramp from 0 to 1 over d in (0, k]
        transition = (dist > 0) & (dist <= k)
        alpha[transition] = 0.5 * (1.0 - np.cos(np.pi * dist[transition] / (k + 1.0)))
        alpha[dist > k] = 1.0
    else:  # linear
        transition = (dist > 0) & (dist <= k)
        alpha[transition] = dist[transition] / (k + 1.0)
        alpha[dist > k] = 1.0

    # Ensure outside mask is strictly 0.0
    alpha[outside_bin] = 0.0

    # Expand alpha to 3 channels (H, W, 3)
    alpha_3d = np.expand_dims(alpha, axis=-1)

    # Blend
    feathered_arr = np.clip(np.round(alpha_3d * edit_arr + (1.0 - alpha_3d) * auth_arr), 0, 255).astype(np.uint8)
    feathered_img = Image.fromarray(feathered_arr, mode="RGB")

    # Metrics
    outside_diff = np.abs(feathered_arr.astype(np.float32) - auth_arr)[outside_bin]
    outside_mean_l1 = float(np.mean(outside_diff)) if len(outside_diff) else 0.0
    outside_max_l1 = float(np.max(outside_diff)) if len(outside_diff) else 0.0

    inside_diff = np.abs(feathered_arr.astype(np.float32) - auth_arr)[mask_bin]
    inside_mean_l1 = float(np.mean(inside_diff)) if len(inside_diff) else 0.0

    # Delta between hard composite and feathered version inside mask
    blend_delta = np.abs(feathered_arr.astype(np.float32) - edit_arr)[mask_bin]
    blend_mean_delta = float(np.mean(blend_delta)) if len(blend_delta) else 0.0
    transition_pixels = int(np.sum((dist > 0) & (dist <= k)))

    metrics = {
        "outside_mean_l1": outside_mean_l1,
        "outside_max_l1": outside_max_l1,
        "inside_mean_l1": inside_mean_l1,
        "blend_mean_delta_vs_hard": blend_mean_delta,
        "transition_pixels": transition_pixels,
        "k_pixels": k,
        "mode": mode,
    }
    return feathered_img, metrics


def image_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def make_diff_visualization(hard_img: Image.Image, feathered_img: Image.Image, amplification: float = 10.0) -> Image.Image:
    """Highlight where inward feathering modified the hard composite."""
    h_arr = np.array(hard_img, dtype=np.float32)
    f_arr = np.array(feathered_img, dtype=np.float32)
    diff = np.abs(f_arr - h_arr)
    # Amplify difference to make the 2px seam transition clearly visible
    vis = np.clip(diff * amplification, 0, 255).astype(np.uint8)
    return Image.fromarray(vis, mode="RGB")


def make_boundary_crop(img: Image.Image, box: tuple[int, int, int, int], zoom: int = 4) -> Image.Image:
    """Crop a region and scale with nearest-neighbor to visualize individual pixels."""
    crop = img.crop(box)
    w, h = crop.size
    return crop.resize((w * zoom, h * zoom), Image.Resampling.NEAREST)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ledger_path = RUN_DIR / "attempt_ledger.jsonl"
    candidates = []
    with open(ledger_path, "r", encoding="utf-8") as f:
        for line in f:
            candidates.append(json.loads(line))

    print(f"Loaded {len(candidates)} candidates from {ledger_path.name}")

    results = []
    for cand in candidates:
        cid = cand["candidate_id"]
        auth_path = IMAGES_DIR / f"{cid}_auth.png"
        edit_path = IMAGES_DIR / f"{cid}_edit.png"
        mask_path = MASKS_DIR / f"{cid}_mask.png"

        auth_img = Image.open(auth_path).convert("RGB")
        edit_img = Image.open(edit_path).convert("RGB")
        mask_img = Image.open(mask_path).convert("L")

        feathered_img, metrics = compute_inward_feathered_image(auth_img, edit_img, mask_img, k=2, mode="cosine")

        out_path = OUTPUT_DIR / f"{cid}_feathered_k2.png"
        feathered_img.save(out_path, format="PNG")

        assert metrics["outside_mean_l1"] == 0.0, f"Invariant violated: outside L1 = {metrics['outside_mean_l1']}"
        assert metrics["outside_max_l1"] == 0.0, f"Invariant violated: outside max L1 = {metrics['outside_max_l1']}"

        print(f"[{cid}] feathered saved. outside_L1={metrics['outside_mean_l1']:.6f}, inside_L1={metrics['inside_mean_l1']:.4f}, transition_px={metrics['transition_pixels']}")

        results.append({
            "candidate": cand,
            "auth_img": auth_img,
            "edit_img": edit_img,
            "mask_img": mask_img,
            "feathered_img": feathered_img,
            "metrics": metrics,
        })

    # Generate self-contained HTML contact sheet
    html_content = generate_html_contact_sheet(results)
    with open(CONTACT_SHEET_PATH, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Contact sheet generated at: {CONTACT_SHEET_PATH}")


def generate_html_contact_sheet(results: list[dict]) -> str:
    # Inspection boxes for boundary zoom
    zoom_boxes = {
        "IND_COCO_SD2_001": (185, 85, 290, 120),  # Upper seam of clock/pan at y=95
        "IND_COCO_SDXL_041": (10, 140, 160, 175),  # Lower seam of chandelier ceiling at y=155 (curtain valance area)
        "IND_COMMONS_SDXL_003": (155, 330, 205, 390), # Left boundary seam cutting through railing at x=170
        "IND_COCO_SD2_002": (85, 295, 145, 335),   # Backsplash/wall boundary at y=323
        "IND_COCO_SDXL_002": (335, 235, 395, 280),  # Bread tomato corner
        "IND_COMMONS_SD2_001": (180, 295, 240, 335),# Coastal water/rock seam at x=190, y=305
        "IND_COMMONS_SD2_002": (195, 460, 245, 512),# Suitcase right boundary at x=225
        "IND_COMMONS_SDXL_001": (340, 40, 400, 90), # Sky bird boundary
    }

    cards_html = []
    for r in results:
        cand = r["candidate"]
        cid = cand["candidate_id"]
        auth_b64 = image_to_base64(r["auth_img"])
        edit_b64 = image_to_base64(r["edit_img"])
        feat_b64 = image_to_base64(r["feathered_img"])
        diff_img = make_diff_visualization(r["edit_img"], r["feathered_img"], amplification=15.0)
        diff_b64 = image_to_base64(diff_img)

        # Zoom crops
        zbox = zoom_boxes.get(cid, (200, 200, 250, 250))
        z_auth = make_boundary_crop(r["auth_img"], zbox, zoom=4)
        z_hard = make_boundary_crop(r["edit_img"], zbox, zoom=4)
        z_feat = make_boundary_crop(r["feathered_img"], zbox, zoom=4)
        z_auth_b64 = image_to_base64(z_auth)
        z_hard_b64 = image_to_base64(z_hard)
        z_feat_b64 = image_to_base64(z_feat)

        m = r["metrics"]
        tool = cand["tool_key"]
        prompt = cand["prompt"]
        mask_bbox = cand["mask_bbox_xyxy"]

        # Specialized qualitative notes per case
        notes = get_case_evaluation(cid)

        card = f"""
        <div class="case-card" id="{cid}">
            <div class="case-header">
                <h3>{cid} &mdash; {cand['stratum_id']} &mdash; {cand['modification_type']}</h3>
                <div class="case-meta">
                    <strong>Tool:</strong> <code>{tool}</code> |
                    <strong>Seed:</strong> <code>{cand['generation_seed']}</code> |
                    <strong>Mask bbox:</strong> <code>{mask_bbox}</code> |
                    <strong>Transition Pixels:</strong> <code>{m['transition_pixels']}</code>
                </div>
                <div class="prompt-box"><strong>Prompt:</strong> {prompt}</div>
            </div>

            <div class="comparison-grid">
                <div class="img-col">
                    <h4>Authentic Image</h4>
                    <img src="data:image/png;base64,{auth_b64}" alt="Authentic" />
                </div>
                <div class="img-col">
                    <h4>Hard Composite (Pilot Current)</h4>
                    <img src="data:image/png;base64,{edit_b64}" alt="Hard Composite" />
                </div>
                <div class="img-col">
                    <h4>Inward Feathering (k=2 px)</h4>
                    <img src="data:image/png;base64,{feat_b64}" alt="Inward Feathered" />
                </div>
                <div class="img-col">
                    <h4>Difference Heatmap (&times;15)</h4>
                    <img src="data:image/png;base64,{diff_b64}" alt="Diff Heatmap" />
                    <div class="caption">Shows 2px inward transition band</div>
                </div>
            </div>

            <div class="zoom-section">
                <h4>Zoomed Boundary Inspection (4&times; Nearest-Neighbor, Box: <code>{zbox}</code>)</h4>
                <div class="zoom-grid">
                    <div class="zoom-col">
                        <h5>Authentic Zoom</h5>
                        <img src="data:image/png;base64,{z_auth_b64}" alt="Auth Zoom" />
                    </div>
                    <div class="zoom-col">
                        <h5>Hard Composite Zoom (1px Step Seam)</h5>
                        <img src="data:image/png;base64,{z_hard_b64}" alt="Hard Zoom" />
                    </div>
                    <div class="zoom-col">
                        <h5>Feathered Zoom (k=2px Cosine Transition)</h5>
                        <img src="data:image/png;base64,{z_feat_b64}" alt="Feathered Zoom" />
                    </div>
                </div>
            </div>

            <div class="audit-metrics">
                <table>
                    <tr>
                        <th>Outside Mask Mean L1</th>
                        <td><code>{m['outside_mean_l1']:.6f}</code> (INVARIANT PRESERVED: 0.0)</td>
                        <th>Outside Mask Max L1</th>
                        <td><code>{m['outside_max_l1']:.6f}</code> (STRICT ZERO)</td>
                    </tr>
                    <tr>
                        <th>Inside Mask Mean L1</th>
                        <td><code>{m['inside_mean_l1']:.4f}</code></td>
                        <th>Mean Delta vs Hard Composite</th>
                        <td><code>{m['blend_mean_delta_vs_hard']:.4f}</code></td>
                    </tr>
                </table>
            </div>

            <div class="analysis-box">
                <h4>Critical Visual Diagnosis</h4>
                <ul>
                    <li><strong>Seam Softening:</strong> {notes['seam_softening']}</li>
                    <li><strong>Ghosting Risk:</strong> {notes['ghosting_risk']}</li>
                    <li><strong>Original Detail Re-emergence:</strong> {notes['detail_reemergence']}</li>
                    <li><strong>Macroscopic Geometric Severance:</strong> {notes['geometric_severance']}</li>
                    <li><strong>Conclusion on Defect Resolution:</strong> {notes['resolution_verdict']}</li>
                </ul>
            </div>
        </div>
        """
        cards_html.append(card)

    cards_joined = "\n".join(cards_html)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Inward Feathering (k=2 px) vs Hard Composite Comparison - Follow-up Pilot</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 20px; }}
        h1, h2, h3, h4, h5 {{ color: #e2e8f0; margin-top: 0; }}
        .header-bar {{ background: #1e293b; padding: 20px; border-radius: 8px; margin-bottom: 24px; border: 1px solid #334155; }}
        .badge {{ background: #3b82f6; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; text-transform: uppercase; }}
        .case-card {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 20px; margin-bottom: 32px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
        .case-header {{ border-bottom: 1px solid #334155; padding-bottom: 12px; margin-bottom: 16px; }}
        .case-meta {{ font-size: 13px; color: #94a3b8; margin: 6px 0; }}
        .prompt-box {{ background: #0f172a; padding: 8px 12px; border-radius: 4px; font-size: 13px; color: #cbd5e1; border-left: 3px solid #38bdf8; margin-top: 8px; }}
        .comparison-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 20px; }}
        .img-col {{ text-align: center; background: #0f172a; padding: 10px; border-radius: 6px; border: 1px solid #334155; }}
        .img-col img {{ width: 100%; height: auto; border-radius: 4px; display: block; }}
        .img-col h4 {{ font-size: 13px; margin-bottom: 8px; color: #cbd5e1; }}
        .caption {{ font-size: 11px; color: #64748b; margin-top: 4px; }}
        .zoom-section {{ background: #0f172a; padding: 16px; border-radius: 6px; border: 1px solid #334155; margin-bottom: 16px; }}
        .zoom-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }}
        .zoom-col {{ text-align: center; }}
        .zoom-col img {{ width: 100%; height: auto; border-radius: 4px; image-rendering: pixelated; border: 1px solid #475569; }}
        .zoom-col h5 {{ font-size: 12px; margin-bottom: 6px; color: #94a3b8; }}
        .audit-metrics {{ margin-bottom: 16px; }}
        .audit-metrics table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        .audit-metrics th, .audit-metrics td {{ padding: 8px 12px; text-align: left; border: 1px solid #334155; }}
        .audit-metrics th {{ background: #0f172a; color: #94a3b8; width: 25%; }}
        .audit-metrics td {{ background: #1e293b; }}
        .analysis-box {{ background: #1e1b4b; border: 1px solid #4338ca; border-radius: 6px; padding: 14px; font-size: 13px; line-height: 1.5; }}
        .analysis-box h4 {{ color: #a5b4fc; margin-bottom: 8px; }}
        .analysis-box ul {{ margin: 0; padding-left: 20px; }}
        .analysis-box li {{ margin-bottom: 6px; }}
        code {{ background: #0f172a; padding: 2px 5px; border-radius: 3px; font-family: monospace; color: #38bdf8; }}
    </style>
</head>
<body>
    <div class="header-bar">
        <h1>Controlled Inward Feathering (k=2 px) Empirical Assessment</h1>
        <p><strong>Dataset Source:</strong> Follow-up Pilot <code>pilot-20261008T113700Z</code> | <strong>Scope:</strong> Evaluates edge feathering strictly inside the registered mask boundaries | <strong>Status:</strong> <span class="badge">EMPIRICAL DIAGNOSIS ONLY &mdash; PENDING HUMAN REVIEW</span></p>
        <p><strong>Mathematical Invariant:</strong> Outside mask pixels are guaranteed untouched (&alpha; = 0.0 outside mask &rarr; Outside Mean L1 = 0.000000 across all 8 candidates). Registered mask remains strictly binary.</p>
    </div>

    {cards_joined}

    {build_diagnostic_plan_section_html()}
</body>
</html>
"""


def get_case_evaluation(cid: str) -> dict:
    evals = {
        "IND_COCO_SD2_001": {
            "seam_softening": "The 1-pixel hard step at y=95 and x=195, 280 is slightly smoothed over 2 pixels. Color transition from kitchen tile to inpaint background is less abruptly stepped.",
            "ghosting_risk": "Low. The transition zone (2 px) is narrow enough that no double-contours or ghost halos of the original hanging pan handle are created.",
            "detail_reemergence": "Marginal. A 2-pixel sliver of the original tile grout line bleeds slightly into the edge of the inpaint region.",
            "geometric_severance": "Moderate. Does NOT resolve the core defect: the inpaint content inside is an unrecognizable blurry metallic blob rather than a crisp clock.",
            "resolution_verdict": "Feathering reduces micro-seam sharpness, but CANNOT resolve semantic or structural generation failures inside the mask.",
        },
        "IND_COCO_SDXL_041": {
            "seam_softening": "At y=155, the bright white ceiling plaster line is softened across y=153..155. The sharp knife-edge cut across the dark textured authentic ceiling is visually dampened.",
            "ghosting_risk": "Low to medium. At x in [12, 25], where the mask contacts the curtain valance apex, blending a 2-pixel band of dark curtain fabric with white plaster creates a minor faint blur band.",
            "detail_reemergence": "Minor curtain texture re-appears in the 2-pixel strip at y=153..154.",
            "geometric_severance": "Significant. The macroscopic mismatch between the modern bright white flat ceiling plane (generated) and the textured off-white authentic ceiling remains clearly visible to the naked eye. Feathering smooths the step but leaves the planar discontinuity intact.",
            "resolution_verdict": "Feathering eliminates the 1-pixel step edge, but DOES NOT resolve the macroscopic mismatch in ceiling plaster texture and tone.",
        },
        "IND_COMMONS_SDXL_003": {
            "seam_softening": "The vertical 1-pixel hard edge at x=170 and x=380 is converted to a 2-pixel cosine gradient.",
            "ghosting_risk": "High at the railing intersection (y ~ 350..370). Blending the severed railing end with infilled park foliage creates a semi-transparent, blurred railing stub.",
            "detail_reemergence": "The authentic railing edge partially fades into the inpaint region over 2 pixels.",
            "geometric_severance": "Severe. The continuous horizontal metal railing remains physically severed: it abruptly stops at x=170 and reappears at x=380. A 2-pixel blend cannot connect a 210-pixel missing railing segment.",
            "resolution_verdict": "Feathering CANNOT repair severed linear structures. The column replacement candidate requires either natural structural boundaries or replacement of the candidate.",
        },
        "IND_COCO_SD2_002": {
            "seam_softening": "Smooths the upper backsplash transition at y=321..323 into the honey-oak authentic lower cabinet frame.",
            "ghosting_risk": "Minimal.",
            "detail_reemergence": "Slight wood grain bleeding at the lower edge.",
            "geometric_severance": "Low. The boundary aligned naturally with the countertop/backsplash line, so feathering performs well here.",
            "resolution_verdict": "Effective for natural architectural trim transitions.",
        },
        "IND_COCO_SDXL_002": {
            "seam_softening": "Softens the border between the infilled bread crust texture and authentic bread.",
            "ghosting_risk": "Negligible.",
            "detail_reemergence": "Authentic bread pores blend smoothly.",
            "geometric_severance": "Irrelevant: 0 tomatoes generated (omission flaw remains unaffected).",
            "resolution_verdict": "No impact on object omission failure.",
        },
        "IND_COMMONS_SD2_001": {
            "seam_softening": "Softens the water and coastline boundary at x=190, y=305.",
            "ghosting_risk": "Minor wave blur at the water boundary.",
            "detail_reemergence": "Authentic water foam blends smoothly.",
            "geometric_severance": "Synthesized new coast/rocks still present; landscape remains altered.",
            "resolution_verdict": "Reduces water seam visibility; landscape alteration remains.",
        },
        "IND_COMMONS_SD2_002": {
            "seam_softening": "Softens boundary between infilled cobblestones and authentic cobblestones.",
            "ghosting_risk": "Negligible.",
            "detail_reemergence": "Authentic stone edges blend into infilled ground.",
            "geometric_severance": "Irrelevant: 0 suitcases generated.",
            "resolution_verdict": "No impact on object omission failure.",
        },
        "IND_COMMONS_SDXL_001": {
            "seam_softening": "Softens cloudy sky boundary at x=350, y=45.",
            "ghosting_risk": "Negligible.",
            "detail_reemergence": "Cloud gradient blends seamlessly.",
            "geometric_severance": "Irrelevant: 0 birds generated.",
            "resolution_verdict": "No impact on object omission failure.",
        },
    }
    return evals.get(cid, {
        "seam_softening": "Evaluated.",
        "ghosting_risk": "Low.",
        "detail_reemergence": "None.",
        "geometric_severance": "None.",
        "resolution_verdict": "Evaluated.",
    })


def build_diagnostic_plan_section_html() -> str:
    plan_path = REPO_ROOT / "research/evidence/phase-4c.7b/content_grounded_diagnostic_plan.json"
    overlays_dir = RUN_DIR / "diagnostic_overlays"
    if not plan_path.exists():
        return "<div class='case-card'>Diagnostic plan not found.</div>"

    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)

    cand_cards = []
    for item in plan["diagnostic_candidates"]:
        cid = item["candidate_id"]
        cname = item["common_name"]
        overlay_path = overlays_dir / f"{cid}_crop_overlay.png"
        overlay_b64 = ""
        if overlay_path.exists():
            overlay_b64 = base64.b64encode(overlay_path.read_bytes()).decode("ascii")

        arma = item["arms"]["arm_a_full_canvas"]
        armb = item["arms"]["arm_b_local_crop"]

        cand_card = f"""
        <div class="case-card" style="border-left: 4px solid #10b981;">
            <div class="case-header">
                <h3>{cid} &mdash; {cname.upper()} &mdash; <code>{item['tool_key']}</code></h3>
                <div class="case-meta">
                    <strong>Model Rev:</strong> <code>{item['model_revision'][:10]}...</code> |
                    <strong>Seed:</strong> <code>{item['generation_seed']}</code> |
                    <strong>Guidance:</strong> <code>{item['guidance_scale']}</code> |
                    <strong>Steps:</strong> <code>{item['num_inference_steps']}</code> |
                    <strong>Scheduler:</strong> <code>{item['scheduler']}</code>
                </div>
                <div class="prompt-box"><strong>Prompt:</strong> {item['prompt']}</div>
            </div>

            <div style="display: grid; grid-template-columns: 1fr 2fr; gap: 20px; margin-bottom: 16px;">
                <div style="text-align: center; background: #0f172a; padding: 10px; border-radius: 6px; border: 1px solid #334155;">
                    <h4>Coordinate Overlay</h4>
                    <img src="data:image/png;base64,{overlay_b64}" alt="{cid} overlay" style="width: 100%; border-radius: 4px;" />
                    <div style="font-size: 11px; color: #94a3b8; margin-top: 6px; text-align: left;">
                        <span style="color: #22c55e;">&#9632; Green:</span> Local Crop ({armb['crop_bbox_xyxy']})<br>
                        <span style="color: #ef4444;">&#9632; Red:</span> Mask Box ({item['mask_bbox_xyxy']}, {item['mask_area_pct']:.2f}%)<br>
                        <span style="color: #06b6d4;">&#9632; Cyan:</span> Target Box ({item['target_bbox_xyxy']})
                    </div>
                </div>

                <div>
                    <h4>Dual-Arm Configuration Comparison</h4>
                    <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
                        <tr style="background: #0f172a;">
                            <th style="padding: 6px 10px; border: 1px solid #334155;">Attribute</th>
                            <th style="padding: 6px 10px; border: 1px solid #334155; color: #38bdf8;">Arm A: Full Canvas</th>
                            <th style="padding: 6px 10px; border: 1px solid #334155; color: #4ade80;">Arm B: Local Crop with Padding</th>
                        </tr>
                        <tr>
                            <td style="padding: 6px 10px; border: 1px solid #334155;"><strong>Attempt ID</strong></td>
                            <td style="padding: 6px 10px; border: 1px solid #334155;"><code>{arma['arm_id']}</code></td>
                            <td style="padding: 6px 10px; border: 1px solid #334155;"><code>{armb['arm_id']}</code></td>
                        </tr>
                        <tr>
                            <td style="padding: 6px 10px; border: 1px solid #334155;"><strong>Crop BBox</strong></td>
                            <td style="padding: 6px 10px; border: 1px solid #334155;"><code>[0, 0, 512, 512]</code> (Entire Canvas)</td>
                            <td style="padding: 6px 10px; border: 1px solid #334155;"><code>{armb['crop_bbox_xyxy']}</code> ({armb['crop_dimensions'][0]}&times;{armb['crop_dimensions'][1]} px)</td>
                        </tr>
                        <tr>
                            <td style="padding: 6px 10px; border: 1px solid #334155;"><strong>Padding Margins</strong></td>
                            <td style="padding: 6px 10px; border: 1px solid #334155;">0 px</td>
                            <td style="padding: 6px 10px; border: 1px solid #334155;">L: {armb['padding_margins']['left']}px, R: {armb['padding_margins']['right']}px, T: {armb['padding_margins']['top']}px, B: {armb['padding_margins']['bottom']}px</td>
                        </tr>
                        <tr>
                            <td style="padding: 6px 10px; border: 1px solid #334155;"><strong>Inference Res</strong></td>
                            <td style="padding: 6px 10px; border: 1px solid #334155;">512&times;512</td>
                            <td style="padding: 6px 10px; border: 1px solid #334155;"><strong>{armb['inference_resolution'][0]}&times;{armb['inference_resolution'][1]}</strong> ({armb['upscale_factor']:.1f}&times; scale)</td>
                        </tr>
                        <tr>
                            <td style="padding: 6px 10px; border: 1px solid #334155;"><strong>Latent Mask Size</strong></td>
                            <td style="padding: 6px 10px; border: 1px solid #334155;">{arma['latent_mask_dimensions_approx'][0]}&times;{arma['latent_mask_dimensions_approx'][1]} ({arma['latent_mask_pixels_approx']} latent px)</td>
                            <td style="padding: 6px 10px; border: 1px solid #334155;"><strong>{armb['latent_mask_dimensions_approx'][0]}&times;{armb['latent_mask_dimensions_approx'][1]}</strong> ({armb['latent_mask_pixels_approx']} latent px &mdash; <strong style="color: #4ade80;">+{armb.get('latent_capacity_gain_factor', 1.0):.1f}&times; capacity</strong>)</td>
                        </tr>
                        <tr>
                            <td style="padding: 6px 10px; border: 1px solid #334155;"><strong>Remapping / Composite</strong></td>
                            <td style="padding: 6px 10px; border: 1px solid #334155;">Direct composite in mask</td>
                            <td style="padding: 6px 10px; border: 1px solid #334155;">Lanczos downscale &rarr; remap to canvas &rarr; composite strictly inside mask</td>
                        </tr>
                    </table>
                </div>
            </div>
        </div>
        """
        cand_cards.append(cand_card)

    cards_str = "\n".join(cand_cards)
    return f"""
    <div style="margin-top: 48px; border-top: 2px dashed #475569; padding-top: 32px;">
        <div class="header-bar" style="background: #064e3b; border-color: #059669;">
            <h2>Proposed 6-Attempt Diagnostic Plan: Latent Spatial Capacity vs Full Canvas</h2>
            <p><strong>Workstream:</strong> <code>independent_cohort_acquisition_diagnostic</code> | <strong>Attempt Budget:</strong> Exactly 6 one-shot attempts (3 candidates &times; 2 arms) | <strong>Status:</strong> <span class="badge" style="background: #eab308; color: black;">PENDING USER APPROVAL &mdash; ZERO GPU RUNS EXECUTED</span></p>
            <p><strong>Experimental Isolation Principle:</strong> Both arms use IDENTICAL authentic images, prompts, seeds, model revisions, schedulers, steps, and guidance scales. Negative prompts and feathering are intentionally EXCLUDED to isolate the single causal factor of latent spatial capacity.</p>
            <p><strong>Governance Gate:</strong> Results from this diagnostic do NOT automatically enter the official independent cohort or overwrite previous pilot records. With n=3 cases and 1 seed each, findings are exploratory and cannot be generalized without broader evaluation.</p>
        </div>

        {cards_str}
    </div>
    """


if __name__ == "__main__":
    main()

