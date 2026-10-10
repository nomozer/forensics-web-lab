#!/usr/bin/env python3
"""Generate a read-only HTML review sheet for a content-grounded edit plan."""

from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path


def generate_contact_sheet(run_dir: Path, edit_plan_path: Path, output_path: Path) -> None:
    plan = json.loads(edit_plan_path.read_text(encoding="utf-8"))
    cards: list[str] = []
    review_rows: list[str] = []
    for entry in plan["candidates"]:
        candidate_id = entry["candidate_id"]
        auth_path = run_dir / "images" / f"{candidate_id}_auth.png"
        if not auth_path.is_file():
            raise FileNotFoundError(f"Missing authentic image for {candidate_id}: {auth_path}")
        embedded_image = "data:image/png;base64," + base64.b64encode(auth_path.read_bytes()).decode("ascii")
        mx0, my0, mx1, my1 = entry["mask_bbox_xyxy"]
        tx0, ty0, tx1, ty1 = entry["target_bbox_xyxy"]
        mask_ratio = ((mx1 - mx0) * (my1 - my0)) / (512 * 512)
        review = entry.get("instruction_review", {})
        review_rows.append(
            "<tr>"
            f"<td><code>{html.escape(candidate_id)}</code></td>"
            f"<td>{html.escape(str(review.get('target_or_placement', 'NOT_REVIEWED')))}</td>"
            f"<td>{html.escape(str(review.get('bbox_prompt_alignment', 'NOT_REVIEWED')))}</td>"
            f"<td>{html.escape(str(review.get('mask_scope', 'NOT_REVIEWED')))}</td>"
            f"<td>{mask_ratio:.6%}</td>"
            f"<td>{html.escape(str(review.get('agent_status', 'NOT_REVIEWED')))}</td>"
            f"<td>{html.escape(str(review.get('human_decision_required', 'Review the instruction.')))}</td>"
            "</tr>"
        )
        cards.append(
            f"""
<section class="card">
  <h2>{html.escape(candidate_id)}</h2>
  <div class="layout">
    <div class="canvas">
      <img src="{embedded_image}" alt="{html.escape(candidate_id)} authentic image">
      <svg viewBox="0 0 512 512" aria-label="planned target and mask overlay">
        <rect class="mask" x="{mx0}" y="{my0}" width="{mx1-mx0}" height="{my1-my0}" />
        <rect class="target" x="{tx0}" y="{ty0}" width="{tx1-tx0}" height="{ty1-ty0}" />
      </svg>
    </div>
    <dl>
      <dt>Stratum / tool</dt><dd>{html.escape(entry['stratum_id'])} / {html.escape(entry['tool_key'])}</dd>
      <dt>Operation / mask class</dt><dd>{html.escape(entry['modification_type'])} / {html.escape(entry['mask_area_class'])}</dd>
      <dt>Target</dt><dd>{html.escape(entry['target_description'])}</dd>
      <dt>Rationale</dt><dd>{html.escape(entry['placement_rationale'])}</dd>
      <dt>Prompt</dt><dd>{html.escape(entry['prompt'])}</dd>
      <dt>Coordinates</dt><dd>target {entry['target_bbox_xyxy']}; mask {entry['mask_bbox_xyxy']}</dd>
      <dt>Mask area</dt><dd><b>Mask area:</b> {mask_ratio:.6%} of the normalized 512x512 canvas</dd>
      <dt>Agent instruction review</dt><dd>{html.escape(str(review.get('note', 'Not reviewed.')))}</dd>
      <dt>Human decision required</dt><dd>{html.escape(str(review.get('human_decision_required', 'Review the instruction.')))}</dd>
    </dl>
  </div>
</section>"""
        )

    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Content-grounded edit plan review</title>
<style>
body {{ margin: 24px; background: #0f172a; color: #e2e8f0; font-family: system-ui, sans-serif; }}
h1 {{ color: #7dd3fc; }} .legend {{ margin-bottom: 20px; }}
.warning {{ padding: 12px; border-left: 4px solid #facc15; background: #422006; }}
table {{ border-collapse: collapse; width: 100%; margin: 20px 0 28px; }}
th, td {{ border: 1px solid #64748b; padding: 8px; text-align: left; vertical-align: top; }}
th {{ background: #334155; color: #bae6fd; }}
.card {{ background: #1e293b; border: 1px solid #475569; border-radius: 10px; padding: 18px; margin: 18px 0; }}
.layout {{ display: flex; flex-wrap: wrap; gap: 24px; }}
.canvas {{ position: relative; width: 512px; height: 512px; flex: 0 0 512px; }}
.canvas img, .canvas svg {{ position: absolute; inset: 0; width: 512px; height: 512px; }}
.mask {{ fill: rgba(250, 204, 21, .16); stroke: #facc15; stroke-width: 4; }}
.target {{ fill: rgba(34, 197, 94, .14); stroke: #22c55e; stroke-width: 4; stroke-dasharray: 10 7; }}
dl {{ max-width: 650px; margin: 0; }} dt {{ color: #7dd3fc; font-weight: 700; margin-top: 10px; }} dd {{ margin: 3px 0 0; }}
</style></head><body>
<h1>Content-grounded edit plan — human review</h1>
<p><b>Review status:</b> {html.escape(str(plan.get('human_review_status', 'UNSPECIFIED')))}</p>
<p class="legend"><b>Yellow:</b> request mask. <b>Green dashed:</b> intended target/object placement. This is a plan, not generated output or Human Content QC approval.</p>
<p class="warning"><b>Agent pre-screen only:</b> instruction checks below do not approve Human Content QC. A row marked NEEDS_USER_DECISION blocks generation approval.</p>
<h2>Eight-instruction review table</h2>
<table>
  <thead><tr><th>Candidate</th><th>Target / placement</th><th>Bbox + prompt</th><th>Mask scope</th><th>Mask area</th><th>Agent status</th><th>Human decision</th></tr></thead>
  <tbody>{''.join(review_rows)}</tbody>
</table>
{''.join(cards)}
</body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--edit-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate_contact_sheet(args.run_dir, args.edit_plan, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
