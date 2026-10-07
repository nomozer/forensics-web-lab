#!/usr/bin/env python3
"""Generate a read-only HTML review sheet for a content-grounded edit plan."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path


def generate_contact_sheet(run_dir: Path, edit_plan_path: Path, output_path: Path) -> None:
    plan = json.loads(edit_plan_path.read_text(encoding="utf-8"))
    cards: list[str] = []
    for entry in plan["candidates"]:
        candidate_id = entry["candidate_id"]
        auth_path = run_dir / "images" / f"{candidate_id}_auth.png"
        if not auth_path.is_file():
            raise FileNotFoundError(f"Missing authentic image for {candidate_id}: {auth_path}")
        rel = Path(os.path.relpath(auth_path, output_path.parent)).as_posix()
        mx0, my0, mx1, my1 = entry["mask_bbox_xyxy"]
        tx0, ty0, tx1, ty1 = entry["target_bbox_xyxy"]
        cards.append(
            f"""
<section class="card">
  <h2>{html.escape(candidate_id)}</h2>
  <div class="layout">
    <div class="canvas">
      <img src="{html.escape(rel)}" alt="{html.escape(candidate_id)} authentic image">
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
    </dl>
  </div>
</section>"""
        )

    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Content-grounded edit plan review</title>
<style>
body {{ margin: 24px; background: #0f172a; color: #e2e8f0; font-family: system-ui, sans-serif; }}
h1 {{ color: #7dd3fc; }} .legend {{ margin-bottom: 20px; }}
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
