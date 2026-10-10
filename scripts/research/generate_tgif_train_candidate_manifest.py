"""Generate machine-readable pending candidate manifest for TGIF-Train-Clean-Subset.

Phase 4C.7B trace - Preregistration and Intake Preparation.
Rules strictly enforced:
1. 1:1 pairing: exactly one unique task per unique COCO source.
2. Inpainting type: segmentation masks only ('segm', PS adapted mask '..._mask_segm.png_ps_mask.png').
3. Variation: SD2-sp variation 0 ('..._sd2_0.png').
4. Disjoint guard: 0 overlap with 684 historical Option P sources and 0 overlap with Phase 4C.7B development candidates.
5. Canvas normalization: 512x512 Lanczos center-crop for images, Nearest-neighbor center-crop for masks.
6. Area brackets: small [1%, 10%), medium [10%, 30%), large [30%, 50%], excluding < 1% and > 50%.
7. Fixed deterministic PRNG seed: 20261010.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("Step 1: Collecting forbidden sources...")
    # 1. Option P historical sources
    option_p_csv = REPO_ROOT / "data/research/tgif/manifests/manifest_pilot_a_option_p.csv"
    option_p_ids: set[str] = set()
    with open(option_p_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = row["source_id"].strip()
            if sid:
                option_p_ids.add(str(int(sid)))

    # 2. Phase 4C.7B development sources
    phase_4c7b_ids: set[str] = set()
    catalog_p = REPO_ROOT / "research/evidence/phase-4c.7b/verified_candidate_catalog_v2.json"
    if catalog_p.exists():
        with open(catalog_p, "r", encoding="utf-8") as f:
            data = json.load(f)
            for c in data.get("coco_candidates", []):
                cid = str(c.get("coco_image_id", "")).strip()
                if cid:
                    phase_4c7b_ids.add(str(int(cid)))

    ext_p = REPO_ROOT / "research/evidence/phase-4c.7b/candidate_catalog_extension_v1.0.0.json"
    if ext_p.exists():
        with open(ext_p, "r", encoding="utf-8") as f:
            data = json.load(f)
            for c in data.get("candidates", []):
                orig = str(c.get("origin_id", ""))
                if "coco:" in orig:
                    phase_4c7b_ids.add(str(int(orig.replace("coco:", "").strip())))

    for p in (REPO_ROOT / "research/evidence/phase-4c.7b").glob("*.json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                text = f.read()
                import re
                for m in re.finditer(r"coco:(\d+)", text):
                    phase_4c7b_ids.add(str(int(m.group(1))))
        except Exception:
            pass

    forbidden_ids = option_p_ids.union(phase_4c7b_ids)
    print(f"Total forbidden source IDs: {len(forbidden_ids)} (Option P: {len(option_p_ids)}, Phase 4C.7B: {len(phase_4c7b_ids)})")

    print("Step 2: Processing TGIF training segmentation masks...")
    train_mask_dir = REPO_ROOT / "data/research/tgif/masks/training"
    ps_masks = sorted(list(train_mask_dir.glob("**/*_mask_segm.png_ps_mask.png")))
    print(f"Total training segm ps_masks on disk: {len(ps_masks)}")

    def transform_mask_512(mask_p: Path) -> tuple[int, float, tuple[int, int]]:
        mask = Image.open(mask_p)
        w, h = mask.size
        tw, th = (512, 512)
        scale = max(tw / w, th / h)
        nw = int(round(w * scale))
        nh = int(round(h * scale))
        mask_c = mask.resize((nw, nh), Image.Resampling.NEAREST)
        left = (nw - tw) // 2
        top = (nh - th) // 2
        mask_512 = mask_c.crop((left, top, left + tw, top + th))
        arr = np.array(mask_512)
        if arr.ndim == 3:
            bin_arr = (np.max(arr, axis=2) > 128).astype(np.uint8)
        else:
            bin_arr = (arr > 128).astype(np.uint8)
        px = int(np.sum(bin_arr))
        pct = px / (512 * 512)
        return px, pct, (w, h)

    def classify_bracket(pct: float) -> str:
        if pct < 0.01:
            return "excluded_under_1pct"
        elif pct < 0.10:
            return "small_under_10pct"
        elif pct < 0.30:
            return "medium_10_to_30pct"
        elif pct <= 0.50:
            return "large_over_30pct"
        else:
            return "excluded_over_50pct"

    tasks_by_source: dict[str, list[dict[str, Any]]] = {}
    for p in ps_masks:
        raw_id = p.name.split("_mask_")[0]
        sid = str(int(raw_id))
        if sid in forbidden_ids:
            continue
        mask_bytes = p.stat().st_size
        mask_sha = sha256_file(p)
        px, pct, orig_dim = transform_mask_512(p)
        bracket = classify_bracket(pct)
        if bracket.startswith("excluded"):
            continue
        if sid not in tasks_by_source:
            tasks_by_source[sid] = []
        rel_mask_path = str(p.relative_to(REPO_ROOT)).replace("\\", "/")
        category = p.parent.name
        tasks_by_source[sid].append({
            "source_id": sid,
            "raw_id": raw_id,
            "category": category,
            "task_id": f"tgif_train_{raw_id}_{category}_segm",
            "mask_type": "segm",
            "variation_idx": 0,
            "mask_rel_path": rel_mask_path,
            "mask_sha256": mask_sha,
            "mask_file_bytes": mask_bytes,
            "orig_rel_path_in_archive": f"orig/training/{category}/{raw_id}_orig.png",
            "sd2_rel_path_in_archive": f"sd2-sp/training/{category}/{raw_id}_mask_segm.png_ps_mask.png_sd2_0.png",
            "canvas_size": [512, 512],
            "orig_native_size": list(orig_dim),
            "mask_px_512": px,
            "mask_pct_512": round(pct, 6),
            "stratum_area_class": bracket,
        })

    print(f"Total eligible unique sources with at least one valid task: {len(tasks_by_source)}")

    print("Step 3: Selecting 1 task per source hierarchically with fixed seed...")
    rng = np.random.default_rng(20261010)

    # Hierarchical stratum selection: Large > Medium > Small
    large_sources: list[dict[str, Any]] = []
    medium_sources: list[dict[str, Any]] = []
    small_sources: list[dict[str, Any]] = []

    for sid in sorted(tasks_by_source.keys()):
        tlist = tasks_by_source[sid]
        tlist_sorted = sorted(tlist, key=lambda x: (x["category"], x["mask_rel_path"]))
        has_large = [t for t in tlist_sorted if t["stratum_area_class"] == "large_over_30pct"]
        has_medium = [t for t in tlist_sorted if t["stratum_area_class"] == "medium_10_to_30pct"]
        has_small = [t for t in tlist_sorted if t["stratum_area_class"] == "small_under_10pct"]

        if has_large:
            idx = int(rng.integers(0, len(has_large))) if len(has_large) > 1 else 0
            large_sources.append(has_large[idx])
        elif has_medium:
            idx = int(rng.integers(0, len(has_medium))) if len(has_medium) > 1 else 0
            medium_sources.append(has_medium[idx])
        elif has_small:
            idx = int(rng.integers(0, len(has_small))) if len(has_small) > 1 else 0
            small_sources.append(has_small[idx])

    print(f"Hierarchical source counts: Large={len(large_sources)}, Medium={len(medium_sources)}, Small={len(small_sources)}")

    # Sort each stratum list deterministically using fixed seed permutation
    large_indices = rng.permutation(len(large_sources)).tolist()
    medium_indices = rng.permutation(len(medium_sources)).tolist()
    small_indices = rng.permutation(len(small_sources)).tolist()

    large_ordered = [large_sources[i] for i in large_indices]
    medium_ordered = [medium_sources[i] for i in medium_indices]
    small_ordered = [small_sources[i] for i in small_indices]

    for rank, item in enumerate(large_ordered, 1):
        item["rank_in_stratum"] = rank
    for rank, item in enumerate(medium_ordered, 1):
        item["rank_in_stratum"] = rank
    for rank, item in enumerate(small_ordered, 1):
        item["rank_in_stratum"] = rank

    # Allocation for Target 400:
    # Large: 14 (100% available)
    # Medium: 221
    # Small: 165
    target_400_sources: list[dict[str, Any]] = []
    for item in large_ordered[:14]:
        item["selected_in_n400"] = True
        item["selected_in_n200"] = True
        target_400_sources.append(item)

    for item in medium_ordered[:221]:
        item["selected_in_n400"] = True
        target_400_sources.append(item)
    for item in medium_ordered[221:]:
        item["selected_in_n400"] = False

    for item in small_ordered[:165]:
        item["selected_in_n400"] = True
        target_400_sources.append(item)
    for item in small_ordered[165:]:
        item["selected_in_n400"] = False

    # Mark n200 selections:
    # 14 Large + 93 Medium + 93 Small = 200
    for idx, item in enumerate(medium_ordered):
        item["selected_in_n200"] = (idx < 93)
    for idx, item in enumerate(small_ordered):
        item["selected_in_n200"] = (idx < 93)

    all_pool_sources = large_ordered + medium_ordered + small_ordered
    print(f"Total pool sources cataloged: {len(all_pool_sources)}")
    print(f"Total selected in N=400: {len(target_400_sources)}")

    # Verify each selected mask on disk matches mask_sha256
    print("Verifying 400 selected masks on disk against computed SHA-256...")
    for idx, s in enumerate(target_400_sources, 1):
        mp = REPO_ROOT / s["mask_rel_path"]
        assert mp.is_file(), f"Selected mask not found on disk: {mp}"
        disk_sha = sha256_file(mp)
        assert disk_sha == s["mask_sha256"], f"Mask hash mismatch for {s['task_id']}: {disk_sha} != {s['mask_sha256']}"
    print("100% of 400 selected masks verified bit-identical on disk.")

    manifest_metadata = {
        "schema_version": "1.0.0",
        "phase": "4C.7B",
        "manifest_name": "tgif_train_candidate_manifest_pending",
        "status": "APPROVED_BY_HUMAN_INTAKE_ONLY",
        "generated_at_utc": "2026-10-09T14:20:00Z",
        "selection_seed": 20261010,
        "rng_engine": "PCG64",
        "pairing_contract": "1_to_1_unique_source_pairing",
        "inpainting_tool": "stable_diffusion_2_inpainting (sd2-sp)",
        "inpainting_variation": 0,
        "mask_type": "segm (Photoshop adapted inpainting mask, ..._mask_segm.png_ps_mask.png)",
        "preprocessing_pipeline": {
            "canvas_size": [512, 512],
            "image_transform": "PIL.Image.Resampling.LANCZOS center-crop to 512x512",
            "mask_transform": "PIL.Image.Resampling.NEAREST center-crop to 512x512, binarized at >128 to {0, 255}",
            "letterboxing_forbidden": True
        },
        "disjoint_guard_verification": {
            "forbidden_option_p_historical_sources": len(option_p_ids),
            "forbidden_phase_4c7b_development_sources": len(phase_4c7b_ids),
            "total_forbidden_ids": len(forbidden_ids),
            "source_id_collision_count": 0,
            "status": "PASS_AT_METADATA_AND_SOURCE_ID_LEVEL"
        },
        "stratum_pool_summary": {
            "large_over_30pct": len(large_ordered),
            "medium_10_to_30pct": len(medium_ordered),
            "small_under_10pct": len(small_ordered),
            "total_eligible_unique_sources": len(all_pool_sources)
        },
        "human_approval_registered": {
            "reviewer": "Dũng Phạm <valdung04@gmail.com>",
            "decision": "APPROVED_OPTION_N400_INTAKE_ONLY",
            "approved_at_utc": "2026-10-09T14:19:44Z",
            "approval_scope": "INTAKE_ONLY (Colab CPU download of orig_training.tar.gz and sd2-sp_training.tar.gz ~18.63 GiB, selective extraction of 400 pairs, return zip ~120-160 MB to local. Detector/evaluation and training NOT PERMITTED)",
            "selected_allocation": {
                "total_pairs": 400,
                "large_over_30pct": 14,
                "medium_10_to_30pct": 221,
                "small_under_10pct": 165
            },
            "scientific_honesty_caveats": [
                "Không coi phân bổ này là tỷ lệ tự nhiên; nhóm large chỉ có 14 nguồn nên không đưa ra kết luận mạnh riêng cho nhóm đó.",
                "Tuyên bố 'N=400 đảm bảo ME<0,01' chưa xác lập cho thiết kế mới; kiểm tra lại căn cứ thống kê trước evaluation.",
                "Bước kiểm tra bộ ba thực hiện chính thức tại LOCAL sau intake (sử dụng 100% masks đã có sẵn trên đĩa local).",
                "Khóa preprocessing và QC; quy tắc xử lý mẫu thiếu/hỏng: FAIL-CLOSED, không tự chọn mẫu thay thế.",
                "Không sửa/composite lại ảnh benchmark; bảo toàn nguyên vẹn độ biến thiên thực tế của tác giả."
            ]
        },
        "deficit_analysis_and_quota_recommendation": {
            "large_over_30pct": {
                "v1_protocol_target_quota_pairs": 120,
                "available_in_tgif_training_pool": 14,
                "quota_deficit": 106,
                "scientific_action": "Strictly report deficit without relaxing mask area thresholds. Take 100% of available large sources (14 pairs)."
            },
            "medium_10_to_30pct": {
                "v1_protocol_target_quota_pairs": 160,
                "available_in_tgif_training_pool": 282,
                "allocated_in_recommended_n400": 221
            },
            "small_under_10pct": {
                "v1_protocol_target_quota_pairs": 120,
                "available_in_tgif_training_pool": 864,
                "allocated_in_recommended_n400": 165
            }
        },
        "cohort_configurations_proposed": {
            "recommended_n400": {
                "total_pairs": 400,
                "large_pairs": 14,
                "medium_pairs": 221,
                "small_pairs": 165,
                "large_pct": 3.5,
                "medium_pct": 55.25,
                "small_pct": 41.25
            },
            "alternative_n200": {
                "total_pairs": 200,
                "large_pairs": 14,
                "medium_pairs": 93,
                "small_pairs": 93,
                "large_pct": 7.0,
                "medium_pct": 46.5,
                "small_pct": 46.5
            }
        },
        "intake_budget_summary": {
            "colab_download_archives_gib": 18.63,
            "colab_archive_orig_gib": 5.26,
            "colab_archive_sd2_gib": 13.37,
            "colab_subset_extracted_authentic_pngs": 400,
            "colab_subset_extracted_edited_pngs": 400,
            "local_download_package_zip_mb": "~120-160 MB",
            "local_resident_masks_mb": 0.0
        }
    }

    manifest_output = {
        "metadata": manifest_metadata,
        "candidates": all_pool_sources
    }

    # 1. Output candidate pool manifest
    out_json = REPO_ROOT / "research/evidence/phase-4c.7b/tgif_train_candidate_manifest_pending.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(manifest_output, f, indent=2, ensure_ascii=False)
    print(f"Wrote JSON pool manifest to {out_json}")

    out_csv = REPO_ROOT / "research/evidence/phase-4c.7b/tgif_train_candidate_manifest_pending.csv"
    csv_fields = [
        "source_id", "raw_id", "category", "task_id", "mask_type", "variation_idx",
        "stratum_area_class", "rank_in_stratum", "selected_in_n400", "selected_in_n200",
        "mask_px_512", "mask_pct_512", "mask_rel_path", "mask_sha256", "mask_file_bytes",
        "orig_rel_path_in_archive", "sd2_rel_path_in_archive"
    ]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for c in all_pool_sources:
            row = {k: c.get(k, "") for k in csv_fields}
            writer.writerow(row)
    print(f"Wrote CSV pool manifest to {out_csv}")

    # 2. Output LOCKED 400-row selection manifest
    locked_metadata = dict(manifest_metadata)
    locked_metadata["manifest_name"] = "tgif_train_clean_subset_manifest_locked_n400"
    locked_metadata["status"] = "LOCKED_APPROVED_SELECTION_N400"
    locked_metadata["selected_cohort_count"] = len(target_400_sources)

    # Re-order target_400_sources deterministically: Large (1..14), Medium (1..221), Small (1..165)
    target_400_sources_sorted = sorted(
        target_400_sources,
        key=lambda x: (
            {"large_over_30pct": 0, "medium_10_to_30pct": 1, "small_under_10pct": 2}[x["stratum_area_class"]],
            x["rank_in_stratum"]
        )
    )

    for overall_idx, item in enumerate(target_400_sources_sorted, 1):
        item["selection_index_1based"] = overall_idx

    locked_json_path = REPO_ROOT / "research/evidence/phase-4c.7b/tgif_train_clean_subset_manifest_locked_n400.json"
    locked_output = {
        "metadata": locked_metadata,
        "selected_candidates": target_400_sources_sorted
    }
    with open(locked_json_path, "w", encoding="utf-8") as f:
        json.dump(locked_output, f, indent=2, ensure_ascii=False)
    print(f"Wrote LOCKED 400 JSON manifest to {locked_json_path}")

    locked_csv_path = REPO_ROOT / "research/evidence/phase-4c.7b/tgif_train_clean_subset_manifest_locked_n400.csv"
    locked_csv_fields = [
        "selection_index_1based", "source_id", "raw_id", "category", "task_id", "mask_type", "variation_idx",
        "stratum_area_class", "rank_in_stratum", "mask_px_512", "mask_pct_512",
        "mask_rel_path", "mask_sha256", "mask_file_bytes",
        "orig_rel_path_in_archive", "sd2_rel_path_in_archive"
    ]
    with open(locked_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=locked_csv_fields)
        writer.writeheader()
        for c in target_400_sources_sorted:
            row = {k: c.get(k, "") for k in locked_csv_fields}
            writer.writerow(row)
    print(f"Wrote LOCKED 400 CSV manifest to {locked_csv_path}")

    pool_json_sha = sha256_file(out_json)
    pool_csv_sha = sha256_file(out_csv)
    locked_json_sha = sha256_file(locked_json_path)
    locked_csv_sha = sha256_file(locked_csv_path)
    print(f"Pool JSON SHA-256: {pool_json_sha}")
    print(f"Pool CSV SHA-256: {pool_csv_sha}")
    print(f"LOCKED 400 JSON SHA-256: {locked_json_sha}")
    print(f"LOCKED 400 CSV SHA-256: {locked_csv_sha}")

if __name__ == "__main__":
    main()
