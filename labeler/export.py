#!/usr/bin/env python3
"""
Apple Dataset Export Script
Exports Stage 1 (Good/Bad) and Stage 2 (Defect Categories) datasets,
computes quality metrics, validates sample balance, and flags review items.
Dependency-free: uses standard library (csv, collections, pathlib).
"""

import os
import sys
import csv
import shutil
from pathlib import Path
from collections import Counter, defaultdict


def run_export(base_dir=None):
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent
    else:
        base_dir = Path(base_dir)

    labels_csv = base_dir / "labels.csv"
    if not labels_csv.exists():
        msg = f"Error: {labels_csv} does not exist. Label images first!"
        print(msg)
        return msg

    # Resolve image source paths
    cut_candidates = [
        base_dir / "apples_clean" / "jpg" / "ilyessaadna",
        base_dir / "apples_clean" / "jpg",
        base_dir / "output" / "processed_gray"
    ]
    cut_dir = None
    for c in cut_candidates:
        if c.exists() and any(c.glob("*.jpg")):
            cut_dir = c
            break

    cutrgba_candidates = [
        base_dir / "apples_clean" / "png_rgba" / "ilyessaadna",
        base_dir / "apples_clean" / "png_rgba",
        base_dir / "output" / "cutouts_rgba"
    ]
    cutrgba_dir = None
    for c in cutrgba_candidates:
        if c.exists() and any(c.glob("*.png")):
            cutrgba_dir = c
            break

    export_dir = base_dir / "dataset_export"
    stage1_dir = export_dir / "stage1"
    stage1_rgba_dir = export_dir / "stage1_rgba"
    stage2_dir = export_dir / "stage2"

    # Reset export directories
    for d in [stage1_dir, stage1_rgba_dir, stage2_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(labels_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    print(f"Read {len(rows)} rows from {labels_csv}")

    # Stage 1: Good vs Bad (Exclude unsure, unlabeled, deleted)
    stage1_rows = [r for r in rows if r.get("label") in ["good", "bad"]]

    copied_s1 = 0
    copied_s1_rgba = 0
    copied_s2 = 0

    for row in stage1_rows:
        fname = row.get("filename", "")
        stem = Path(fname).stem
        lbl = row.get("label", "")

        # Find gray cutout
        found_gray = None
        if cut_dir:
            p = cut_dir / f"{stem}.jpg"
            if p.exists():
                found_gray = p
        if not found_gray:
            # Fallback search
            matches = list(base_dir.rglob(f"{stem}.jpg"))
            if matches:
                found_gray = matches[0]

        if found_gray:
            dest_dir = stage1_dir / lbl
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(found_gray, dest_dir / found_gray.name)
            copied_s1 += 1

            # Stage 2: Bad apples by primary defect
            if lbl == "bad":
                primary = row.get("defect_primary", "").strip() or "unclassified_defect"
                dest_s2 = stage2_dir / primary
                dest_s2.mkdir(parents=True, exist_ok=True)
                shutil.copy2(found_gray, dest_s2 / found_gray.name)
                copied_s2 += 1

        # Copy RGBA if present
        if cutrgba_dir:
            p_rgba = cutrgba_dir / f"{stem}.png"
            if p_rgba.exists():
                dest_rgba = stage1_rgba_dir / lbl
                dest_rgba.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p_rgba, dest_rgba / p_rgba.name)
                copied_s1_rgba += 1

    # Save export labels CSV
    if stage1_rows:
        headers = list(stage1_rows[0].keys())
        with open(export_dir / "labels.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(stage1_rows)

    # ==========================================
    # SUMMARY & VALIDATION REPORT
    # ==========================================
    report_lines = []
    def log(msg=""):
        print(msg)
        report_lines.append(msg)

    s1_good_cnt = sum(1 for r in stage1_rows if r.get("label") == "good")
    s1_bad_cnt = sum(1 for r in stage1_rows if r.get("label") == "bad")

    log("\n" + "="*50)
    log(" DATASET EXPORT SUMMARY")
    log("="*50)
    log(f"Stage 1 Images Exported: {copied_s1} (Good: {s1_good_cnt}, Bad: {s1_bad_cnt})")
    if copied_s1_rgba > 0:
        log(f"Stage 1 RGBA Cutouts Exported: {copied_s1_rgba}")
    log(f"Stage 2 Defect Images Exported: {copied_s2}")

    log("\n--- Counts per Label ---")
    label_counts = Counter(r.get("label") or "unlabeled" for r in rows)
    for lbl, count in label_counts.most_common():
        log(f"  {lbl:<15}: {count}")

    log("\n--- Counts per Lighting (Stage 1) ---")
    lighting_counts = Counter(r.get("lighting") or "unspecified" for r in stage1_rows)
    for lt, count in lighting_counts.most_common():
        log(f"  {lt:<15}: {count}")

    log("\n--- Counts per Variety (Stage 1) ---")
    variety_counts = Counter(r.get("variety") or "unspecified" for r in stage1_rows)
    for v, count in variety_counts.most_common():
        log(f"  {v:<15}: {count}")

    log("\n--- Stage 2 Defect Breakdown (Bad Apples) ---")
    bad_rows = [r for r in stage1_rows if r.get("label") == "bad"]
    defect_counts = Counter(r.get("defect_primary") or "unclassified" for r in bad_rows)
    for dt, count in defect_counts.most_common():
        log(f"  {dt:<15}: {count}")

    # 2x2 Contingency Table
    contingency = {
        "good": {"normal": 0, "dim": 0},
        "bad": {"normal": 0, "dim": 0}
    }
    for r in stage1_rows:
        lbl = r.get("label")
        lt = r.get("lighting")
        if lbl in contingency and lt in contingency[lbl]:
            contingency[lbl][lt] += 1

    log("\n--- 2x2 Contingency Table (Label x Lighting) ---")
    log(f"{'':<10} {'normal':<10} {'dim':<10} {'Total':<10}")
    g_norm = contingency["good"]["normal"]
    g_dim = contingency["good"]["dim"]
    b_norm = contingency["bad"]["normal"]
    b_dim = contingency["bad"]["dim"]
    log(f"{'good':<10} {g_norm:<10} {g_dim:<10} {g_norm + g_dim:<10}")
    log(f"{'bad':<10} {b_norm:<10} {b_dim:<10} {b_norm + b_dim:<10}")
    log(f"{'Total':<10} {g_norm + b_norm:<10} {g_dim + b_dim:<10} {len(stage1_rows):<10}")

    # Warnings
    warnings = []
    log("\n--- Quality & Distribution Warnings ---")
    for lbl in ["good", "bad"]:
        for light in ["normal", "dim"]:
            val = contingency[lbl][light]
            if val < 30:
                warnings.append(f"Cell ({lbl}, {light}) has only {val} images (< 30). Potential bias risk.")

    # Defect type check
    for dt, cnt in defect_counts.items():
        if cnt < 30 and dt != "unclassified":
            warnings.append(f"Defect type '{dt}' has only {cnt} images (< 30). Suggest merging into 'other'.")

    # Variety balance check
    var_good = Counter(r.get("variety") for r in stage1_rows if r.get("label") == "good" and r.get("variety"))
    var_bad = Counter(r.get("variety") for r in stage1_rows if r.get("label") == "bad" and r.get("variety"))
    all_vars = set(var_good.keys()) | set(var_bad.keys())

    for var in sorted(all_vars):
        g_cnt = var_good[var]
        b_cnt = var_bad[var]
        total_var = g_cnt + b_cnt
        if total_var > 10:
            ratio = max(g_cnt, b_cnt) / max(1, min(g_cnt, b_cnt))
            if ratio > 4.0:
                warnings.append(f"Variety '{var}' is heavily imbalanced: Good={g_cnt}, Bad={b_cnt} (ratio {ratio:.1f}:1).")

    if warnings:
        for w in warnings:
            log(f"⚠️  [WARNING] {w}")
    else:
        log("✅ No class balance warnings! All 2x2 cells and defect categories meet minimum targets.")

    # Segmentation Problems Review List
    seg_problems = [r for r in rows if str(r.get("seg_problem", "0")) == "1"]
    log(f"\n--- Segmentation Problems Review ({len(seg_problems)} flagged images) ---")
    if seg_problems:
        for idx, r in enumerate(seg_problems, 1):
            log(f"  {idx}. {r.get('filename')} (label: {r.get('label') or 'unlabeled'}, defect: {r.get('defect_primary') or 'none'})")
    else:
        log("  None. All images have clean segmentation.")

    log("\n" + "="*50)
    log(f"Export directory: {export_dir}")
    log("="*50 + "\n")

    return "\n".join(report_lines)


if __name__ == "__main__":
    run_export()
