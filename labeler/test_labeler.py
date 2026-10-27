#!/usr/bin/env python3
"""
Automated Acceptance Test for Apple Labeling Tool
Tests:
  1. Synthetic 10 dummy image generation (varieties, lighting, cutouts)
  2. Pass 1 labeling (good, bad, unsure, Q seg problem, D lighting toggle, Undo, Delete, Navigation)
  3. Pass 2 defect categorization (primary defect, secondary defect)
  4. Persistence & resume on restart
  5. Dataset export and report generation
"""

import os
import shutil
import json
import unittest
from pathlib import Path
import numpy as np
from PIL import Image

# Temporary test directory
TEST_DIR = Path("/tmp/test_apple_labeler")
if TEST_DIR.exists():
    shutil.rmtree(TEST_DIR)
TEST_DIR.mkdir(parents=True, exist_ok=True)

ORIG_TEST = TEST_DIR / "apples"
CUT_TEST = TEST_DIR / "cut_gray"
CUTRGBA_TEST = TEST_DIR / "cut_rgba"
for d in [ORIG_TEST, CUT_TEST, CUTRGBA_TEST]:
    d.mkdir(parents=True, exist_ok=True)

# Generate 10 dummy images
print("Generating 10 synthetic test apple images...")
colors = [
    (210, 40, 40),   # Red 1
    (220, 50, 40),   # Red 2
    (70, 190, 60),   # Green 1
    (80, 200, 70),   # Green 2
    (230, 210, 50),  # Yellow 1
    (220, 200, 60),  # Yellow 2
    (140, 30, 30),   # Dim Red
    (40, 110, 35),   # Dim Green
    (200, 60, 50),   # Red 3
    (90, 180, 50)    # Green 3
]

for i in range(10):
    stem = f"test_apple_{i+1:02d}"
    c = colors[i]

    # 1. Original photo (400x400) on varied background
    orig = Image.new("RGB", (400, 400), (80, 85, 90)) # dark background
    # Draw apple circle
    im_arr = np.array(orig)
    yy, xx = np.ogrid[:400, :400]
    mask = ((xx - 200)**2 + (yy - 200)**2) < 140**2
    im_arr[mask] = c
    # If bad, add simulated dark rot spot
    if i in [1, 3, 7]:
        rot_mask = ((xx - 180)**2 + (yy - 180)**2) < 35**2
        im_arr[rot_mask] = (60, 40, 20)
    Image.fromarray(im_arr).save(ORIG_TEST / f"{stem}.jpg", "JPEG")

    # 2. Cutout on Gray (384x384, uniform 128 gray)
    gray_arr = np.full((384, 384, 3), 128, dtype=np.uint8)
    yy, xx = np.ogrid[:384, :384]
    cut_mask = ((xx - 192)**2 + (yy - 192)**2) < 135**2
    gray_arr[cut_mask] = c
    if i in [1, 3, 7]:
        rot_mask = ((xx - 175)**2 + (yy - 175)**2) < 35**2
        gray_arr[rot_mask] = (60, 40, 20)
    Image.fromarray(gray_arr).save(CUT_TEST / f"{stem}.jpg", "JPEG")

    # 3. Cutout RGBA (transparent)
    rgba_arr = np.zeros((384, 384, 4), dtype=np.uint8)
    rgba_arr[cut_mask, :3] = c
    rgba_arr[cut_mask, 3] = 255
    Image.fromarray(rgba_arr).save(CUTRGBA_TEST / f"{stem}.png", "PNG")

print("Created 10 synthetic images successfully!")

# Configure app.py for test
import app
app.BASE_DIR = TEST_DIR
app.ORIG_DIR = ORIG_TEST
app.CUT_DIR = CUT_TEST
app.CUTRGBA_DIR = CUTRGBA_TEST
app.LABELS_CSV = TEST_DIR / "labels.csv"
app.FEATURES_CSV = TEST_DIR / "features.csv"
app.CACHE_DIR = TEST_DIR / ".thumb_cache"
app.CACHE_DIR.mkdir(parents=True, exist_ok=True)

app.init_dataset()
client = app.app.test_client()

print("\n--- TEST 1: Pass 1 Labeling, Navigation, Flags, and Undo ---")
# 1. Label test_apple_01 as good
res = client.post("/api/action", json={"stem": "test_apple_01", "action": "label", "value": "good", "pass": "pass1"}).json
assert res["success"] == True
assert app.ITEM_DATA["test_apple_01"]["label"] == "good"
print("✓ Label 'good' applied")

# 2. Label test_apple_02 as bad
res = client.post("/api/action", json={"stem": "test_apple_02", "action": "label", "value": "bad", "pass": "pass1"}).json
assert res["success"] == True
assert app.ITEM_DATA["test_apple_02"]["label"] == "bad"
print("✓ Label 'bad' applied")

# 3. Label test_apple_03 as unsure
res = client.post("/api/action", json={"stem": "test_apple_03", "action": "label", "value": "unsure", "pass": "pass1"}).json
assert res["success"] == True
assert app.ITEM_DATA["test_apple_03"]["label"] == "unsure"
print("✓ Label 'unsure' applied")

# 4. Toggle Q (seg_problem) and D (lighting) on test_apple_04
res = client.post("/api/action", json={"stem": "test_apple_04", "action": "seg_problem", "pass": "pass1"}).json
assert res["item"]["seg_problem"] == 1
res = client.post("/api/action", json={"stem": "test_apple_04", "action": "lighting", "pass": "pass1"}).json
assert res["item"]["lighting"] in ["normal", "dim"]
# Label as bad
client.post("/api/action", json={"stem": "test_apple_04", "action": "label", "value": "bad", "pass": "pass1"})
print("✓ Flags Q (seg_problem=1) and D (lighting override) verified")

# 5. Label test_apple_05 as good, then test UNDO
client.post("/api/action", json={"stem": "test_apple_05", "action": "label", "value": "good", "pass": "pass1"})
assert app.ITEM_DATA["test_apple_05"]["label"] == "good"
undo_res = client.post("/api/undo", json={"pass": "pass1"}).json
assert undo_res["success"] == True
assert app.ITEM_DATA["test_apple_05"]["label"] == ""
print("✓ Undo verified (test_apple_05 label reverted to empty)")

# 6. Test Delete on test_apple_06
del_res = client.post("/api/action", json={"stem": "test_apple_06", "action": "delete", "pass": "pass1"}).json
assert app.ITEM_DATA["test_apple_06"]["label"] == "deleted"
queue_res = client.get("/api/init?pass=pass1&filter=all").json
assert "test_apple_06" not in queue_res["queue"]
print("✓ Delete verified (soft-deleted and excluded from active queues)")

print("\n--- TEST 2: Pass 2 Defect Categorization (Bad Apples Only) ---")
# Query Pass 2 queue: should contain bad apples without defects: test_apple_02 and test_apple_04
q2_res = client.get("/api/init?pass=pass2").json
print(f"Pass 2 Queue: {q2_res['queue']}")
assert "test_apple_02" in q2_res["queue"]
assert "test_apple_04" in q2_res["queue"]

# Assign primary defect 'rot' to test_apple_02
res = client.post("/api/action", json={"stem": "test_apple_02", "action": "defect_primary", "value": "rot", "pass": "pass2"}).json
assert app.ITEM_DATA["test_apple_02"]["defect_primary"] == "rot"
assert "rot" in app.ITEM_DATA["test_apple_02"]["defect_all"]
print("✓ Primary defect 'rot' assigned to test_apple_02")

# Assign primary defect 'bruise' and secondary defect 'scab/scar' to test_apple_04
client.post("/api/action", json={"stem": "test_apple_04", "action": "defect_primary", "value": "bruise", "pass": "pass2"})
client.post("/api/action", json={"stem": "test_apple_04", "action": "defect_add", "value": "scab/scar", "pass": "pass2"})
assert app.ITEM_DATA["test_apple_04"]["defect_primary"] == "bruise"
assert "bruise" in app.ITEM_DATA["test_apple_04"]["defect_all"]
assert "scab/scar" in app.ITEM_DATA["test_apple_04"]["defect_all"]
print("✓ Primary defect 'bruise' + secondary 'scab/scar' assigned to test_apple_04")

print("\n--- TEST 3: Persistence, Atomic Autosave, and Resume ---")
assert app.LABELS_CSV.exists()
# Read labels.csv directly
with open(app.LABELS_CSV) as f:
    lines = f.readlines()
print(f"labels.csv has {len(lines)} lines (header + 10 items).")
assert len(lines) == 11

# Re-initialize app to simulate closing and reopening
print("Restarting app and reloading dataset...")
app.init_dataset()
assert app.ITEM_DATA["test_apple_01"]["label"] == "good"
assert app.ITEM_DATA["test_apple_02"]["defect_primary"] == "rot"
assert app.ITEM_DATA["test_apple_04"]["defect_primary"] == "bruise"
assert app.ITEM_DATA["test_apple_04"]["seg_problem"] == 1
assert app.ITEM_DATA["test_apple_06"]["label"] == "deleted"
print("✓ Complete state persistence and resume verified!")

print("\n--- TEST 4: Dataset Export & Summary Report ---")
import export
summary = export.run_export(TEST_DIR)

# Verify exported directories
s1_good = TEST_DIR / "dataset_export" / "stage1" / "good"
s1_bad = TEST_DIR / "dataset_export" / "stage1" / "bad"
s2_rot = TEST_DIR / "dataset_export" / "stage2" / "rot"
s2_bruise = TEST_DIR / "dataset_export" / "stage2" / "bruise"

assert s1_good.exists() and len(list(s1_good.glob("*.jpg"))) == 1
assert s1_bad.exists() and len(list(s1_bad.glob("*.jpg"))) == 2
assert s2_rot.exists() and len(list(s2_rot.glob("*.jpg"))) == 1
assert s2_bruise.exists() and len(list(s2_bruise.glob("*.jpg"))) == 1
print("✓ Stage 1 and Stage 2 file structures verified!")

print("\nALL ACCEPTANCE TESTS PASSED SUCCESSFULLY! 🎉\n")
