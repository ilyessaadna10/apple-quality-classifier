#!/usr/bin/env python3
"""
Apple Photo Labeling Tool (Fast, Keyboard-Driven Two-Pass Web App)
Single-file Flask application with plain HTML/JS/CSS frontend.
"""

import os
import sys
import glob
import json
import time
import shutil
from datetime import datetime
from pathlib import Path
from io import BytesIO

from flask import Flask, request, jsonify, send_file, render_template_string
from PIL import Image, ImageOps
import numpy as np
import cv2
from skimage.filters import threshold_otsu

# ==========================================
# CONFIGURATION BLOCK
# ==========================================
BASE_DIR = Path(__file__).resolve().parent

# Default folder candidates
def resolve_dir(candidates, default):
    for c in candidates:
        p = Path(c)
        if p.exists() and p.is_dir() and any(p.glob("*")):
            return p
    return Path(default)

ORIG_DIR = resolve_dir(
    [BASE_DIR / "apples", BASE_DIR / "raw_photos"],
    BASE_DIR / "apples"
)

CUT_DIR = BASE_DIR / "apples_clean" / "jpg"
CUTRGBA_DIR = BASE_DIR / "apples_clean" / "png_rgba"

LABELS_CSV = BASE_DIR / "labels.csv"
FEATURES_CSV = BASE_DIR / "features.csv"
CACHE_DIR = BASE_DIR / ".thumb_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFECT_TYPES = ["rot", "bruise", "scab/scar", "hole/worm", "other"]

PORT = 5000
HOST = "0.0.0.0"

# ==========================================
# GLOBAL STATE & CACHE
# ==========================================
app = Flask(__name__)

# In-memory datasets
ITEMS = []            # list of stem strings in sorted order
ITEM_DATA = {}        # stem -> dict of metadata & labels
UNDO_STACK = []       # list of previous states for undo
ACTION_COUNT = 0
OTSU_THRESHOLD = 50.0 # will be calculated

# ==========================================
# FEATURE EXTRACTION ENGINE
# ==========================================
def extract_apple_features(orig_path, cut_path, cutrgba_path):
    """
    Computes mean L*, mean b*, and Laplacian sharpness on apple pixels only.
    """
    try:
        # Load cutout image
        cut_img = None
        has_alpha = False
        alpha_mask = None

        if cutrgba_path and cutrgba_path.exists():
            im = Image.open(cutrgba_path)
            if im.mode == 'RGBA':
                arr = np.array(im)
                alpha_mask = arr[:, :, 3] > 120
                cut_img = arr[:, :, :3]
                has_alpha = True

        if cut_img is None and cut_path and cut_path.exists():
            im = Image.open(cut_path).convert('RGB')
            arr = np.array(im)
            # Apple pixels on uniform 128 gray (128, 128, 128)
            diff = np.abs(arr.astype(int) - 128)
            alpha_mask = (diff[:, :, 0] > 14) | (diff[:, :, 1] > 14) | (diff[:, :, 2] > 14)
            cut_img = arr

        if cut_img is None and orig_path and orig_path.exists():
            im = Image.open(orig_path).convert('RGB')
            arr = np.array(im)
            alpha_mask = np.ones((arr.shape[0], arr.shape[1]), dtype=bool)
            cut_img = arr

        if cut_img is None or alpha_mask is None or not np.any(alpha_mask):
            return {"mean_L": 50.0, "mean_b": 0.0, "sharpness": 100.0, "variety_guess": "green"}

        # Mask apple pixels
        apple_pixels = cut_img[alpha_mask]

        # Convert apple to CIELAB
        bgr = cv2.cvtColor(cut_img, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        L_chan = lab[:, :, 0][alpha_mask]
        a_chan = lab[:, :, 1][alpha_mask]
        b_chan = lab[:, :, 2][alpha_mask]

        mean_L = float(np.mean(L_chan)) * (100.0 / 255.0) # normalize to 0-100
        mean_b = float(np.mean(b_chan)) - 128.0

        # Sharpness: variance of Laplacian on apple region
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap_apple = lap[alpha_mask]
        sharpness = float(np.var(lap_apple)) if len(lap_apple) > 0 else 0.0

        # Variety guess: median hue in HSV and a* in LAB
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h_chan = hsv[:, :, 0][alpha_mask]
        med_h = float(np.median(h_chan))
        med_a = float(np.median(a_chan))

        # OpenCV Hue: 0-179 (Red is around 0-10 and 160-179, Yellow is 18-35, Green is 35-85)
        if med_a > 138 or (med_h < 12 or med_h > 155):
            variety_guess = "red"
        elif 16 <= med_h <= 34:
            variety_guess = "yellow"
        else:
            variety_guess = "green"

        return {
            "mean_L": round(mean_L, 2),
            "mean_b": round(mean_b, 2),
            "sharpness": round(sharpness, 2),
            "variety_guess": variety_guess
        }
    except Exception as e:
        print(f"Error extracting features for {orig_path}: {e}")
        return {"mean_L": 50.0, "mean_b": 0.0, "sharpness": 100.0, "variety_guess": "green"}


def compute_or_load_features():
    global OTSU_THRESHOLD
    features_cache = {}

    if FEATURES_CSV.exists():
        try:
            with open(FEATURES_CSV, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    headers = [h.strip() for h in lines[0].split(',')]
                    for line in lines[1:]:
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 5:
                            stem = parts[0]
                            features_cache[stem] = {
                                "mean_L": float(parts[1]),
                                "mean_b": float(parts[2]),
                                "sharpness": float(parts[3]),
                                "variety_guess": parts[4]
                            }
            print(f"Loaded {len(features_cache)} cached features from {FEATURES_CSV}")
        except Exception as e:
            print(f"Failed to read {FEATURES_CSV}: {e}")

    # Check for missing features
    missing = [stem for stem in ITEMS if stem not in features_cache]
    if missing:
        print(f"Extracting features for {len(missing)} images...")
        t0 = time.time()
        for idx, stem in enumerate(missing, 1):
            orig_p = ITEM_DATA[stem]["orig_path"]
            cut_p = ITEM_DATA[stem]["cut_path"]
            cutrgba_p = ITEM_DATA[stem]["cutrgba_path"]
            feat = extract_apple_features(orig_p, cut_p, cutrgba_p)
            features_cache[stem] = feat
            if idx % 50 == 0 or idx == len(missing):
                print(f"[{idx}/{len(missing)}] features extracted ({time.time()-t0:.1f}s)")

        # Save to features.csv
        try:
            with open(FEATURES_CSV, 'w', encoding='utf-8') as f:
                f.write("stem,mean_L,mean_b,sharpness,variety_guess\n")
                for stem in sorted(features_cache.keys()):
                    fd = features_cache[stem]
                    f.write(f"{stem},{fd['mean_L']},{fd['mean_b']},{fd['sharpness']},{fd['variety_guess']}\n")
            print(f"Saved {len(features_cache)} features to {FEATURES_CSV}")
        except Exception as e:
            print(f"Error saving {FEATURES_CSV}: {e}")

    # Compute Otsu threshold across all mean_L values
    all_L = [fd["mean_L"] for fd in features_cache.values() if fd["mean_L"] > 0]
    if all_L and len(all_L) > 1:
        try:
            OTSU_THRESHOLD = float(threshold_otsu(np.array(all_L)))
        except Exception:
            OTSU_THRESHOLD = float(np.median(all_L))
    else:
        OTSU_THRESHOLD = 50.0

    print(f"Computed Otsu Lightness Threshold: {OTSU_THRESHOLD:.2f}")

    # Apply to ITEM_DATA
    for stem, fd in features_cache.items():
        if stem in ITEM_DATA:
            ITEM_DATA[stem].update(fd)
            # Set auto lighting
            ITEM_DATA[stem]["lighting_auto"] = "dim" if fd["mean_L"] < OTSU_THRESHOLD else "normal"
            if not ITEM_DATA[stem]["lighting"]:
                ITEM_DATA[stem]["lighting"] = ITEM_DATA[stem]["lighting_auto"]
            if not ITEM_DATA[stem]["variety"]:
                ITEM_DATA[stem]["variety"] = fd["variety_guess"]


# ==========================================
# LABELS CSV AUTOSAVE & ATOMIC WRITE
# ==========================================
def save_labels():
    global ACTION_COUNT
    ACTION_COUNT += 1

    temp_file = BASE_DIR / "labels.csv.tmp"
    headers = [
        "filename", "label", "defect_primary", "defect_all",
        "lighting", "lighting_auto", "variety", "seg_problem",
        "mean_L", "mean_b", "sharpness", "labeled_at"
    ]

    try:
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(",".join(headers) + "\n")
            for stem in ITEMS:
                d = ITEM_DATA[stem]
                row = [
                    d.get("orig_filename", f"{stem}.jpg"),
                    d.get("label", ""),
                    d.get("defect_primary", ""),
                    d.get("defect_all", ""),
                    d.get("lighting", ""),
                    d.get("lighting_auto", ""),
                    d.get("variety", ""),
                    str(d.get("seg_problem", 0)),
                    str(d.get("mean_L", "")),
                    str(d.get("mean_b", "")),
                    str(d.get("sharpness", "")),
                    d.get("labeled_at", "")
                ]
                f.write(",".join(row) + "\n")

        # Atomic replace
        os.replace(temp_file, LABELS_CSV)

        # Rolling backup every 20 actions
        if ACTION_COUNT % 20 == 0:
            bak_file = BASE_DIR / "labels.csv.bak"
            shutil.copy2(LABELS_CSV, bak_file)

    except Exception as e:
        print(f"Error saving labels.csv: {e}")


def load_labels():
    if not LABELS_CSV.exists():
        return

    try:
        with open(LABELS_CSV, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if len(lines) <= 1:
                return
            headers = [h.strip() for h in lines[0].split(',')]
            for line in lines[1:]:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= len(headers):
                    fname = parts[0]
                    stem = Path(fname).stem
                    if stem in ITEM_DATA:
                        ITEM_DATA[stem]["label"] = parts[1]
                        ITEM_DATA[stem]["defect_primary"] = parts[2]
                        ITEM_DATA[stem]["defect_all"] = parts[3]
                        if parts[4]:
                            ITEM_DATA[stem]["lighting"] = parts[4]
                        if parts[5]:
                            ITEM_DATA[stem]["lighting_auto"] = parts[5]
                        if parts[6]:
                            ITEM_DATA[stem]["variety"] = parts[6]
                        ITEM_DATA[stem]["seg_problem"] = int(parts[7]) if parts[7].isdigit() else 0
                        ITEM_DATA[stem]["labeled_at"] = parts[11] if len(parts) > 11 else ""
        print(f"Loaded existing labels from {LABELS_CSV}")
    except Exception as e:
        print(f"Error loading labels: {e}")


# ==========================================
# DATASET INITIALIZATION
# ==========================================
def init_dataset():
    global ITEMS, ITEM_DATA
    ITEMS = []
    ITEM_DATA = {}

    # Scan ORIG_DIR
    orig_files = {}
    if ORIG_DIR.exists():
        for p in ORIG_DIR.rglob("*"):
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                orig_files[p.stem] = p

    # Scan CUT_DIR
    cut_files = {}
    if CUT_DIR.exists():
        for p in CUT_DIR.rglob("*"):
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                cut_files[p.stem] = p

    # Scan CUTRGBA_DIR
    cutrgba_files = {}
    if CUTRGBA_DIR.exists():
        for p in CUTRGBA_DIR.rglob("*"):
            if p.suffix.lower() in [".png"]:
                cutrgba_files[p.stem] = p

    # All unique stems
    all_stems = sorted(list(set(list(orig_files.keys()) + list(cut_files.keys()))))
    ITEMS = all_stems

    for stem in ITEMS:
        orig_p = orig_files.get(stem)
        cut_p = cut_files.get(stem)
        cutrgba_p = cutrgba_files.get(stem)
        has_cutout = cut_p is not None and cut_p.exists()

        ITEM_DATA[stem] = {
            "stem": stem,
            "orig_filename": orig_p.name if orig_p else f"{stem}.jpg",
            "orig_path": orig_p,
            "cut_path": cut_p,
            "cutrgba_path": cutrgba_p,
            "has_cutout": has_cutout,
            "label": "",           # good, bad, unsure, deleted
            "defect_primary": "",
            "defect_all": "",
            "lighting": "",        # normal, dim
            "lighting_auto": "",   # normal, dim
            "variety": "",         # red, green, yellow
            "variety_guess": "",
            "seg_problem": 1 if not has_cutout else 0,
            "mean_L": 50.0,
            "mean_b": 0.0,
            "sharpness": 100.0,
            "labeled_at": ""
        }

    print(f"Dataset initialized with {len(ITEMS)} images.")
    compute_or_load_features()
    load_labels()


# ==========================================
# STATS & QUEUE LOGIC
# ==========================================
def get_stats():
    counts = {"good": 0, "bad": 0, "unsure": 0, "unlabeled": 0, "deleted": 0}
    contingency = {
        "good": {"normal": 0, "dim": 0},
        "bad": {"normal": 0, "dim": 0}
    }
    defect_counts = {d: 0 for d in DEFECT_TYPES}

    for stem in ITEMS:
        d = ITEM_DATA[stem]
        lbl = d["label"]
        if lbl in counts:
            counts[lbl] += 1
        else:
            counts["unlabeled"] += 1

        light = d["lighting"] or d["lighting_auto"] or "normal"
        if lbl in ["good", "bad"]:
            if light in contingency[lbl]:
                contingency[lbl][light] += 1

        primary = d["defect_primary"]
        if primary in defect_counts:
            defect_counts[primary] += 1

    return {
        "counts": counts,
        "contingency": contingency,
        "defect_counts": defect_counts,
        "total": len(ITEMS),
        "otsu_threshold": OTSU_THRESHOLD
    }


def get_queue(pass_mode="pass1", filter_mode="all"):
    """
    Returns list of stems belonging to the active queue.
    """
    queue = []
    for stem in ITEMS:
        d = ITEM_DATA[stem]
        if d["label"] == "deleted":
            continue  # deleted items always excluded from active queue

        if pass_mode == "pass1":
            if filter_mode == "unlabeled":
                if not d["label"]:
                    queue.append(stem)
            elif filter_mode == "unsure":
                if d["label"] == "unsure":
                    queue.append(stem)
            elif filter_mode == "seg_problem":
                if d["seg_problem"] == 1:
                    queue.append(stem)
            else: # all
                queue.append(stem)
        elif pass_mode == "pass2":
            # Pass 2: only images with label == 'bad' and no defect_primary
            if d["label"] == "bad" and not d["defect_primary"]:
                queue.append(stem)

    return queue


# ==========================================
# RESIZED IMAGE CACHE & SERVER
# ==========================================
def get_cached_image(file_path, prefix="img", max_dim=1200):
    if not file_path or not file_path.exists():
        return None

    stem = file_path.stem
    ext = file_path.suffix.lower()
    cached_path = CACHE_DIR / f"{prefix}_{stem}_{max_dim}{ext}"

    if cached_path.exists():
        return cached_path

    try:
        im = ImageOps.exif_transpose(Image.open(file_path))
        w, h = im.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            im = im.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        
        # Save to cache
        if ext == '.png':
            im.save(cached_path, format="PNG")
        else:
            im.convert('RGB').save(cached_path, format="JPEG", quality=88, optimize=True)
        return cached_path
    except Exception as e:
        print(f"Error caching {file_path}: {e}")
        return file_path


def serialize_item(d):
    if not d:
        return {}
    res = dict(d)
    for k in ["orig_path", "cut_path", "cutrgba_path"]:
        if k in res and res[k] is not None:
            res[k] = str(res[k])
    return res


# ==========================================
# REST API ENDPOINTS
# ==========================================
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, defect_types=DEFECT_TYPES)


@app.route("/api/init")
def api_init():
    pass_mode = request.args.get("pass", "pass1")
    filter_mode = request.args.get("filter", "all")

    queue = get_queue(pass_mode, filter_mode)
    stats = get_stats()

    # Find first unlabeled or first item
    start_stem = None
    if pass_mode == "pass1":
        for s in queue:
            if not ITEM_DATA[s]["label"]:
                start_stem = s
                break
    elif pass_mode == "pass2":
        if queue:
            start_stem = queue[0]

    if not start_stem and queue:
        start_stem = queue[0]

    return jsonify({
        "items": ITEMS,
        "queue": queue,
        "start_stem": start_stem,
        "stats": stats,
        "defect_types": DEFECT_TYPES
    })


@app.route("/api/item/<stem>")
def api_item(stem):
    if stem not in ITEM_DATA:
        return jsonify({"error": "Item not found"}), 404
    return jsonify(serialize_item(ITEM_DATA[stem]))


@app.route("/api/image/<img_type>/<stem>")
def api_image(img_type, stem):
    if stem not in ITEM_DATA:
        return "Not found", 404

    d = ITEM_DATA[stem]
    if img_type == "orig":
        fp = d["orig_path"]
    elif img_type == "cut":
        fp = d["cut_path"] or d["cutrgba_path"] or d["orig_path"]
    elif img_type == "cutrgba":
        fp = d["cutrgba_path"] or d["cut_path"]
    else:
        return "Invalid image type", 400

    cached_p = get_cached_image(fp, prefix=img_type)
    if not cached_p or not cached_p.exists():
        return "Image file not found", 404

    mimetype = "image/png" if cached_p.suffix.lower() == ".png" else "image/jpeg"
    return send_file(cached_p, mimetype=mimetype)


@app.route("/api/action", methods=["POST"])
def api_action():
    data = request.json
    stem = data.get("stem")
    action = data.get("action")
    val = data.get("value")

    if stem not in ITEM_DATA:
        return jsonify({"error": "Invalid stem"}), 400

    prev_state = dict(ITEM_DATA[stem])

    d = ITEM_DATA[stem]
    now_iso = datetime.now().isoformat()

    if action == "label":
        # Pass 1: good / bad / unsure
        d["label"] = val
        d["labeled_at"] = now_iso
        if val == "good":
            d["defect_primary"] = ""
            d["defect_all"] = ""
    elif action == "seg_problem":
        d["seg_problem"] = 1 if d["seg_problem"] == 0 else 0
        d["labeled_at"] = now_iso
    elif action == "lighting":
        # toggle lighting normal / dim
        current = d["lighting"] or d["lighting_auto"] or "normal"
        d["lighting"] = "dim" if current == "normal" else "normal"
        d["labeled_at"] = now_iso
    elif action == "variety":
        # cycle red -> green -> yellow
        cycle = {"red": "green", "green": "yellow", "yellow": "red"}
        curr = d["variety"] or d["variety_guess"] or "green"
        d["variety"] = cycle.get(curr, "green")
        d["labeled_at"] = now_iso
    elif action == "defect_primary":
        # Pass 2: sets primary defect
        d["defect_primary"] = val
        # Also ensure in defect_all
        all_defs = [x for x in d["defect_all"].split("|") if x] if d["defect_all"] else []
        if val not in all_defs:
            all_defs.insert(0, val)
        d["defect_all"] = "|".join(all_defs)
        d["labeled_at"] = now_iso
    elif action == "defect_add":
        # Shift+number: add secondary defect
        all_defs = [x for x in d["defect_all"].split("|") if x] if d["defect_all"] else []
        if val not in all_defs:
            all_defs.append(val)
        if not d["defect_primary"]:
            d["defect_primary"] = val
        d["defect_all"] = "|".join(all_defs)
        d["labeled_at"] = now_iso
    elif action == "delete":
        d["label"] = "deleted"
        d["labeled_at"] = now_iso

    UNDO_STACK.append({"stem": stem, "prev_state": prev_state})
    save_labels()

    stats = get_stats()
    pass_mode = data.get("pass", "pass1")
    filter_mode = data.get("filter", "all")
    queue = get_queue(pass_mode, filter_mode)

    return jsonify({
        "success": True,
        "item": serialize_item(d),
        "stats": stats,
        "queue": queue
    })


@app.route("/api/undo", methods=["POST"])
def api_undo():
    if not UNDO_STACK:
        return jsonify({"success": False, "message": "Nothing to undo"}), 200

    last = UNDO_STACK.pop()
    stem = last["stem"]
    prev_state = last["prev_state"]

    if stem in ITEM_DATA:
        ITEM_DATA[stem].update(prev_state)
        save_labels()

    data = request.json or {}
    pass_mode = data.get("pass", "pass1")
    filter_mode = data.get("filter", "all")
    queue = get_queue(pass_mode, filter_mode)
    stats = get_stats()

    return jsonify({
        "success": True,
        "stem": stem,
        "item": serialize_item(ITEM_DATA.get(stem, {})),
        "stats": stats,
        "queue": queue
    })


@app.route("/api/settings/threshold", methods=["POST"])
def api_set_threshold():
    global OTSU_THRESHOLD
    data = request.json
    new_thr = float(data.get("threshold", OTSU_THRESHOLD))
    OTSU_THRESHOLD = round(new_thr, 2)

    # Recompute lighting_auto live
    for stem in ITEMS:
        d = ITEM_DATA[stem]
        d["lighting_auto"] = "dim" if d["mean_L"] < OTSU_THRESHOLD else "normal"
        # If not manually overridden, update lighting
        if not d["labeled_at"]:
            d["lighting"] = d["lighting_auto"]

    save_labels()
    stats = get_stats()
    return jsonify({"success": True, "otsu_threshold": OTSU_THRESHOLD, "stats": stats})


@app.route("/api/export", methods=["POST"])
def api_export():
    import export
    summary = export.run_export(BASE_DIR)
    return jsonify({"success": True, "summary": summary})


@app.route("/roboflow_gallery")
def serve_roboflow_gallery():
    gallery_path = BASE_DIR / "roboflow_gallery.html"
    if gallery_path.exists():
        return send_file(gallery_path, mimetype="text/html")
    return "Roboflow gallery not found", 404


# ==========================================
# EMBEDDED SINGLE-PAGE UI (HTML + CSS + JS)
# ==========================================
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Apple Photo Labeling Tool</title>
<style>
  :root {
    --bg: #121417;
    --panel: #1b1f24;
    --border: #2d333b;
    --text: #e6edf3;
    --text-dim: #8b949e;
    --accent: #2ea043;
    --bad: #da3633;
    --unsure: #d29922;
    --highlight: #58a6ff;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  body { background: var(--bg); color: var(--text); height: 100vh; display: flex; flex-direction: column; overflow: hidden; }

  /* Top Bar */
  header {
    background: var(--panel);
    border-bottom: 1px solid var(--border);
    padding: 8px 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
    z-index: 10;
  }
  .header-left { display: flex; align-items: center; gap: 12px; }
  .mode-badge {
    background: #238636; color: #fff; font-size: 13px; font-weight: 700;
    padding: 4px 10px; border-radius: 6px; cursor: pointer; user-select: none;
    display: flex; align-items: center; gap: 6px;
  }
  .mode-badge.pass2 { background: #1f6feb; }
  .counter { font-size: 14px; font-weight: 600; color: var(--text); }
  .filename-title { font-family: monospace; font-size: 13px; color: var(--text-dim); }

  /* 2x2 Contingency Table */
  .table-box {
    display: flex; align-items: center; background: #0d1117;
    border: 1px solid var(--border); border-radius: 6px; padding: 4px 8px; font-size: 11px; gap: 8px;
  }
  .mini-table { border-collapse: collapse; text-align: center; }
  .mini-table th, .mini-table td { padding: 1px 6px; border: 1px solid #30363d; }
  .mini-table th { color: var(--text-dim); font-weight: 600; }
  .mini-table td.good { color: #3fb950; font-weight: bold; }
  .mini-table td.bad { color: #f85149; font-weight: bold; }

  /* Counts Bar */
  .counts-row { display: flex; align-items: center; gap: 6px; font-size: 12px; font-weight: 600; }
  .cnt-badge { padding: 2px 7px; border-radius: 4px; background: #21262d; border: 1px solid var(--border); }
  .cnt-good { color: #3fb950; }
  .cnt-bad { color: #f85149; }
  .cnt-unsure { color: #e3b341; }
  .cnt-unlabeled { color: var(--text-dim); }
  .cnt-deleted { color: #a371f7; }

  /* Main Workspace */
  main {
    flex: 1; display: grid; grid-template-columns: 1fr 1fr;
    gap: 12px; padding: 12px; height: calc(100vh - 120px); overflow: hidden;
  }
  .image-card {
    background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
    display: flex; flex-direction: column; overflow: hidden; position: relative;
  }
  .card-header {
    background: #161b22; padding: 6px 12px; border-bottom: 1px solid var(--border);
    font-size: 12px; font-weight: 600; color: var(--text-dim); display: flex; justify-content: space-between;
  }
  .viewport {
    flex: 1; display: flex; align-items: center; justify-content: center;
    background: #090d12; overflow: hidden; position: relative;
  }
  .viewport img {
    max-width: 100%; max-height: 100%; object-fit: contain; user-select: none;
    transition: transform 0.05s ease-out;
  }
  .viewport.zoomable img { cursor: grab; }
  .viewport.zoomable.dragging img { cursor: grabbing; }

  /* Chips & Overlays */
  .chips-row {
    position: absolute; bottom: 12px; left: 12px; display: flex; gap: 6px; z-index: 5;
  }
  .chip {
    background: rgba(13, 17, 23, 0.85); backdrop-filter: blur(4px);
    border: 1px solid var(--border); border-radius: 12px; padding: 4px 10px;
    font-size: 11px; font-weight: 600; display: flex; align-items: center; gap: 4px;
  }
  .chip.override { border-color: var(--highlight); color: var(--highlight); }
  .chip.seg-problem { background: rgba(218, 54, 51, 0.25); border-color: var(--bad); color: #ff7b72; }
  .chip.variety-red { color: #ff7b72; }
  .chip.variety-green { color: #7ee787; }
  .chip.variety-yellow { color: #f2cc60; }

  /* Bottom Controls & Legend */
  footer {
    background: var(--panel); border-top: 1px solid var(--border);
    padding: 8px 16px; display: flex; align-items: center; justify-content: space-between;
    font-size: 12px; gap: 12px; flex-wrap: wrap;
  }
  .keys-legend { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .key-item { display: flex; align-items: center; gap: 4px; color: var(--text-dim); font-size: 12px; }
  kbd {
    background: #21262d; border: 1px solid #30363d; border-radius: 4px;
    padding: 2px 6px; font-size: 11px; font-weight: bold; color: var(--text);
  }
  .btn-group { display: flex; gap: 6px; }
  .btn {
    background: #21262d; border: 1px solid var(--border); color: var(--text);
    padding: 5px 12px; border-radius: 6px; font-size: 12px; font-weight: 600;
    cursor: pointer; display: flex; align-items: center; gap: 4px; user-select: none;
  }
  .btn:hover { background: #30363d; }
  .btn.btn-good { background: #238636; border-color: #2ea043; color: #fff; }
  .btn.btn-bad { background: #da3633; border-color: #f85149; color: #fff; }
  .btn.btn-unsure { background: #9e6a03; border-color: #d29922; color: #fff; }
  .btn.btn-del { background: #490202; border-color: #82071e; color: #ff7b72; }

  /* Settings Modal / Drawer */
  .modal-overlay {
    position: fixed; inset: 0; background: rgba(0,0,0,0.6); display: none;
    align-items: center; justify-content: center; z-index: 100;
  }
  .modal {
    background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
    padding: 20px; width: 440px; max-width: 90vw; display: flex; flex-direction: column; gap: 16px;
  }
  .modal-header { display: flex; justify-content: space-between; align-items: center; font-size: 16px; font-weight: 600; }
  .modal-close { cursor: pointer; color: var(--text-dim); font-size: 18px; }

  /* Pass 2 Defects Legend Bar */
  .pass2-bar {
    background: #101c2e; border: 1px solid #1f6feb; border-radius: 6px;
    padding: 6px 12px; display: flex; align-items: center; gap: 12px; font-size: 12px;
  }
</style>
</head>
<body>

<header>
  <div class="header-left">
    <div id="modeBtn" class="mode-badge" onclick="togglePass()">
      <span id="modeText">PASS 1: GOOD / BAD</span> <kbd>M</kbd>
    </div>
    <div class="counter" id="counterText">0 / 0</div>
    <div class="filename-title" id="filenameText">loading...</div>
    <div id="currentLabelBadge" class="cnt-badge" style="font-size: 13px; font-weight: 700; padding: 3px 10px; border-radius: 6px;">UNLABELED</div>
  </div>

  <div class="table-box">
    <table class="mini-table">
      <thead>
        <tr><th></th><th>Normal</th><th>Dim</th><th>Total</th></tr>
      </thead>
      <tbody>
        <tr><th>Good</th><td id="t-gn" class="good">0</td><td id="t-gd" class="good">0</td><td id="t-gt">0</td></tr>
        <tr><th>Bad</th><td id="t-bn" class="bad">0</td><td id="t-bd" class="bad">0</td><td id="t-bt">0</td></tr>
      </tbody>
    </table>
  </div>

  <div class="counts-row">
    <span class="cnt-badge cnt-good">Good: <span id="cntGood">0</span></span>
    <span class="cnt-badge cnt-bad">Bad: <span id="cntBad">0</span></span>
    <span class="cnt-badge cnt-unsure">Unsure: <span id="cntUnsure">0</span></span>
    <span class="cnt-badge cnt-unlabeled">Unlabeled: <span id="cntUnlabeled">0</span></span>
    <span class="cnt-badge cnt-deleted">Deleted: <span id="cntDeleted">0</span></span>
    <a href="/roboflow_gallery" target="_blank" class="btn" style="text-decoration: none;" title="Inspect Original Roboflow Dataset">🔍 Roboflow Data</a>
    <button class="btn" onclick="openSettings()" title="Settings">⚙</button>
    <button class="btn" onclick="triggerExport()" title="Export Dataset">📦 Export</button>
  </div>
</header>

<!-- Pass 2 Defect Legend Row (Visible only in Pass 2) -->
<div id="pass2Row" class="pass2-bar" style="display: none; margin: 6px 16px -4px 16px;">
  <span style="font-weight: 600; color: #58a6ff;">Defect Types (Bad Only):</span>
  {% for dt in defect_types %}
  <span class="key-item"><kbd>{{ loop.index }}</kbd> {{ dt }} (<span id="defCnt-{{ dt }}">0</span>)</span>
  {% endfor %}
  <span style="color: var(--text-dim); margin-left: auto;"><kbd>Shift + 1..5</kbd> Add Secondary | <kbd>G</kbd> Actually Good | <kbd>U</kbd> Unsure | <kbd>Z</kbd> 2x Zoom</span>
</div>

<main>
  <!-- Left Card: Original Photo -->
  <div class="image-card">
    <div class="card-header">
      <span>ORIGINAL PHOTO</span>
      <span id="origDims">--</span>
    </div>
    <div class="viewport">
      <img id="origImg" src="" alt="Original Photo">
    </div>
  </div>

  <!-- Right Card: Cutout on Gray -->
  <div class="image-card">
    <div class="card-header">
      <span>CUTOUT ON GRAY (384x384)</span>
      <span id="zoomLevel">1.0x (Z toggles 2x)</span>
    </div>
    <div class="viewport zoomable" id="cutoutViewport">
      <img id="cutImg" src="" alt="Cutout Photo">
      <div class="chips-row">
        <div class="chip" id="chipLighting">Lighting: Normal</div>
        <div class="chip" id="chipVariety">Variety: Green</div>
        <div class="chip" id="chipSeg">Seg: OK</div>
      </div>
    </div>
  </div>
</main>

<footer>
  <div class="keys-legend" id="keysLegend">
    <!-- Populated dynamically based on Pass -->
  </div>

  <div class="btn-group">
    <button class="btn" onclick="nav(-1)" title="Previous Image"><kbd>←</kbd> Prev</button>
    <button class="btn" onclick="nav(1)" title="Next Image"><kbd>→</kbd> Next</button>
    <button class="btn" onclick="skipImage()" title="Skip without labeling"><kbd>Space</kbd> Skip</button>
    <button class="btn btn-del" onclick="deleteImage()" title="Delete Image from Dataset"><kbd>Del</kbd> Delete</button>
    <button class="btn" onclick="undo()" title="Undo Last Action"><kbd>Ctrl+Z</kbd> Undo</button>
  </div>
</footer>

<!-- Settings Modal -->
<div class="modal-overlay" id="settingsModal" onclick="closeSettings(event)">
  <div class="modal" onclick="event.stopPropagation()">
    <div class="modal-header">
      <span>Lighting Threshold Settings</span>
      <span class="modal-close" onclick="closeSettings()">&times;</span>
    </div>
    <div style="font-size: 13px; color: var(--text-dim); line-height: 1.4;">
      Images with mean apple lightness $L^*$ below this threshold are automatically categorized as <b>Dim</b>. Manual overrides always take precedence.
    </div>
    <div style="display: flex; align-items: center; gap: 12px;">
      <input type="range" id="otsuSlider" min="20" max="80" step="0.5" style="flex: 1;" oninput="updateSliderVal(this.value)">
      <span id="otsuVal" style="font-weight: bold; width: 45px;">50.0</span>
    </div>
    <div style="display: flex; justify-content: flex-end; gap: 8px;">
      <button class="btn" onclick="closeSettings()">Cancel</button>
      <button class="btn btn-good" onclick="applyThreshold()">Apply & Re-evaluate</button>
    </div>
  </div>
</div>

<script>
// Application State
let appState = {
  pass: "pass1",       // pass1 or pass2
  filter: "all",       // all, unlabeled, unsure, seg_problem
  queue: [],
  currentIndex: 0,
  currentItem: null,
  defectTypes: {{ defect_types | tojson }},
  zoom: 1.0,
  panX: 0,
  panY: 0,
  isDragging: false,
  dragStartX: 0,
  dragStartY: 0
};

// Prefetch Cache
const prefetchCache = {};

// On Load
window.addEventListener("DOMContentLoaded", () => {
  initApp();
  setupZoomAndPan();
  setupKeyboard();
});

async function initApp() {
  const res = await fetch(`/api/init?pass=${appState.pass}&filter=${appState.filter}`);
  const data = await res.json();
  appState.queue = data.queue;
  appState.defectTypes = data.defect_types;
  updateStats(data.stats);

  if (data.start_stem) {
    const idx = appState.queue.indexOf(data.start_stem);
    appState.currentIndex = idx >= 0 ? idx : 0;
  } else {
    appState.currentIndex = 0;
  }

  loadCurrentItem();
}

function updateStats(stats) {
  if (!stats) return;
  document.getElementById("cntGood").innerText = stats.counts.good;
  document.getElementById("cntBad").innerText = stats.counts.bad;
  document.getElementById("cntUnsure").innerText = stats.counts.unsure;
  document.getElementById("cntUnlabeled").innerText = stats.counts.unlabeled;
  document.getElementById("cntDeleted").innerText = stats.counts.deleted;

  const c = stats.contingency;
  document.getElementById("t-gn").innerText = c.good.normal;
  document.getElementById("t-gd").innerText = c.good.dim;
  document.getElementById("t-gt").innerText = c.good.normal + c.good.dim;
  document.getElementById("t-bn").innerText = c.bad.normal;
  document.getElementById("t-bd").innerText = c.bad.dim;
  document.getElementById("t-bt").innerText = c.bad.normal + c.bad.dim;

  // Defect counts
  for (const [dt, cnt] of Object.entries(stats.defect_counts)) {
    const el = document.getElementById(`defCnt-${dt}`);
    if (el) el.innerText = cnt;
  }

  // Slider val
  document.getElementById("otsuSlider").value = stats.otsu_threshold;
  document.getElementById("otsuVal").innerText = stats.otsu_threshold.toFixed(1);
}

async function loadCurrentItem() {
  if (appState.queue.length === 0) {
    document.getElementById("counterText").innerText = "0 / 0";
    document.getElementById("filenameText").innerText = "(Queue is empty)";
    document.getElementById("origImg").src = "";
    document.getElementById("cutImg").src = "";
    return;
  }

  if (appState.currentIndex < 0) appState.currentIndex = 0;
  if (appState.currentIndex >= appState.queue.length) appState.currentIndex = appState.queue.length - 1;

  const stem = appState.queue[appState.currentIndex];
  document.getElementById("counterText").innerText = `${appState.currentIndex + 1} / ${appState.queue.length}`;
  document.getElementById("filenameText").innerText = `${stem}.jpg`;

  const res = await fetch(`/api/item/${stem}`);
  const item = await res.json();
  appState.currentItem = item;

  // Reset zoom
  resetZoom();

  // Load images
  const origUrl = `/api/image/orig/${stem}`;
  const cutUrl = `/api/image/cut/${stem}`;
  document.getElementById("origImg").src = origUrl;
  document.getElementById("cutImg").src = cutUrl;

  // Update Chips
  const light = item.lighting || item.lighting_auto || "normal";
  const lightIsOver = item.lighting && item.lighting !== item.lighting_auto;
  const chipL = document.getElementById("chipLighting");
  chipL.innerText = `Lighting: ${light.toUpperCase()} ${lightIsOver ? '(D: manual)' : '(auto)'}`;
  chipL.className = `chip ${lightIsOver ? 'override' : ''}`;

  const varStr = item.variety || item.variety_guess || "green";
  const varIsOver = item.variety && item.variety !== item.variety_guess;
  const chipV = document.getElementById("chipVariety");
  chipV.innerText = `Variety: ${varStr.toUpperCase()} ${varIsOver ? '(V: manual)' : '(auto)'}`;
  chipV.className = `chip variety-${varStr} ${varIsOver ? 'override' : ''}`;

  const chipS = document.getElementById("chipSeg");
  chipS.innerText = item.seg_problem ? "Seg: PROBLEM (Q)" : "Seg: OK";
  chipS.className = `chip ${item.seg_problem ? 'seg-problem' : ''}`;

  // Update current label badge
  const lblBadge = document.getElementById("currentLabelBadge");
  if (item.label === "good") {
    lblBadge.innerText = "✓ GOOD";
    lblBadge.style.background = "#238636";
    lblBadge.style.color = "#ffffff";
    lblBadge.style.borderColor = "#2ea043";
  } else if (item.label === "bad") {
    const def = item.defect_primary ? ` (${item.defect_primary})` : "";
    lblBadge.innerText = `✗ BAD${def}`;
    lblBadge.style.background = "#da3633";
    lblBadge.style.color = "#ffffff";
    lblBadge.style.borderColor = "#f85149";
  } else if (item.label === "unsure") {
    lblBadge.innerText = "? UNSURE";
    lblBadge.style.background = "#9e6a03";
    lblBadge.style.color = "#ffffff";
    lblBadge.style.borderColor = "#d29922";
  } else if (item.label === "deleted") {
    lblBadge.innerText = "🗑 DELETED";
    lblBadge.style.background = "#490202";
    lblBadge.style.color = "#ff7b72";
    lblBadge.style.borderColor = "#82071e";
  } else {
    lblBadge.innerText = "UNLABELED";
    lblBadge.style.background = "#21262d";
    lblBadge.style.color = "var(--text-dim)";
    lblBadge.style.borderColor = "var(--border)";
  }

  // Update legend
  updateLegend();

  // Prefetch next 3 items
  prefetchNext();
}

function prefetchNext() {
  for (let i = 1; i <= 3; i++) {
    const nextIdx = appState.currentIndex + i;
    if (nextIdx < appState.queue.length) {
      const stem = appState.queue[nextIdx];
      if (!prefetchCache[stem]) {
        const img1 = new Image();
        img1.src = `/api/image/orig/${stem}`;
        const img2 = new Image();
        img2.src = `/api/image/cut/${stem}`;
        prefetchCache[stem] = true;
      }
    }
  }
}

function updateLegend() {
  const container = document.getElementById("keysLegend");
  if (appState.pass === "pass1") {
    container.innerHTML = `
      <span class="key-item"><kbd>G</kbd> Good</span>
      <span class="key-item"><kbd>B</kbd> Bad</span>
      <span class="key-item"><kbd>U</kbd> Unsure</span>
      <span class="key-item"><kbd>Q</kbd> Seg Problem</span>
      <span class="key-item"><kbd>D</kbd> Lighting (${appState.currentItem ? (appState.currentItem.lighting || appState.currentItem.lighting_auto) : 'normal'})</span>
      <span class="key-item"><kbd>V</kbd> Variety (${appState.currentItem ? (appState.currentItem.variety || appState.currentItem.variety_guess) : 'green'})</span>
      <span class="key-item"><kbd>F</kbd> Filter: ${appState.filter.toUpperCase()}</span>
      <span class="key-item"><kbd>Z</kbd> 2x Zoom</span>
      <span class="key-item"><kbd>Del</kbd> Delete</span>
      <span class="key-item"><kbd>M</kbd> Switch to Pass 2</span>
    `;
  } else {
    container.innerHTML = `
      <span class="key-item"><kbd>1..5</kbd> Set Primary & Advance</span>
      <span class="key-item"><kbd>Shift + 1..5</kbd> Add Secondary</span>
      <span class="key-item"><kbd>G</kbd> Actually Good</span>
      <span class="key-item"><kbd>U</kbd> Unsure</span>
      <span class="key-item"><kbd>Z</kbd> 2x Zoom</span>
      <span class="key-item"><kbd>Del</kbd> Delete</span>
      <span class="key-item"><kbd>M</kbd> Switch to Pass 1</span>
    `;
  }
}

function togglePass() {
  appState.pass = appState.pass === "pass1" ? "pass2" : "pass1";
  document.getElementById("modeText").innerText = appState.pass === "pass1" ? "PASS 1: GOOD / BAD" : "PASS 2: DEFECT TYPES";
  const btn = document.getElementById("modeBtn");
  if (appState.pass === "pass2") {
    btn.classList.add("pass2");
    document.getElementById("pass2Row").style.display = "flex";
  } else {
    btn.classList.remove("pass2");
    document.getElementById("pass2Row").style.display = "none";
  }
  refreshQueue();
}

function cycleFilter() {
  const filters = ["all", "unlabeled", "unsure", "seg_problem"];
  const currIdx = filters.indexOf(appState.filter);
  appState.filter = filters[(currIdx + 1) % filters.length];
  refreshQueue();
}

async function refreshQueue() {
  const res = await fetch(`/api/init?pass=${appState.pass}&filter=${appState.filter}`);
  const data = await res.json();
  appState.queue = data.queue;
  appState.currentIndex = 0;
  updateStats(data.stats);
  loadCurrentItem();
}

async function sendAction(action, value, autoAdvance = true) {
  if (!appState.currentItem) return;
  const stem = appState.currentItem.stem;

  const res = await fetch("/api/action", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      stem: stem,
      action: action,
      value: value,
      pass: appState.pass,
      filter: appState.filter
    })
  });
  const data = await res.json();
  if (data.success) {
    appState.queue = data.queue;
    updateStats(data.stats);
    if (autoAdvance) {
      nav(1);
    } else {
      appState.currentItem = data.item;
      loadCurrentItem();
    }
  }
}

function nav(delta) {
  if (appState.queue.length === 0) return;
  appState.currentIndex += delta;
  if (appState.currentIndex < 0) appState.currentIndex = 0;
  if (appState.currentIndex >= appState.queue.length) appState.currentIndex = appState.queue.length - 1;
  loadCurrentItem();
}

function skipImage() {
  nav(1);
}

async function deleteImage() {
  if (!confirm("Are you sure you want to remove this image from the active dataset? (Can be undone with Ctrl+Z)")) return;
  await sendAction("delete", null, true);
}

async function undo() {
  const res = await fetch("/api/undo", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pass: appState.pass, filter: appState.filter })
  });
  const data = await res.json();
  if (data.success) {
    appState.queue = data.queue;
    updateStats(data.stats);
    const idx = appState.queue.indexOf(data.stem);
    if (idx >= 0) appState.currentIndex = idx;
    loadCurrentItem();
  }
}

// Zoom & Pan
function setupZoomAndPan() {
  const vp = document.getElementById("cutoutViewport");
  const img = document.getElementById("cutImg");

  vp.addEventListener("wheel", (e) => {
    e.preventDefault();
    const zoomDelta = e.deltaY < 0 ? 0.25 : -0.25;
    appState.zoom = Math.max(1.0, Math.min(5.0, appState.zoom + zoomDelta));
    if (appState.zoom === 1.0) { appState.panX = 0; appState.panY = 0; }
    applyTransform();
  });

  vp.addEventListener("mousedown", (e) => {
    if (appState.zoom > 1.0) {
      appState.isDragging = true;
      appState.dragStartX = e.clientX - appState.panX;
      appState.dragStartY = e.clientY - appState.panY;
      vp.classList.add("dragging");
    }
  });

  window.addEventListener("mousemove", (e) => {
    if (appState.isDragging) {
      appState.panX = e.clientX - appState.dragStartX;
      appState.panY = e.clientY - appState.dragStartY;
      applyTransform();
    }
  });

  window.addEventListener("mouseup", () => {
    appState.isDragging = false;
    vp.classList.remove("dragging");
  });
}

function resetZoom() {
  appState.zoom = 1.0;
  appState.panX = 0;
  appState.panY = 0;
  applyTransform();
}

function toggle2xZoom() {
  appState.zoom = appState.zoom === 1.0 ? 2.0 : 1.0;
  if (appState.zoom === 1.0) { appState.panX = 0; appState.panY = 0; }
  applyTransform();
}

function applyTransform() {
  const img = document.getElementById("cutImg");
  img.style.transform = `translate(${appState.panX}px, ${appState.panY}px) scale(${appState.zoom})`;
  document.getElementById("zoomLevel").innerText = `${appState.zoom.toFixed(1)}x (Z toggles 2x)`;
}

// Keyboard Handling
function setupKeyboard() {
  window.addEventListener("keydown", (e) => {
    // Ignore input fields
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;

    const key = e.key.toUpperCase();

    // Global keys
    if (key === "M") { e.preventDefault(); togglePass(); return; }
    if (key === "ARROWLEFT") { e.preventDefault(); nav(-1); return; }
    if (key === "ARROWRIGHT") { e.preventDefault(); nav(1); return; }
    if (key === " " || key === "S") { e.preventDefault(); skipImage(); return; }
    if (key === "BACKSPACE" || (e.ctrlKey && key === "Z")) { e.preventDefault(); undo(); return; }
    if (key === "DELETE" || key === "X") { e.preventDefault(); deleteImage(); return; }
    if (key === "Z") { e.preventDefault(); toggle2xZoom(); return; }
    if (key === "Q") { e.preventDefault(); sendAction("seg_problem", null, false); return; }
    if (key === "D") { e.preventDefault(); sendAction("lighting", null, false); return; }
    if (key === "V") { e.preventDefault(); sendAction("variety", null, false); return; }
    if (key === "F") { e.preventDefault(); cycleFilter(); return; }

    if (appState.pass === "pass1") {
      if (key === "G") { e.preventDefault(); sendAction("label", "good", true); return; }
      if (key === "B") { e.preventDefault(); sendAction("label", "bad", true); return; }
      if (key === "U") { e.preventDefault(); sendAction("label", "unsure", true); return; }
    } else if (appState.pass === "pass2") {
      // Number keys 1..5
      const num = parseInt(e.key);
      if (!isNaN(num) && num >= 1 && num <= appState.defectTypes.length) {
        e.preventDefault();
        const defectName = appState.defectTypes[num - 1];
        if (e.shiftKey) {
          // Shift+number: Add secondary defect
          sendAction("defect_add", defectName, false);
        } else {
          // Normal number: Set primary defect and advance
          sendAction("defect_primary", defectName, true);
        }
        return;
      }
      if (key === "ENTER") { e.preventDefault(); nav(1); return; }
      if (key === "G") { e.preventDefault(); sendAction("label", "good", true); return; }
      if (key === "U") { e.preventDefault(); sendAction("label", "unsure", true); return; }
    }
  });
}

// Settings Modal
function openSettings() {
  document.getElementById("settingsModal").style.display = "flex";
}

function closeSettings() {
  document.getElementById("settingsModal").style.display = "none";
}

function updateSliderVal(v) {
  document.getElementById("otsuVal").innerText = parseFloat(v).toFixed(1);
}

async function applyThreshold() {
  const val = document.getElementById("otsuSlider").value;
  const res = await fetch("/api/settings/threshold", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ threshold: parseFloat(val) })
  });
  const data = await res.json();
  if (data.success) {
    updateStats(data.stats);
    closeSettings();
    loadCurrentItem();
  }
}

async function triggerExport() {
  const btn = event.target;
  btn.innerText = "Exporting...";
  btn.disabled = true;
  try {
    const res = await fetch("/api/export", { method: "POST" });
    const data = await res.json();
    alert("Export Completed! Summary:\n" + data.summary);
  } catch (err) {
    alert("Export failed: " + err);
  } finally {
    btn.innerText = "📦 Export";
    btn.disabled = false;
  }
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    init_dataset()
    print(f"\n==========================================")
    print(f" Apple Labeling Tool Ready!")
    print(f" Open your browser at: http://localhost:{PORT}")
    print(f"==========================================\n")
    app.run(host=HOST, port=PORT, debug=False)
