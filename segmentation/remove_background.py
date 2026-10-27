#!/usr/bin/env python3
"""
Apple + Hand Background Removal Pipeline
Features:
  1. Segments and keeps both the Hand and the Apple intact.
  2. Eliminates all external background clutter (truck beds, floor, sandals/feet, background fruit crates).
  3. 100% preserves apple defects (bruises, holes, rot) and stem.
  4. Keeps hand, palm, and fingers completely natural with zero awkward clipping or cuts.
  5. Strict rule: Raw RGB pixels are never color-corrected or filtered.
  6. Outputs:
       - output/cutouts_rgba/<name>.png : hand + apple on transparent background, ~5% padding
       - output/processed_gray/<name>.jpg : centered on (128,128,128) gray, 384x384 un-stretched
       - output/contact_sheets/ : 5x5 grid sheets
"""

import os
import sys
import argparse
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

try:
    import rembg
    from ultralytics import YOLO
except ImportError as e:
    print(f"Error loading dependencies: {e}")
    sys.exit(1)

# Default Directories
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "apples"
OUTPUT_DIR = BASE_DIR / "output"
CUTOUTS_DIR = OUTPUT_DIR / "cutouts_rgba"
GRAY_DIR = OUTPUT_DIR / "processed_gray"
CONTACT_DIR = OUTPUT_DIR / "contact_sheets"


def setup_dirs():
    for d in [CUTOUTS_DIR, GRAY_DIR, CONTACT_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def segment_hand_and_apple(img_bgr, yolo_seg, rembg_session):
    """
    Extracts the combined silhouette of the hand and the apple,
    removing all background (truck bed, ground, feet, background crates).
    """
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 1. Person segmentation for hand (person class 0)
    res = yolo_seg(img_bgr, classes=[0], imgsz=1280, conf=0.10, verbose=False)
    hand_mask = np.zeros((h, w), dtype=np.uint8)
    if res[0].masks is not None:
        for i, box in enumerate(res[0].boxes):
            bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
            # Filter out feet/shoes near bottom edge
            if by1 > 0.70 * h:
                continue
            m = res[0].masks.data[i].cpu().numpy()
            m_up = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            hand_mask = np.maximum(hand_mask, (m_up > 0.35).astype(np.uint8) * 255)

    # 2. rembg for the apple
    rembg_mask = rembg.remove(rgb, session=rembg_session, only_mask=True)

    # 3. Skin tone boundary support (ensures finger edges touching the apple are solid)
    R = rgb[:, :, 0].astype(int)
    G = rgb[:, :, 1].astype(int)
    B = rgb[:, :, 2].astype(int)
    is_skin = (R > G + 8) & (R > B + 15)

    combined = (hand_mask > 128) | (rembg_mask > 120)
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    hand_dilated = cv2.dilate(hand_mask, k5)
    combined = combined | ((hand_dilated > 0) & is_skin)
    combined = combined.astype(np.uint8) * 255

    # 4. Fill internal holes (apple defects, core, palm hollows stay 100% solid)
    flood = combined.copy()
    fmask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, fmask, (0, 0), 255)
    solid = (combined | (flood == 0)).astype(np.uint8) * 255

    # 5. Keep largest connected component (drops any detached background specks or crates)
    num_l, labels, stats, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)
    if num_l > 1:
        max_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        solid = (labels == max_idx).astype(np.uint8) * 255

    # 6. Smooth feather 1-2px
    feathered = cv2.GaussianBlur(solid.astype(np.float32), (5, 5), 1.5).astype(np.uint8)
    return feathered


def process_image(img_path, yolo_seg, rembg_session):
    """
    End-to-end processing of a single image.
    """
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        return False, "Failed to read image", None

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    alpha = segment_hand_and_apple(bgr, yolo_seg, rembg_session)

    y_idx, x_idx = np.where(alpha > 30)
    if len(y_idx) == 0:
        return False, "Empty mask after segmentation", None

    bx_min, bx_max = x_idx.min(), x_idx.max()
    by_min, by_max = y_idx.min(), y_idx.max()
    bw = bx_max - bx_min + 1
    bh = by_max - by_min + 1

    # ~5% margin around bounding box
    pad_x = int(bw * 0.05)
    pad_y = int(bh * 0.05)
    cx1 = max(0, bx_min - pad_x)
    cy1 = max(0, by_min - pad_y)
    cx2 = min(w, bx_max + pad_x + 1)
    cy2 = min(h, by_max + pad_y + 1)

    # 1. Output RGBA cutout (Raw RGB preserved bit-for-bit)
    tight_rgb = rgb[cy1:cy2, cx1:cx2]
    tight_alpha = alpha[cy1:cy2, cx1:cx2]
    cutout_rgba = np.dstack([tight_rgb, tight_alpha])

    rgba_path = CUTOUTS_DIR / f"{img_path.stem}.png"
    Image.fromarray(cutout_rgba).save(str(rgba_path), format="PNG")

    # 2. Output 384x384 uniform gray (128, 128, 128) square without stretching
    target_dim = 384
    scale = (target_dim * 0.92) / max(bw, bh)
    new_w = max(1, int(bw * scale))
    new_h = max(1, int(bh * scale))

    crop_rgb = rgb[by_min:by_max+1, bx_min:bx_max+1]
    crop_alpha = alpha[by_min:by_max+1, bx_min:bx_max+1]
    res_rgb = cv2.resize(crop_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    res_alpha = (cv2.resize(crop_alpha, (new_w, new_h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0)[:, :, None]

    canvas = np.full((target_dim, target_dim, 3), 128, dtype=np.uint8)
    ox = (target_dim - new_w) // 2
    oy = (target_dim - new_h) // 2
    region = canvas[oy:oy+new_h, ox:ox+new_w].astype(np.float32)
    blended = (res_rgb.astype(np.float32) * res_alpha + region * (1.0 - res_alpha)).astype(np.uint8)
    canvas[oy:oy+new_h, ox:ox+new_w] = blended

    gray_path = GRAY_DIR / f"{img_path.stem}.jpg"
    Image.fromarray(canvas).save(str(gray_path), format="JPEG", quality=95)

    return True, "OK", (rgb, cutout_rgba, canvas)


def create_contact_sheet(items, output_path, sheet_title="Contact Sheet"):
    if not items:
        return
    thumb_w, thumb_h = 200, 260
    cols = 5
    rows = (len(items) + cols - 1) // cols
    card_w = thumb_w * 2 + 10
    card_h = thumb_h + 30
    margin = 20

    sheet_w = margin * 2 + cols * card_w
    sheet_h = margin * 2 + rows * card_h + 40

    sheet = np.full((sheet_h, sheet_w, 3), 245, dtype=np.uint8)
    cv2.putText(sheet, sheet_title, (margin, margin + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2)

    for idx, (name, orig_rgb, cutout_rgba) in enumerate(items):
        r = idx // cols
        c = idx % cols
        x0 = margin + c * card_w
        y0 = margin + 40 + r * card_h

        # Original
        o_h, o_w = orig_rgb.shape[:2]
        s_o = min(thumb_w / max(1, o_w), thumb_h / max(1, o_h))
        scaled_o = cv2.resize(orig_rgb, (max(1, int(o_w * s_o)), max(1, int(o_h * s_o))))
        sheet[y0:y0+scaled_o.shape[0], x0:x0+scaled_o.shape[1]] = scaled_o

        # Cutout on neutral gray
        c_h, c_w = cutout_rgba.shape[:2]
        s_c = min(thumb_w / max(1, c_w), thumb_h / max(1, c_h))
        scaled_c = cv2.resize(cutout_rgba, (max(1, int(c_w * s_c)), max(1, int(c_h * s_c))))
        rgb_c = scaled_c[:, :, :3]
        alpha_c = scaled_c[:, :, 3:4].astype(np.float32) / 255.0

        bg_gray = np.full(scaled_c.shape[:2] + (3,), 200, dtype=np.uint8)
        comp = (rgb_c.astype(np.float32) * alpha_c + bg_gray.astype(np.float32) * (1.0 - alpha_c)).astype(np.uint8)

        cx0 = x0 + thumb_w + 5
        sheet[y0:y0+comp.shape[0], cx0:cx0+comp.shape[1]] = comp
        cv2.putText(sheet, f"{name[:16]}", (x0, y0 + thumb_h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (50, 150, 50), 1)

    cv2.imwrite(str(output_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print(f"Contact sheet saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Hand + Apple Background Removal Pipeline")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images")
    args = parser.parse_args()

    setup_dirs()
    all_images = sorted(list(INPUT_DIR.glob("*.jpg")))

    if args.limit:
        all_images = all_images[:args.limit]
        print(f"Running on {len(all_images)} test images...")

    print("Initializing YOLO segmentation and rembg session...")
    yolo_seg = YOLO("yolov8n-seg.pt")
    rembg_session = rembg.new_session("isnet-general-use")

    processed = 0
    contact_items = []

    for idx, img_p in enumerate(all_images, 1):
        t1 = time.time()
        success, reason, visuals = process_image(img_p, yolo_seg, rembg_session)
        dt = time.time() - t1
        if success:
            processed += 1
            orig_rgb, cutout_rgba, gray_canvas = visuals
            contact_items.append((img_p.name, orig_rgb, cutout_rgba))
            print(f"[{idx}/{len(all_images)}] {img_p.name} | OK ({dt:.1f}s)")
        else:
            print(f"[{idx}/{len(all_images)}] {img_p.name} | FAILED ({reason})")

    # Generate Contact Sheets (in 25-image chunks if more than 25)
    chunk_size = 25
    for sheet_idx, start in enumerate(range(0, len(contact_items), chunk_size), 1):
        chunk = contact_items[start:start+chunk_size]
        sheet_path = CONTACT_DIR / f"contact_sheet_{sheet_idx:02d}.jpg"
        create_contact_sheet(chunk, sheet_path, f"Hand + Apple Cutouts - Sheet {sheet_idx} ({len(chunk)} Images)")

    # Copy primary contact sheet to root folder
    root_sheet = BASE_DIR / "contact_sheet.jpg"
    primary_sheet = CONTACT_DIR / "contact_sheet_01.jpg"
    if primary_sheet.exists():
        import shutil
        shutil.copy2(primary_sheet, root_sheet)
        print(f"Copied contact sheet to: {root_sheet}")

    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
