# Task: Remove backgrounds and fingers from apple photos on Kaggle (for a good/bad apple classifier)

## Goal
I have about 400 phone photos of apples, each held in one hand, with a blurry crate of other apples behind. For each photo, keep ONLY the held apple. Remove the background and the fingers/hand, with no fingertip bumps on the outline. The output feeds an image classifier (good vs bad apple), so **defects must never be cut off or invented**: blemishes, holes and bruises at the apple's edge must stay exactly as in the photo.

## Hard rules
1. Do NOT use colour thresholds to detect skin. Apples may be red, green or yellow. Use the detector + SAM method below.
2. Do NOT erode, blur or morphologically "smooth" the mask. This was tried and it distorted the real outline. Only fill interior holes, keep the largest component, and apply a 1px feather.
3. Do NOT touch the pixels inside the apple. Never inpaint or recolour.
4. The same code path (`process()`) must be reusable later for new photos at prediction time.
5. Do not silently drop images. Every image ends up in `report.csv` with flags, and suspicious ones go to a review folder.

## What the human must do first (an agent usually cannot)
1. Log in to kaggle.com and verify the phone number (needed for GPU and Internet).
2. Upload the photos as a dataset: Create, then New Dataset, then upload a folder with one subfolder per class, for example `good/` and `bad/` (if the photos are not sorted yet, one flat folder works too).
3. Create a new Notebook and add that dataset via Add Input.
4. In the notebook settings: **Accelerator = GPU (T4 or P100)** and **Internet = On** (needed to download the models from Hugging Face).

## Method
1. **Grounding DINO** (text-prompted detector) finds apples. Pick the **largest** box, because it is the held apple. The apples in the crate behind are smaller and blurry.
2. **SAM** (Segment Anything) gets that box and returns a pixel-accurate apple mask.
3. Grounding DINO also finds "a hand. a finger." and SAM masks them. Subtract the hand from the apple mask **only in the outer rim** of the apple. A protected core (the apple eroded by about 12% of its size) is always kept, so SAM can never eat the apple's middle.
4. If the hand subtraction removes more than 25% of the apple, it is untrustworthy: use the plain apple mask and flag the image for review.
5. Crop to the apple, paste it onto neutral grey (128,128,128), pad to a square and resize. Also save an RGBA PNG with transparency.

## Notebook code (paste into cells in order)

### Cell 1: install and imports
```python
!pip install -q -U transformers accelerate
import os, glob, random, numpy as np, cv2, torch, pandas as pd
from pathlib import Path
from PIL import Image, ImageOps
from scipy import ndimage as ndi
from tqdm.auto import tqdm
from transformers import (AutoProcessor, AutoModelForZeroShotObjectDetection,
                          SamModel, SamProcessor)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(dev)
```

### Cell 2: config (edit IN_DIR)
```python
IN_DIR  = Path("/kaggle/input/YOUR-DATASET-NAME")   # contains good/ bad/ (or flat images)
OUT_DIR = Path("/kaggle/working/apples_clean")
SIZE    = 384          # final square size (use what your classifier needs)
BG      = (128,128,128)
PAD     = 0.04         # margin around the apple, as a fraction of its size
DET_THR = 0.30
(OUT_DIR/"jpg").mkdir(parents=True, exist_ok=True)
(OUT_DIR/"png_rgba").mkdir(exist_ok=True)
(OUT_DIR/"review").mkdir(exist_ok=True)
```

### Cell 3: load models
```python
det_id = "IDEA-Research/grounding-dino-base"
det_proc = AutoProcessor.from_pretrained(det_id)
det = AutoModelForZeroShotObjectDetection.from_pretrained(det_id).to(dev).eval()
sam_proc = SamProcessor.from_pretrained("facebook/sam-vit-large")
sam = SamModel.from_pretrained("facebook/sam-vit-large").to(dev).eval()
```

### Cell 4: core functions
```python
def detect(img, text, thr=DET_THR):
    inp = det_proc(images=img, text=text, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = det(**inp)
    kw = dict(text_threshold=thr, target_sizes=[img.size[::-1]])
    try:   # name of the threshold argument differs between transformers versions
        res = det_proc.post_process_grounded_object_detection(out, inp.input_ids, threshold=thr, **kw)[0]
    except TypeError:
        res = det_proc.post_process_grounded_object_detection(out, inp.input_ids, box_threshold=thr, **kw)[0]
    return res["boxes"].cpu().numpy(), res["scores"].cpu().numpy()

def sam_mask(img, box):
    inp = sam_proc(img, input_boxes=[[[float(v) for v in box]]], return_tensors="pt").to(dev)
    with torch.no_grad():
        out = sam(**inp, multimask_output=True)
    masks = sam_proc.image_processor.post_process_masks(
        out.pred_masks.cpu(), inp["original_sizes"].cpu(), inp["reshaped_input_sizes"].cpu())[0][0]
    best = out.iou_scores.cpu()[0, 0].argmax().item()
    return masks[best].numpy().astype(bool)

def clean(m):
    m = ndi.binary_fill_holes(m)                       # interior holes only
    lab, n = ndi.label(m)
    if n > 1:
        sizes = ndi.sum(m, lab, range(1, n + 1))
        m = lab == (1 + int(np.argmax(sizes)))
    return m

def process(img):
    """img: PIL RGB. Returns (square PIL RGB on grey, RGBA PIL cropped, info dict). Reuse at prediction time."""
    info = {"flags": []}
    boxes, scores = detect(img, "an apple.")
    if len(boxes) == 0:
        info["flags"].append("no_apple"); return None, None, info
    areas = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
    i = int(np.argmax(areas))
    info["det_score"] = round(float(scores[i]), 3)
    info["n_similar_boxes"] = int((areas > 0.6 * areas[i]).sum())
    if info["n_similar_boxes"] > 1: info["flags"].append("several_big_apples")
    if scores[i] < 0.4:             info["flags"].append("low_det_score")

    apple = clean(sam_mask(img, boxes[i]))
    info["mask_over_box"] = round(float(apple.sum() / areas[i]), 3)
    if info["mask_over_box"] < 0.5:  info["flags"].append("mask_small_vs_box")

    # fingers / hand
    hb, _ = detect(img, "a hand. a finger.")
    hand = np.zeros_like(apple)
    for b in hb:
        hand |= sam_mask(img, b)
    hand = cv2.dilate(hand.astype(np.uint8), np.ones((5,5), np.uint8)) > 0
    r = max(3, int(0.12 * np.sqrt(apple.sum())))
    core = cv2.erode(apple.astype(np.uint8),
                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))) > 0
    final = clean((apple & ~hand) | core)
    removed = 1 - final.sum() / apple.sum()
    info["hand_removed_frac"] = round(float(removed), 3)
    if removed > 0.25:                 # untrustworthy: fall back to plain apple mask
        final = apple; info["flags"].append("hand_overlap_review")

    # 1px feather, no erosion
    alpha = cv2.GaussianBlur(final.astype(np.float32) * 255, (0,0), 0.8).clip(0,255).astype(np.uint8)
    alpha[final] = 255
    arr = np.array(img)
    ys, xs = np.where(final)
    y0, y1, x0, x1 = ys.min(), ys.max()+1, xs.min(), xs.max()+1
    p = int(PAD * max(y1-y0, x1-x0))
    y0, x0 = max(0, y0-p), max(0, x0-p); y1, x1 = min(arr.shape[0], y1+p), min(arr.shape[1], x1+p)
    rgb, a = arr[y0:y1, x0:x1], alpha[y0:y1, x0:x1, None] / 255.0
    comp = (rgb * a + np.array(BG) * (1 - a)).astype(np.uint8)
    h, w = comp.shape[:2]; s = max(h, w)
    sq = np.full((s, s, 3), BG, np.uint8)
    sq[(s-h)//2:(s-h)//2+h, (s-w)//2:(s-w)//2+w] = comp
    out = Image.fromarray(sq).resize((SIZE, SIZE), Image.LANCZOS)
    rgba = Image.fromarray(np.dstack([rgb, alpha[y0:y1, x0:x1]]))
    return out, rgba, info
```

### Cell 5: run on a small test batch FIRST (20 images)
```python
exts = {".jpg", ".jpeg", ".png"}
files = sorted(p for p in IN_DIR.rglob("*") if p.suffix.lower() in exts)
print(len(files), "images")
def label_of(p):
    rel = p.relative_to(IN_DIR).parts
    return rel[0] if len(rel) > 1 else "unlabeled"

def run(file_list):
    rows = []
    for p in tqdm(file_list):
        cls = label_of(p)
        try:
            img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
            out, rgba, info = process(img)
        except Exception as e:
            out, rgba, info = None, None, {"flags": [f"error:{type(e).__name__}"]}
        stem = f"{cls}__{p.stem}"
        if out is not None:
            (OUT_DIR/"jpg"/cls).mkdir(exist_ok=True); (OUT_DIR/"png_rgba"/cls).mkdir(exist_ok=True)
            out.save(OUT_DIR/"jpg"/cls/f"{p.stem}.jpg", quality=95)
            rgba.save(OUT_DIR/"png_rgba"/cls/f"{p.stem}.png")
        rows.append({"file": str(p), "cls": cls, **info, "flags": ";".join(info["flags"])})
    return pd.DataFrame(rows)

test_df = run(random.sample(files, min(20, len(files))))
test_df
```

**STOP and look at the 20 results** (use the contact sheet in the next cell). Only continue if the apples look right. If they do not, change the model sizes or thresholds before running everything.

### Cell 6: contact sheet for visual checks
```python
def sheet(paths, out_path, cols=6, tile=220):
    paths = list(paths)
    rows = (len(paths) + cols - 1) // cols
    S = Image.new("RGB", (cols*tile, rows*tile), (255, 0, 255))
    for k, pth in enumerate(paths):
        t = Image.open(pth).convert("RGB").resize((tile, tile))
        S.paste(t, ((k % cols)*tile, (k // cols)*tile))
    S.save(out_path); return out_path

done = sorted((OUT_DIR/"jpg").rglob("*.jpg"))
sheet(done[:36], OUT_DIR/"qc_sheet_test.png")
```
Open `qc_sheet_test.png` and check: no fingers, edges follow the apple, stems and defects intact.

### Cell 7: full run, report and review sheets
```python
df = run(files)
df.to_csv(OUT_DIR/"report.csv", index=False)
print(df["flags"].replace("", "ok").value_counts())

flagged = df[df.flags != ""]
# Review sheet of flagged images (cutouts that exist)
fl = [OUT_DIR/"jpg"/r.cls/(Path(r.file).stem + ".jpg") for r in flagged.itertuples()]
fl = [p for p in fl if p.exists()]
if fl: sheet(fl[:60], OUT_DIR/"review"/"flagged_sheet.png")
# Random sample per class for QC
for c in df.cls.unique():
    s = sorted((OUT_DIR/"jpg"/c).glob("*.jpg"))
    if s: sheet(random.sample(s, min(36, len(s))), OUT_DIR/f"qc_sheet_{c}.png")
```

### Cell 8: shortcut check (important for the classifier)
```python
print(df.groupby("cls")[["mask_over_box", "hand_removed_frac", "det_score"]].mean())
print(df.groupby("cls")["flags"].apply(lambda s: (s != "").mean()))
```
If one class has clearly different mask statistics or a higher flag rate (for example bad apples get rougher masks), the classifier can learn "rough edge = bad". In that case fix the masks for that class manually before training.

### Cell 9: package the output
```python
!cd /kaggle/working && zip -rq apples_clean.zip apples_clean
```
Then Save Version, Save & Run All, and download `apples_clean.zip` from the Output tab (or Create Dataset from the output).

## Acceptance checklist (the agent must confirm each one)
- [ ] All images are in `report.csv`, with a flag count and a list of the failures.
- [ ] I looked at `qc_sheet_*.png` and `flagged_sheet.png`. No fingers remain; stems and edge defects are intact.
- [ ] Flagged images were reviewed. Unfixable ones are listed by filename so I can check them by hand.
- [ ] Cell 8 shows no class-dependent difference in mask quality.
- [ ] The notebook is saved as a version so the output persists.

## Using it for the classifier (do not skip)
1. **Split by apple, not by photo.** If the same apple appears in several photos, keep all of its photos in the same split (train, val or test), or accuracy will be inflated.
2. **Run a quick ablation:** train the same model on the raw photos and on these cutouts, using the same split, then compare accuracy on the held-out set. This shows whether the cutouts help.
3. **Apply the identical `process()` at prediction time** to every new photo, with the same `SIZE` and `BG`.
4. Use the grey-background JPGs (3 channels) for training. The RGBA PNGs are for inspection only.
5. If a class ends up with very few examples, balance the classes (augmentation, class weights) rather than altering the masks.

## Troubleshooting
- Out of GPU memory: use `facebook/sam-vit-base` and `grounding-dino-tiny`.
- `post_process_grounded_object_detection` raises an argument error: the code already tries both `threshold` and `box_threshold`; if it still fails, print the function signature and adapt.
- No apple detected: lower `DET_THR` to 0.2.
- Wrong apple picked (a crate apple is chosen): this shows up as `several_big_apples`. Select the box nearest the image centre or the sharpest one (Laplacian variance) instead of the largest.
- The code was written without being run on Kaggle, so do the 20-image test first and fix any version issues before the full run.
