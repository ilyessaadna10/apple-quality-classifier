#!/usr/bin/env python3
"""Grade apples as good or bad with the 5-fold ConvNeXt-Tiny ensemble.

Input: background-removed apple images (RGBA PNG cut-outs), or apples already
composited on a neutral gray (128) background.
Output: P(bad) per image, averaged over the 5 fold models and 4 flip views.

    python predict.py --models ../models path/to/apple.png [more.png ...]
"""
import argparse
import glob
import time
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage as ndi

IMG_SIZE = 384      # training resolution
APPLE_FILL = 0.92   # apple size relative to the canvas, same as training
PRELOAD_SIDE = 512
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_rgb_alpha(path):
    im = Image.open(path)
    if im.mode == "RGBA":
        a = np.array(im)
        return a[..., :3].copy(), a[..., 3].copy()
    rgb = np.array(im.convert("RGB"))
    # gray-background image: recover the apple mask from the neutral (128) background
    m = (np.abs(rgb.astype(np.int16) - 128).max(2) > 10).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    lbl, n = ndi.label(m)
    if n:
        sizes = ndi.sum(m, lbl, range(1, n + 1))
        m = ndi.binary_fill_holes(lbl == (int(np.argmax(sizes)) + 1)).astype(np.uint8)
    return rgb, m * 255


def to_canvas(rgb, a, S=IMG_SIZE):
    """Scale the apple to fill 92% of an SxS gray canvas (matches training eval view)."""
    s = min(1.0, PRELOAD_SIDE / max(a.shape))
    if s < 1.0:
        size = (max(1, round(a.shape[1] * s)), max(1, round(a.shape[0] * s)))
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        a = cv2.resize(a, size, interpolation=cv2.INTER_AREA)
    h, w = a.shape
    k = APPLE_FILL * S / max(h, w)
    size = (max(2, round(w * k)), max(2, round(h * k)))
    interp = cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR
    rgb = cv2.resize(np.ascontiguousarray(rgb), size, interpolation=interp).astype(np.float32)
    a = cv2.resize(np.ascontiguousarray(a), size, interpolation=interp).astype(np.float32) / 255.0
    h, w = a.shape
    M = np.float32([[1, 0, S / 2 - w / 2], [0, 1, S / 2 - h / 2]])
    rgb = cv2.warpAffine(rgb, M, (S, S), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    a = cv2.warpAffine(a, M, (S, S), flags=cv2.INTER_LINEAR, borderValue=0)
    img = rgb * a[..., None] + 128.0 * (1 - a[..., None])
    return np.clip(img, 0, 255).astype(np.uint8)


class AppleGrader:
    def __init__(self, model_dir, device=None, fp16=False, tta=True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # fp16 is slower than fp32 on GTX 16xx cards (no tensor cores), so it is opt-in
        self.fp16, self.tta = fp16 and self.device == "cuda", tta
        torch.backends.cudnn.benchmark = True
        self.models = []
        for p in sorted(glob.glob(str(Path(model_dir) / "convnext_fold*.pt"))):
            m = timm.create_model("convnext_tiny", pretrained=False, num_classes=2)
            sd = torch.load(p, map_location="cpu")
            m.load_state_dict({k: v.float() for k, v in sd.items()})
            self.models.append(m.to(self.device).eval().to(memory_format=torch.channels_last))
        if not self.models:
            raise FileNotFoundError(f"no convnext_fold*.pt in {model_dir}")

    @torch.no_grad()
    def p_bad(self, canvases):
        x = torch.from_numpy(np.stack(canvases)).permute(0, 3, 1, 2).float() / 255.0
        x = ((x - MEAN) / STD).to(self.device).contiguous(memory_format=torch.channels_last)
        n = x.shape[0]
        # all flip views in one batch: [original, h-flip, v-flip, both]
        xv = torch.cat([x, x.flip(3), x.flip(2), x.flip(2).flip(3)]) if self.tta else x
        probs = []
        with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.fp16):
            for m in self.models:
                p = F.softmax(m(xv).float(), 1)[:, 1]
                probs.append(p.view(-1, n).mean(0))
        return torch.stack(probs).mean(0).cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--models", default=str(Path(__file__).resolve().parent.parent / "models"))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--fast", action="store_true", help="one model, no flip views (for real-time use)")
    args = ap.parse_args()
    grader = AppleGrader(args.models, tta=not args.fast)
    if args.fast:
        grader.models = grader.models[:1]
    for path in args.images:
        t = time.perf_counter()
        p = float(grader.p_bad([to_canvas(*load_rgb_alpha(path))])[0])
        ms = (time.perf_counter() - t) * 1000
        print(f"{'BAD ' if p >= args.threshold else 'GOOD'}  P(bad)={p:.3f}  {ms:6.1f} ms  {path}")


if __name__ == "__main__":
    main()
