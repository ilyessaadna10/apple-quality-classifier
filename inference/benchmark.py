#!/usr/bin/env python3
"""Measure grading latency: full ensemble (5 models x 4 flips) vs a single model."""
import argparse, glob, time
import numpy as np, torch
from predict import AppleGrader, load_rgb_alpha, to_canvas

ap = argparse.ArgumentParser()
ap.add_argument("image_dir"); ap.add_argument("--models", required=True); ap.add_argument("--n", type=int, default=50)
args = ap.parse_args()
paths = sorted(glob.glob(f"{args.image_dir}/**/*.png", recursive=True))[: args.n]
canv = [to_canvas(*load_rgb_alpha(p)) for p in paths]
g = AppleGrader(args.models)
print("device:", g.device, torch.cuda.get_device_name(0) if g.device == "cuda" else "")

def bench(fn, label):
    for c in canv[:3]: fn([c])  # warm-up
    if g.device == "cuda": torch.cuda.synchronize()
    t = time.perf_counter()
    for c in canv: fn([c])
    if g.device == "cuda": torch.cuda.synchronize()
    ms = (time.perf_counter() - t) * 1000 / len(canv)
    print(f"{label:<38} {ms:7.1f} ms/apple  ({1000/ms:5.1f} apples/s)")

bench(g.p_bad, "ensemble: 5 models x 4 flip views")
full = g.models
g.models = full[:1]
bench(g.p_bad, "1 model x 4 flip views")
g.tta = False
bench(g.p_bad, "1 model, 1 view (--fast)")
g.models, g.tta = full, True
