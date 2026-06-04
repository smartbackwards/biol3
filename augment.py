"""
augment.py — In-the-wild image augmentation for L3

Two difficulty types:

1. make_low_res(img, factor, ...)
   Simulates CCTV / low-quality capture by downsampling then upsampling.
   CelebA images are ~178×218 px; factor=4 → 44×54 effective resolution.

2. make_occluded(img, occlusion_type)
   Adds a synthetic occluder to a face-aligned CelebA image.
   CelebA is centre-cropped so fixed vertical proportions hit the right regions.
"""

import numpy as np
import cv2
import random as _rng


# ── Low-resolution simulation ──────────────────────────────────────────────────

def make_low_res(img: np.ndarray,
                 factor: int = 4,
                 noise_std: float = 0.0,
                 jpeg_quality: int = None) -> np.ndarray:
    """
    Downsample then upsample to simulate pixelated low-quality capture.

    factor       : integer downscale factor (2 = mild, 4 = CCTV, 8 = severe)
    noise_std    : add Gaussian noise with this stddev after upscaling (0 = off)
    jpeg_quality : if set (1-95), add JPEG compression artefacts
    """
    h, w = img.shape[:2]
    small    = cv2.resize(img,
                          (max(1, w // factor), max(1, h // factor)),
                          interpolation=cv2.INTER_AREA)
    restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    if noise_std > 0:
        noise    = np.random.normal(0, noise_std, restored.shape).astype(np.float32)
        restored = np.clip(restored.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if jpeg_quality is not None:
        _, enc   = cv2.imencode(".jpg", restored,
                                [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
        restored = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    return restored


# ── Occlusion simulation ───────────────────────────────────────────────────────

def make_occluded(img: np.ndarray,
                  occlusion_type: str = "mask",
                  seed: int = None) -> np.ndarray:
    """
    Add a synthetic occluder to a face-aligned image.

    occlusion_type:
      "mask"    — light-grey rectangle over nose+mouth (simulates COVID/surgical mask)
      "glasses" — dark bar over the eye region (simulates opaque glasses)
      "random"  — choose one of the above at random
    """
    rng = _rng.Random(seed)
    h, w  = img.shape[:2]
    result = img.copy()

    if occlusion_type == "random":
        occlusion_type = rng.choice(["mask", "glasses"])

    if occlusion_type == "mask":
        # Cover lower ~45 % of the image (nose, mouth, chin)
        y1    = int(h * 0.50)
        color = (rng.randint(180, 220),) * 3    # grey mask
        cv2.rectangle(result, (0, y1), (w, h), color, thickness=-1)
        # Thin shadow line at the top edge of the mask
        cv2.line(result, (0, y1), (w, y1),
                 tuple(max(0, c - 50) for c in color), thickness=2)

    elif occlusion_type == "glasses":
        # Cover the eye band (~28 % – 46 % vertically)
        y1    = int(h * 0.28)
        y2    = int(h * 0.46)
        x_pad = int(w * 0.08)
        # Dark lens fill
        cv2.rectangle(result, (x_pad, y1), (w - x_pad, y2),
                      (20, 20, 20), thickness=-1)
        # Thin bridge in the centre
        cx = w // 2
        cv2.rectangle(result,
                      (cx - 5, y1 + (y2 - y1) // 3),
                      (cx + 5, y2 - (y2 - y1) // 3),
                      (80, 80, 80), thickness=-1)
        # Frame outline
        cv2.rectangle(result, (x_pad, y1), (w - x_pad, y2),
                      (60, 60, 60), thickness=2)

    return result
