"""
demo.py — Interactive face authentication demo for L3 in-the-wild

Shows clean → degraded → recovered side-by-side for a single image.

Usage:
  python demo.py path/to/face.jpg identity_id
  python demo.py path/to/face.jpg identity_id --simulate lowres
  python demo.py path/to/face.jpg identity_id --simulate mask
  python demo.py path/to/face.jpg identity_id --simulate glasses
"""

import argparse, tempfile, os
import cv2
import numpy as np

from augment import make_low_res, make_occluded
from verify  import verify, DEFAULT_THRESHOLD


def annotate(img: np.ndarray, label: str, accepted: bool) -> np.ndarray:
    out   = img.copy()
    color = (0, 180, 0) if accepted else (0, 0, 200)
    cv2.putText(out, label, (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("identity")
    parser.add_argument("--simulate", choices=["lowres", "mask", "glasses"],
                        default=None,
                        help="Degradation type to simulate")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--db_path",   default="database/embeddings.pkl")
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    if img is None:
        print(f"Cannot read: {args.image_path}")
        return

    # ── Clean result ──────────────────────────────────────────────────────────
    acc_clean, score_clean = verify(
        args.image_path, args.identity,
        threshold=args.threshold, db_path=args.db_path,
    )
    print(f"Clean    : {'ACCEPT' if acc_clean else 'REJECT'}  "
          f"score={score_clean:.4f}")

    if args.simulate is None:
        panel = annotate(img,
                         f"Clean {'ACCEPT' if acc_clean else 'REJECT'} "
                         f"({score_clean:.3f})", acc_clean)
        cv2.imshow("L3 Demo", panel)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # ── Apply degradation ─────────────────────────────────────────────────────
    if args.simulate == "lowres":
        aug_img = make_low_res(img, factor=4)
        enhance, robust = True, False
    else:
        aug_img = make_occluded(img, occlusion_type=args.simulate)
        enhance, robust = False, True

    # Save to a temp file so verify() can read it
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(tmp.name, aug_img)
    tmp.close()

    acc_raw,  score_raw  = verify(tmp.name, args.identity,
                                  threshold=args.threshold,
                                  enhance=False, robust=False,
                                  db_path=args.db_path)
    acc_fix,  score_fix  = verify(tmp.name, args.identity,
                                  threshold=args.threshold,
                                  enhance=enhance, robust=robust,
                                  db_path=args.db_path)
    os.unlink(tmp.name)

    print(f"Degraded : {'ACCEPT' if acc_raw else 'REJECT'}  "
          f"score={score_raw:.4f}  (no fix)")
    print(f"Recovered: {'ACCEPT' if acc_fix else 'REJECT'}  "
          f"score={score_fix:.4f}  (with fix)")

    # ── Display ───────────────────────────────────────────────────────────────
    h, w = img.shape[:2]
    gap  = np.full((h, 10, 3), 200, dtype=np.uint8)

    p1 = annotate(img,
                  f"Clean {'ACC' if acc_clean else 'REJ'} ({score_clean:.3f})",
                  acc_clean)
    p2 = annotate(aug_img,
                  f"Degraded {'ACC' if acc_raw else 'REJ'} ({score_raw:.3f})",
                  acc_raw)
    p3 = annotate(aug_img,
                  f"Fixed {'ACC' if acc_fix else 'REJ'} ({score_fix:.3f})",
                  acc_fix)

    display = np.hstack([p1, gap, p2, gap, p3])
    cv2.imshow("L3 In-the-Wild Demo  (any key to close)", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
