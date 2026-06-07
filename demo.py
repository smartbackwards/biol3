"""
demo.py — Interactive face authentication demo for L3 in-the-wild

Shows clean → degraded → recovered side-by-side for a single image.

Usage (file):
  python demo.py path/to/face.jpg identity_id
  python demo.py path/to/face.jpg identity_id --simulate lowres
  python demo.py path/to/face.jpg identity_id --simulate mask
  python demo.py path/to/face.jpg identity_id --simulate glasses

Usage (webcam):
  python demo.py --webcam identity_id
  python demo.py --webcam identity_id --simulate mask
  python demo.py --webcam identity_id --simulate glasses
"""

import argparse, tempfile, os
import cv2
import numpy as np

from augment import make_low_res, make_occluded
from verify  import verify, DEFAULT_THRESHOLD


def capture_from_webcam(camera_index: int = 0) -> np.ndarray:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")
    print("Webcam open — press SPACE to capture, ESC to quit")
    frame = None
    while True:
        ret, f = cap.read()
        if not ret:
            break
        cv2.putText(f, "SPACE = capture  |  ESC = quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Webcam — L3 Demo", f)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            frame = f.copy()
            break
        elif key == 27:
            break
    cap.release()
    cv2.destroyWindow("Webcam — L3 Demo")
    return frame


def annotate(img: np.ndarray, label: str, accepted: bool) -> np.ndarray:
    out   = img.copy()
    color = (0, 180, 0) if accepted else (0, 0, 200)
    cv2.putText(out, label, (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return out


def run_demo(img: np.ndarray, img_path: str, identity: str,
             simulate: str, threshold: float, db_path: str):
    # ── Clean result ──────────────────────────────────────────────────────────
    acc_clean, score_clean = verify(
        img_path, identity,
        threshold=threshold, db_path=db_path,
    )
    print(f"Clean    : {'ACCEPT' if acc_clean else 'REJECT'}  "
          f"score={score_clean:.4f}")

    if simulate is None:
        panel = annotate(img,
                         f"Clean {'ACCEPT' if acc_clean else 'REJECT'} "
                         f"({score_clean:.3f})", acc_clean)
        cv2.imshow("L3 Demo", panel)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # ── Apply degradation ─────────────────────────────────────────────────────
    if simulate == "lowres":
        aug_img = make_low_res(img, factor=4)
        enhance, robust = True, False
    else:
        aug_img = make_occluded(img, occlusion_type=simulate)
        enhance, robust = False, True

    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(tmp.name, aug_img)
    tmp.close()

    acc_raw,  score_raw  = verify(tmp.name, identity,
                                  threshold=threshold,
                                  enhance=False, robust=False,
                                  db_path=db_path)
    acc_fix,  score_fix  = verify(tmp.name, identity,
                                  threshold=threshold,
                                  enhance=enhance, robust=robust,
                                  db_path=db_path)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", nargs="?", default=None,
                        help="Path to input image (omit when using --webcam)")
    parser.add_argument("identity")
    parser.add_argument("--webcam", action="store_true",
                        help="Capture image from webcam instead of loading a file")
    parser.add_argument("--camera", type=int, default=0,
                        help="Webcam device index (default: 0)")
    parser.add_argument("--simulate", choices=["lowres", "mask", "glasses"],
                        default=None)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--db_path",   default="database/embeddings.pkl")
    args = parser.parse_args()

    if args.webcam:
        img = capture_from_webcam(args.camera)
        if img is None:
            print("No frame captured.")
            return
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        cv2.imwrite(tmp.name, img)
        tmp.close()
        img_path = tmp.name
        cleanup = True
    else:
        if args.image_path is None:
            parser.error("Provide image_path or use --webcam")
        img = cv2.imread(args.image_path)
        if img is None:
            print(f"Cannot read: {args.image_path}")
            return
        img_path = args.image_path
        cleanup = False

    try:
        run_demo(img, img_path, args.identity,
                 args.simulate, args.threshold, args.db_path)
    finally:
        if cleanup:
            os.unlink(img_path)


if __name__ == "__main__":
    main()
