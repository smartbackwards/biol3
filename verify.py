"""
verify.py — Face verification with in-the-wild robustness

Two robustness modes (combinable):
  enhance=True  — CLAHE + unsharp mask before detection  (helps low-res images)
  robust=True   — cascade of 4 detectors + enforce_detection=False fallback
                  (helps occluded or low-quality faces that confuse one detector)

Usage:
  python verify.py path/to/face.jpg identity_id
  python verify.py path/to/face.jpg identity_id --enhance
  python verify.py path/to/face.jpg identity_id --robust
"""

import pickle, argparse
import numpy as np
import cv2
from pathlib import Path
from deepface import DeepFace

DB_PATH           = "database/embeddings.pkl"
MODEL_NAME        = "ArcFace"
DEFAULT_THRESHOLD = 0.40    # cosine similarity; experiments calibrate the EER threshold
BACKENDS = ("mtcnn", "ssd", "opencv", "retinaface")

_db_cache = None


def load_db(db_path: str = DB_PATH) -> dict:
    global _db_cache
    if _db_cache is None:
        with open(db_path, "rb") as f:
            _db_cache = pickle.load(f)
    return _db_cache


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def score_against_identity(query_emb: np.ndarray, identity_data: dict) -> float:
    """Nearest-neighbour cosine similarity against all enrolled embeddings."""
    return max(cosine_sim(query_emb, e) for e in identity_data["embeddings"])


# ── Enhancement ────────────────────────────────────────────────────────────────

def enhance_low_res(img: np.ndarray) -> np.ndarray:
    """
    CLAHE local contrast equalisation + unsharp masking.
    Recovers edge detail and contrast lost by low-resolution capture or blurring.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img_eq = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    blurred = cv2.GaussianBlur(img_eq, (0, 0), 3)
    return cv2.addWeighted(img_eq, 1.5, blurred, -0.5, 0)


# ── Embedding extraction ───────────────────────────────────────────────────────

def get_embedding(img: np.ndarray, enhance: bool = False,
                  robust: bool = False) -> np.ndarray:
    """
    Extract a normalised ArcFace embedding from a BGR numpy image.

    enhance=True  → apply CLAHE + unsharp mask first (helps low-res)
    robust=True   → try multiple detection backends in order (helps occlusions)
    """
    if enhance:
        img = enhance_low_res(img)

    backends = BACKENDS if robust else (BACKENDS[0],)

    for backend in backends:
        try:
            result = DeepFace.represent(
                img_path=img,
                model_name=MODEL_NAME,
                detector_backend=backend,
                enforce_detection=True,
            )
            emb = np.array(result[0]["embedding"], dtype=np.float32)
            return emb / np.linalg.norm(emb)
        except Exception:
            continue

    # Final fallback: accept any face-sized region without landmark checks
    result = DeepFace.represent(
        img_path=img,
        model_name=MODEL_NAME,
        detector_backend="opencv",
        enforce_detection=False,
    )
    emb = np.array(result[0]["embedding"], dtype=np.float32)
    return emb / np.linalg.norm(emb)


# ── Verification (1:1) ─────────────────────────────────────────────────────────

def verify(img_path: str, claimed_identity,
           threshold: float = DEFAULT_THRESHOLD,
           enhance: bool = False,
           robust: bool = False,
           db_path: str = DB_PATH) -> tuple:
    """
    Returns (accepted: bool, score: float).
    score = cosine similarity in [0, 1]; accepted = (score >= threshold).
    Returns (False, 0.0) on any detection failure.
    """
    db  = load_db(db_path)
    key = str(claimed_identity)
    if key not in db:
        return False, 0.0

    img = cv2.imread(str(img_path))
    if img is None:
        return False, 0.0

    try:
        query_emb = get_embedding(img, enhance=enhance, robust=robust)
    except Exception:
        return False, 0.0

    score = score_against_identity(query_emb, db[key])
    return score >= threshold, score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("identity")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--enhance",   action="store_true",
                        help="Apply CLAHE+unsharp mask (for low-res images)")
    parser.add_argument("--robust",    action="store_true",
                        help="Multi-backend detection cascade (for occluded faces)")
    parser.add_argument("--db_path",   default=DB_PATH)
    args = parser.parse_args()

    accepted, score = verify(
        args.image_path, args.identity,
        threshold=args.threshold,
        enhance=args.enhance,
        robust=args.robust,
        db_path=args.db_path,
    )
    verdict = "ACCEPTED" if accepted else "REJECTED"
    print(f"{verdict}  score={score:.4f}  threshold={args.threshold:.4f}")


if __name__ == "__main__":
    main()
