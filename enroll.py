"""
enroll.py — ArcFace enrollment for L3 in-the-wild face authentication

Reads data/enroll.csv and builds database/embeddings.pkl.
Multiple images per identity are stored individually (nearest-neighbour matching
at verify time is more robust than a single mean embedding).

Usage:
  python enroll.py
  python enroll.py --manifest data/enroll.csv --db_path database/embeddings.pkl
"""

import csv, pickle, argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from deepface import DeepFace

DB_PATH    = "database/embeddings.pkl"
MODEL_NAME = "ArcFace"
BACKENDS   = ("retinaface", "mtcnn", "ssd", "opencv")


def get_embedding(image_path: str, detector: str = "retinaface") -> np.ndarray:
    """Extract a normalised ArcFace embedding. Raises on detection failure."""
    result = DeepFace.represent(
        img_path=str(image_path),
        model_name=MODEL_NAME,
        detector_backend=detector,
        enforce_detection=True,
    )
    emb = np.array(result[0]["embedding"], dtype=np.float32)
    return emb / np.linalg.norm(emb)


def get_embedding_robust(image_path: str) -> np.ndarray:
    """Try each backend in order; fall back to enforce_detection=False."""
    for backend in BACKENDS:
        try:
            return get_embedding(image_path, detector=backend)
        except Exception:
            continue
    result = DeepFace.represent(
        img_path=str(image_path),
        model_name=MODEL_NAME,
        detector_backend="opencv",
        enforce_detection=False,
    )
    emb = np.array(result[0]["embedding"], dtype=np.float32)
    return emb / np.linalg.norm(emb)


def enroll_from_manifest(manifest_path: str, db_path: str = DB_PATH) -> dict:
    by_id = defaultdict(list)
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            by_id[row["identity"]].append(row["image_path"])

    db      = {}
    failed  = 0
    print(f"Enrolling {len(by_id)} identities from {manifest_path}")

    for identity, paths in tqdm(by_id.items()):
        embeddings = []
        for p in paths:
            try:
                emb = get_embedding_robust(p)
                embeddings.append(emb)
            except Exception as e:
                print(f"  [warn] {Path(p).name}: {e}")

        if not embeddings:
            print(f"  [warn] No embedding for identity {identity} — skipping")
            failed += 1
            continue

        mean_emb = np.mean(embeddings, axis=0).astype(np.float32)
        mean_emb /= np.linalg.norm(mean_emb)

        db[str(identity)] = {
            "embeddings":    embeddings,
            "mean_embedding": mean_emb,
        }

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with open(db_path, "wb") as f:
        pickle.dump(db, f)

    print(f"\nEnrolled {len(db)} identities "
          f"({failed} failed) → {db_path}")
    return db


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/enroll.csv")
    parser.add_argument("--db_path",  default=DB_PATH)
    args = parser.parse_args()
    enroll_from_manifest(args.manifest, args.db_path)


if __name__ == "__main__":
    main()
