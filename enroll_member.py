"""
enroll_member.py — Add a group member to the face authentication database

Takes a directory of personal photos and appends that person's embeddings to
the existing database (database/embeddings.pkl).

Usage:
  python enroll_member.py --name "Bartek T." --photos_dir my_photos/bartek_t/
  python enroll_member.py --name "Krzysiek"  --photos_dir my_photos/krzysiek/  --id krzysiek

Photo tips:
  - Take 5-10 photos in varied lighting and angles (front, slight left/right)
  - Plain background, face clearly visible and not occluded
  - Save as JPG or PNG, any resolution (the model handles alignment internally)
  - Keep each photo to one face only

The script tries each detector backend in order (retinaface → mtcnn → ssd → opencv)
so even slightly non-ideal photos usually succeed.
"""

import argparse, pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm
from deepface import DeepFace

DB_PATH    = "database/embeddings.pkl"
MODEL_NAME = "ArcFace"
BACKENDS   = ("retinaface", "mtcnn", "ssd", "opencv")
IMG_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_embedding(img_path: str) -> np.ndarray:
    for backend in BACKENDS:
        try:
            result = DeepFace.represent(
                img_path=str(img_path),
                model_name=MODEL_NAME,
                detector_backend=backend,
                enforce_detection=True,
            )
            emb = np.array(result[0]["embedding"], dtype=np.float32)
            return emb / np.linalg.norm(emb)
        except Exception:
            continue
    # Final fallback
    result = DeepFace.represent(
        img_path=str(img_path),
        model_name=MODEL_NAME,
        detector_backend="opencv",
        enforce_detection=False,
    )
    emb = np.array(result[0]["embedding"], dtype=np.float32)
    return emb / np.linalg.norm(emb)


def load_db(db_path: str) -> dict:
    p = Path(db_path)
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return {}


def save_db(db: dict, db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with open(db_path, "wb") as f:
        pickle.dump(db, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",       required=True,
                        help="Display name, e.g. 'Bartek T.'")
    parser.add_argument("--photos_dir", required=True,
                        help="Directory containing .jpg/.png personal photos")
    parser.add_argument("--id",         default=None,
                        help="Identity key stored in DB (default: sanitised name)")
    parser.add_argument("--db_path",    default=DB_PATH)
    args = parser.parse_args()

    member_id   = args.id or args.name.lower().replace(" ", "_").replace(".", "")
    photos_dir  = Path(args.photos_dir)
    photo_files = sorted(p for p in photos_dir.iterdir()
                         if p.suffix.lower() in IMG_EXTS)

    if not photo_files:
        print(f"No images found in {photos_dir}  "
              f"(looking for {', '.join(IMG_EXTS)})")
        return

    print(f"Enrolling '{args.name}' (id='{member_id}') "
          f"from {len(photo_files)} photos in {photos_dir}")

    embeddings = []
    for p in tqdm(photo_files):
        try:
            emb = get_embedding(str(p))
            embeddings.append(emb)
            print(f"  ✓  {p.name}")
        except Exception as e:
            print(f"  ✗  {p.name}  ({e})")

    if not embeddings:
        print("No embeddings extracted — check that photos contain a clear face.")
        return

    mean_emb = np.mean(embeddings, axis=0).astype(np.float32)
    mean_emb /= np.linalg.norm(mean_emb)

    db = load_db(args.db_path)
    if member_id in db:
        print(f"[info] '{member_id}' already in DB — extending with new embeddings")
        db[member_id]["embeddings"].extend(embeddings)
        # Recompute mean
        all_embs = db[member_id]["embeddings"]
        new_mean = np.mean(all_embs, axis=0).astype(np.float32)
        db[member_id]["mean_embedding"] = new_mean / np.linalg.norm(new_mean)
    else:
        db[member_id] = {
            "embeddings":     embeddings,
            "mean_embedding": mean_emb,
        }

    save_db(db, args.db_path)
    print(f"\nDone. '{args.name}' enrolled with {len(embeddings)} embeddings "
          f"→ {args.db_path}  (DB now has {len(db)} identities)")


if __name__ == "__main__":
    main()
