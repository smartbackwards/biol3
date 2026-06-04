"""
prepare_data.py

Creates data manifests for the L3 in-the-wild face authentication project.

Reads the CelebA identity subset from L1, selects 100 identities with the most
images, and splits them into enrollment + balanced test sets.

Outputs (in data/):
  enroll.csv          - 2 images per identity for enrollment
  test_clean.csv      - 1000 clean test trials (genuine + impostor)
  test_lowres.csv     - 2500 low-res test trials (images augmented at test time)
  test_occluded.csv   - 2500 occlusion test trials (images augmented at test time)

Usage:
  python prepare_data.py
  python prepare_data.py --identity_txt path/to/identity_subset.txt
"""

import os, csv, random, argparse
from pathlib import Path
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent.resolve()
L1_SUBSET    = SCRIPT_DIR / ".." / "L1" / "facial_auth" / "celeba_subset"
IDENTITY_TXT = L1_SUBSET / "identity_subset.txt"
IMG_DIR      = L1_SUBSET / "img_align_celeba"

OUT_DIR           = SCRIPT_DIR / "data"
ENROLL_CSV        = OUT_DIR / "enroll.csv"
TEST_CLEAN_CSV    = OUT_DIR / "test_clean.csv"
TEST_LOWRES_CSV   = OUT_DIR / "test_lowres.csv"
TEST_OCCLUDED_CSV = OUT_DIR / "test_occluded.csv"

# ── Config ─────────────────────────────────────────────────────────────────────
MIN_IMAGES     = 10     # identities with fewer images are skipped
N_ENROLLED     = 100    # target number of enrolled identities
N_ENROLL_IMGS  = 2      # images per identity used for enrollment (not used in tests)
N_CLEAN        = 1000   # total clean test trials
N_DIFFICULT    = 2500   # total difficult test trials per augmentation type
N_IMP_PER_IMG  = 3      # impostor trials created per test image (different claimed IDs)
SEED           = 42


def load_identity_map(txt_path: Path) -> dict:
    """Returns {identity_str: [img_name, ...]} from identity_subset.txt."""
    id_map = defaultdict(list)
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                id_map[parts[1]].append(parts[0])
    return dict(id_map)


def make_trials(test_imgs: list, n_trials: int,
                rng: random.Random, n_imp: int = N_IMP_PER_IMG) -> list:
    """
    Build balanced genuine + impostor trial list.

    test_imgs : list of (abs_image_path, identity_str)
    n_trials  : desired total (capped by available data)
    n_imp     : number of impostor trials to create per test image
    """
    by_id = defaultdict(list)
    for path, ident in test_imgs:
        by_id[ident].append(path)
    ids = list(by_id.keys())

    genuine  = []
    impostor = []

    for ident, paths in by_id.items():
        other_ids = [x for x in ids if x != ident]
        for p in paths:
            genuine.append({
                "image_path": p, "true_identity": ident,
                "claimed_identity": ident, "is_genuine": True,
            })
            for claimed in rng.sample(other_ids, min(n_imp, len(other_ids))):
                impostor.append({
                    "image_path": p, "true_identity": ident,
                    "claimed_identity": claimed, "is_genuine": False,
                })

    half       = n_trials // 2
    gen_sample = rng.sample(genuine,  min(half, len(genuine)))
    imp_sample = rng.sample(impostor, min(half, len(impostor)))
    trials     = gen_sample + imp_sample
    rng.shuffle(trials)
    return trials


def main(identity_txt: Path = IDENTITY_TXT):
    rng = random.Random(SEED)

    id_map = load_identity_map(identity_txt)
    print(f"Loaded {len(id_map)} identities, "
          f"{sum(len(v) for v in id_map.values())} images total")

    # Keep only identities with enough images; sort descending by count
    qualified = {k: v for k, v in id_map.items() if len(v) >= MIN_IMAGES}
    print(f"Identities with >= {MIN_IMAGES} images: {len(qualified)}")

    selected_count = min(N_ENROLLED, len(qualified))
    selected = sorted(qualified.items(), key=lambda x: -len(x[1]))[:selected_count]
    print(f"Selected {len(selected)} identities for enrollment")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    enroll_rows = []
    test_imgs   = []   # (abs_path, identity)

    for ident, img_names in selected:
        imgs = list(img_names)
        rng.shuffle(imgs)

        enroll_subset = imgs[:N_ENROLL_IMGS]
        test_subset   = imgs[N_ENROLL_IMGS:]

        for img in enroll_subset:
            enroll_rows.append({
                "image_path": str((IMG_DIR / img).resolve()),
                "identity":   ident,
            })
        for img in test_subset:
            test_imgs.append((str((IMG_DIR / img).resolve()), ident))

    # ── Write enrollment manifest ──────────────────────────────────────────────
    with open(ENROLL_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "identity"])
        w.writeheader()
        w.writerows(enroll_rows)
    print(f"Enrollment : {len(enroll_rows)} images → {ENROLL_CSV.name}")

    # ── Write test manifests ───────────────────────────────────────────────────
    configs = [
        (TEST_CLEAN_CSV,    N_CLEAN),
        (TEST_LOWRES_CSV,   N_DIFFICULT),
        (TEST_OCCLUDED_CSV, N_DIFFICULT),
    ]
    for csv_path, n_trials in configs:
        trials = make_trials(test_imgs, n_trials, rng)
        n_gen  = sum(1 for t in trials if t["is_genuine"])
        n_imp  = len(trials) - n_gen
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["image_path", "true_identity",
                                              "claimed_identity", "is_genuine"])
            w.writeheader()
            w.writerows(trials)
        print(f"{csv_path.name:<26} {len(trials):>5} trials "
              f"({n_gen} genuine + {n_imp} impostor)")

    print(f"\nDone. Next: python enroll.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity_txt", default=str(IDENTITY_TXT))
    args = parser.parse_args()
    main(Path(args.identity_txt))
