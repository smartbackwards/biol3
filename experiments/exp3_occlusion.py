"""
experiments/exp3_occlusion.py

Occlusion challenge — face partially covered by a synthetic mask or glasses.

The 2500-trial manifest is split into two equal groups, one per occlusion type
(mask = lower face covered, glasses = eye region covered).  Every trial in a
group is run in two conditions:

  no-fix  — single retinaface detector; score=0 on detection failure
  robust  — cascade of 4 detectors + enforce_detection=False fallback

Total inference calls: 2500 trials × 2 conditions = 5 000  (~1-2 h on CPU)

Results compared against the clean EER from exp1.

Outputs: results/exp3/
  scores.csv, metrics.txt, roc.png

Usage:
  python experiments/exp3_occlusion.py
  python experiments/exp3_occlusion.py --rerun
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import csv, time, pickle, argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from verify import load_db, get_embedding, score_against_identity
from augment import make_occluded
from metrics import threshold_sweep, compute_eer

OUT_DIR     = Path("results/exp3")
SCORES_CSV  = OUT_DIR / "scores.csv"
METRICS_TXT = OUT_DIR / "metrics.txt"
ROC_PNG     = OUT_DIR / "roc.png"

OCC_TYPES = ["mask", "glasses"]
COLORS    = {
    ("mask",    False): "#e74c3c",
    ("mask",    True):  "#2ecc71",
    ("glasses", False): "#e67e22",
    ("glasses", True):  "#27ae60",
}


def assign_occ_types(rows: list, occ_types: list, seed: int = 42) -> list:
    """
    Round-robin assign one occlusion type to each trial so each group has
    roughly the same number of genuine and impostor trials.
    """
    import random
    rng = random.Random(seed)
    gen = [r for r in rows if str(r["is_genuine"]).lower() == "true"]
    imp = [r for r in rows if str(r["is_genuine"]).lower() != "true"]
    rng.shuffle(gen)
    rng.shuffle(imp)

    result = []
    for i, row in enumerate(gen):
        result.append({**row, "occlusion_type": occ_types[i % len(occ_types)]})
    for i, row in enumerate(imp):
        result.append({**row, "occlusion_type": occ_types[i % len(occ_types)]})
    return result


def run_trial(db, img_path: str, claimed: str,
              occ_type: str, robust: bool) -> tuple:
    img   = cv2.imread(img_path)
    img   = make_occluded(img, occlusion_type=occ_type)
    t0    = time.perf_counter()
    score = 0.0
    try:
        emb   = get_embedding(img, enhance=False, robust=robust)
        if claimed in db:
            score = score_against_identity(emb, db[claimed])
    except Exception:
        pass
    lat = (time.perf_counter() - t0) * 1000
    return score, lat


def run_all(db: dict, manifest_path: str) -> list:
    rows = []
    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))

    # Each trial gets one occlusion type; each trial is run twice (no-fix + robust)
    assigned = assign_occ_types(rows, OCC_TYPES)
    total    = len(assigned) * 2

    results = []
    with tqdm(total=total, desc="exp3 occlusion") as pbar:
        for row in assigned:
            img_path = row["image_path"]
            claimed  = str(row["claimed_identity"])
            genuine  = str(row["is_genuine"]).lower() == "true"
            occ_type = row["occlusion_type"]
            for robust in (False, True):
                score, lat = run_trial(db, img_path, claimed, occ_type, robust)
                results.append({
                    "image_path":       img_path,
                    "claimed_identity": claimed,
                    "is_genuine":       genuine,
                    "occlusion_type":   occ_type,
                    "robust":           robust,
                    "score":            score,
                    "latency_ms":       lat,
                })
                pbar.update(1)
    return results


def subset_metrics(results: list, occ_type: str,
                   robust: bool, eer_baseline: float) -> str:
    sub = [r for r in results
           if r["occlusion_type"] == occ_type and r["robust"] == robust]
    if not sub:
        return ""
    scores = np.array([r["score"] for r in sub])
    labels = np.array([1 if r["is_genuine"] else 0 for r in sub])
    thresholds, fars, frrs = threshold_sweep(scores, labels)
    eer, eer_t = compute_eer(thresholds, fars, frrs)
    delta = eer - eer_baseline
    cond  = "robust  " if robust else "no-fix  "
    return (f"  occ={occ_type:<8}  {cond}  "
            f"EER={eer:.4f} ({eer*100:.2f}%)  "
            f"Δ={delta*100:+.2f}pp  "
            f"n={len(sub)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",     default="data/test_occluded.csv")
    parser.add_argument("--db_path",      default="database/embeddings.pkl")
    parser.add_argument("--baseline_dir", default="results/exp1")
    parser.add_argument("--rerun",        action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load baseline EER ─────────────────────────────────────────────────────
    eer_baseline = 0.0
    pkl_path     = Path(args.baseline_dir) / "scores_clean.pkl"
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)
        eer_baseline = float(d["eer"])
        print(f"Baseline EER (exp1): {eer_baseline:.4f}")
    else:
        print("[warn] Baseline scores not found — run exp1_clean.py first")

    # ── Run or load ───────────────────────────────────────────────────────────
    if SCORES_CSV.exists() and not args.rerun:
        print(f"Loading existing scores from {SCORES_CSV}")
        results = []
        with open(SCORES_CSV, newline="") as f:
            for row in csv.DictReader(f):
                results.append({
                    "image_path":       row["image_path"],
                    "claimed_identity": row["claimed_identity"],
                    "is_genuine":       row["is_genuine"].lower() == "true",
                    "occlusion_type":   row["occlusion_type"],
                    "robust":           row["robust"].lower() == "true",
                    "score":            float(row["score"]),
                    "latency_ms":       float(row["latency_ms"]),
                })
    else:
        db      = load_db(args.db_path)
        results = run_all(db, args.manifest)
        with open(SCORES_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["image_path", "claimed_identity",
                                              "is_genuine", "occlusion_type", "robust",
                                              "score", "latency_ms"])
            w.writeheader()
            w.writerows(results)
        print(f"Scores saved → {SCORES_CSV}")

    # ── Metrics ───────────────────────────────────────────────────────────────
    lines = [
        "Experiment 3 — Occlusion Challenge",
        "=" * 55,
        f"Baseline EER (clean): {eer_baseline:.4f} ({eer_baseline*100:.2f}%)",
        f"Each occ-type group:  ~{len(results) // (len(OCC_TYPES)*2)} trials",
        "",
    ]
    for occ_type in OCC_TYPES:
        for robust in (False, True):
            lines.append(subset_metrics(results, occ_type, robust, eer_baseline))
    lines.append("")
    summary = "\n".join(lines)
    print("\n" + summary)
    with open(METRICS_TXT, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"Metrics saved → {METRICS_TXT}")

    # ── ROC plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    for occ_type in OCC_TYPES:
        for robust in (False, True):
            sub = [r for r in results
                   if r["occlusion_type"] == occ_type and r["robust"] == robust]
            if not sub:
                continue
            scores = np.array([r["score"] for r in sub])
            labels = np.array([1 if r["is_genuine"] else 0 for r in sub])
            thresholds, fars, frrs = threshold_sweep(scores, labels)
            eer, _ = compute_eer(thresholds, fars, frrs)
            color  = COLORS[(occ_type, robust)]
            ls     = "-" if robust else "--"
            tag    = "robust" if robust else "raw"
            ax.plot(fars, 1 - frrs, color=color, lw=2, ls=ls,
                    label=f"{occ_type} {tag}  EER={eer:.2%}")

    ax.axhline(1 - eer_baseline, color="black", lw=1.2, ls=":",
               label=f"clean EER={eer_baseline:.2%}")
    ax.set_xlabel("FAR")
    ax.set_ylabel("TAR (1 − FRR)")
    ax.set_title("ROC — Experiment 3: Occlusion Challenge")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROC_PNG, dpi=150)
    plt.close(fig)
    print(f"ROC saved → {ROC_PNG}")


if __name__ == "__main__":
    main()
