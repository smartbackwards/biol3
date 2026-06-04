"""
experiments/exp1_clean.py

Baseline test on clean (unaugmented) images.
Results saved to results/exp1/ and scores_clean.npy (read by exp2/exp3).

Usage:
  python experiments/exp1_clean.py
  python experiments/exp1_clean.py --rerun   # force re-run even if scores exist
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
from metrics import threshold_sweep, compute_eer, compute_FAR, compute_FRR

OUT_DIR     = Path("results/exp1")
SCORES_CSV  = OUT_DIR / "scores.csv"
METRICS_TXT = OUT_DIR / "metrics.txt"
ROC_PNG     = OUT_DIR / "roc.png"
SCORES_PKL  = OUT_DIR / "scores_clean.pkl"    # shared with exp2 and exp3


def run_trials(db: dict, manifest_path: str) -> list:
    rows = []
    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))

    results = []
    for row in tqdm(rows, desc="exp1 clean baseline"):
        img_path = row["image_path"]
        claimed  = str(row["claimed_identity"])
        genuine  = row["is_genuine"].lower() == "true"
        score    = 0.0
        lat      = 0.0

        try:
            t0  = time.perf_counter()
            img = cv2.imread(img_path)
            emb = get_embedding(img, enhance=False, robust=False)
            if claimed in db:
                score = score_against_identity(emb, db[claimed])
            lat = (time.perf_counter() - t0) * 1000
        except Exception as e:
            pass

        results.append({
            "image_path":       img_path,
            "claimed_identity": claimed,
            "is_genuine":       genuine,
            "score":            score,
            "latency_ms":       lat,
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/test_clean.csv")
    parser.add_argument("--db_path",  default="database/embeddings.pkl")
    parser.add_argument("--rerun",    action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

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
                    "score":            float(row["score"]),
                    "latency_ms":       float(row["latency_ms"]),
                })
    else:
        db      = load_db(args.db_path)
        results = run_trials(db, args.manifest)
        with open(SCORES_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["image_path", "claimed_identity",
                                              "is_genuine", "score", "latency_ms"])
            w.writeheader()
            w.writerows(results)
        print(f"Scores saved → {SCORES_CSV}")

    # ── Metrics ───────────────────────────────────────────────────────────────
    scores = np.array([r["score"] for r in results])
    labels = np.array([1 if r["is_genuine"] else 0 for r in results])

    thresholds, fars, frrs = threshold_sweep(scores, labels)
    eer, eer_t = compute_eer(thresholds, fars, frrs)
    far_at_eer = compute_FAR(scores, labels, eer_t)
    frr_at_eer = compute_FRR(scores, labels, eer_t)
    avg_lat    = np.mean([r["latency_ms"] for r in results])

    lines = [
        "Experiment 1 — Clean Baseline",
        "=" * 50,
        f"Trials      : {len(results)}  "
        f"({int(labels.sum())} genuine + {int((1-labels).sum())} impostor)",
        f"EER         : {eer:.4f}  ({eer*100:.2f}%)",
        f"EER thr     : {eer_t:.4f}",
        f"FAR @ EER   : {far_at_eer:.4f}",
        f"FRR @ EER   : {frr_at_eer:.4f}",
        f"TAR @ EER   : {1 - frr_at_eer:.4f}",
        f"Avg latency : {avg_lat:.1f} ms",
    ]
    summary = "\n".join(lines)
    print("\n" + summary)
    with open(METRICS_TXT, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"Metrics saved → {METRICS_TXT}")

    # ── Save for downstream experiments ───────────────────────────────────────
    with open(SCORES_PKL, "wb") as f:
        pickle.dump({"scores": scores, "labels": labels,
                     "eer": eer, "eer_t": eer_t}, f)
    print(f"Baseline data saved → {SCORES_PKL}")

    # ── ROC curve ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fars, 1 - frrs, lw=2, color="#2980b9",
            label=f"Clean  EER={eer:.2%}")
    ax.scatter([eer], [1 - eer], color="red", zorder=5)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("FAR")
    ax.set_ylabel("TAR (1 − FRR)")
    ax.set_title("ROC — Experiment 1: Clean Baseline")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROC_PNG, dpi=150)
    plt.close(fig)
    print(f"ROC saved → {ROC_PNG}")


if __name__ == "__main__":
    main()
