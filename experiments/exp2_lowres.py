"""
experiments/exp2_lowres.py

Low-resolution / CCTV-quality challenge.

The 2500-trial manifest is split into three equal groups, each group tested at
one downscale factor (2×, 4×, 8×).  Every trial in a group is run in two
conditions:

  no-fix   — augmented image fed directly to the model
  enhanced — CLAHE + unsharp mask applied before detection

Total inference calls: 2500 trials × 2 conditions = 5 000  (~1-2 h on CPU)

Results compared against the clean EER from exp1.

Outputs: results/exp2/
  scores.csv, metrics.txt, roc.png

Usage:
  python experiments/exp2_lowres.py
  python experiments/exp2_lowres.py --rerun
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
from augment import make_low_res
from metrics import threshold_sweep, compute_eer

OUT_DIR     = Path("results/exp2")
SCORES_CSV  = OUT_DIR / "scores.csv"
METRICS_TXT = OUT_DIR / "metrics.txt"
ROC_PNG     = OUT_DIR / "roc.png"

FACTORS      = [2, 4, 8]
COLORS_NOFIX = {2: "#f39c12", 4: "#e74c3c", 8: "#8e44ad"}
COLORS_FIX   = {2: "#2ecc71", 4: "#27ae60", 8: "#1abc9c"}


def assign_factors(rows: list, factors: list, seed: int = 42) -> list:
    """
    Round-robin assign one factor to each trial so every factor group has
    roughly the same number of genuine and impostor trials.
    Returns rows with an added 'factor' key.
    """
    import random
    rng = random.Random(seed)
    # Separate genuine/impostor then interleave so assignments are balanced
    gen = [r for r in rows if str(r["is_genuine"]).lower() == "true"]
    imp = [r for r in rows if str(r["is_genuine"]).lower() != "true"]
    rng.shuffle(gen)
    rng.shuffle(imp)

    result = []
    for i, row in enumerate(gen):
        result.append({**row, "factor": factors[i % len(factors)]})
    for i, row in enumerate(imp):
        result.append({**row, "factor": factors[i % len(factors)]})
    return result


def run_trial(db, img_path: str, claimed: str,
              factor: int, enhance: bool) -> tuple:
    img   = cv2.imread(img_path)
    img   = make_low_res(img, factor=factor)
    t0    = time.perf_counter()
    score = 0.0
    try:
        emb   = get_embedding(img, enhance=enhance, robust=False)
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

    # Each trial gets one factor; each trial is run twice (no-fix + enhanced)
    assigned = assign_factors(rows, FACTORS)
    total    = len(assigned) * 2   # 2 conditions per trial

    results = []
    with tqdm(total=total, desc="exp2 low-res") as pbar:
        for row in assigned:
            img_path = row["image_path"]
            claimed  = str(row["claimed_identity"])
            genuine  = str(row["is_genuine"]).lower() == "true"
            factor   = int(row["factor"])
            for enhance in (False, True):
                score, lat = run_trial(db, img_path, claimed, factor, enhance)
                results.append({
                    "image_path":       img_path,
                    "claimed_identity": claimed,
                    "is_genuine":       genuine,
                    "factor":           factor,
                    "enhance":          enhance,
                    "score":            score,
                    "latency_ms":       lat,
                })
                pbar.update(1)
    return results


def subset_metrics(results: list, factor: int,
                   enhance: bool, eer_baseline: float) -> str:
    sub = [r for r in results
           if r["factor"] == factor and r["enhance"] == enhance]
    if not sub:
        return ""
    scores = np.array([r["score"] for r in sub])
    labels = np.array([1 if r["is_genuine"] else 0 for r in sub])
    thresholds, fars, frrs = threshold_sweep(scores, labels)
    eer, _  = compute_eer(thresholds, fars, frrs)
    delta   = eer - eer_baseline
    cond    = "enhanced" if enhance else "no-fix  "
    return (f"  factor={factor}x  {cond}  "
            f"EER={eer:.4f} ({eer*100:.2f}%)  "
            f"Δ={delta*100:+.2f}pp  "
            f"n={len(sub)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",     default="data/test_lowres.csv")
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
                    "factor":           int(row["factor"]),
                    "enhance":          row["enhance"].lower() == "true",
                    "score":            float(row["score"]),
                    "latency_ms":       float(row["latency_ms"]),
                })
    else:
        db      = load_db(args.db_path)
        results = run_all(db, args.manifest)
        with open(SCORES_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["image_path", "claimed_identity",
                                              "is_genuine", "factor", "enhance",
                                              "score", "latency_ms"])
            w.writeheader()
            w.writerows(results)
        print(f"Scores saved → {SCORES_CSV}")

    # ── Metrics ───────────────────────────────────────────────────────────────
    lines = [
        "Experiment 2 — Low-Resolution Challenge",
        "=" * 55,
        f"Baseline EER (clean): {eer_baseline:.4f} ({eer_baseline*100:.2f}%)",
        f"Each factor group:    ~{len(results) // (len(FACTORS)*2)} trials",
        "",
    ]
    for factor in FACTORS:
        for enhance in (False, True):
            lines.append(subset_metrics(results, factor, enhance, eer_baseline))
    lines.append("")
    summary = "\n".join(lines)
    print("\n" + summary)
    with open(METRICS_TXT, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"Metrics saved → {METRICS_TXT}")

    # ── ROC plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    for factor in FACTORS:
        for enhance in (False, True):
            sub = [r for r in results
                   if r["factor"] == factor and r["enhance"] == enhance]
            if not sub:
                continue
            scores = np.array([r["score"] for r in sub])
            labels = np.array([1 if r["is_genuine"] else 0 for r in sub])
            thresholds, fars, frrs = threshold_sweep(scores, labels)
            eer, _ = compute_eer(thresholds, fars, frrs)
            color  = COLORS_FIX[factor] if enhance else COLORS_NOFIX[factor]
            ls     = "-" if enhance else "--"
            tag    = "enh" if enhance else "raw"
            ax.plot(fars, 1 - frrs, color=color, lw=2, ls=ls,
                    label=f"×{factor} {tag}  EER={eer:.2%}")

    ax.axhline(1 - eer_baseline, color="black", lw=1.2, ls=":",
               label=f"clean EER={eer_baseline:.2%}")
    ax.set_xlabel("FAR")
    ax.set_ylabel("TAR (1 − FRR)")
    ax.set_title("ROC — Experiment 2: Low-Resolution Challenge")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROC_PNG, dpi=150)
    plt.close(fig)
    print(f"ROC saved → {ROC_PNG}")


if __name__ == "__main__":
    main()
