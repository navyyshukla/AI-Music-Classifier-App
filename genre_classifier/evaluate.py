"""Metrics, plots, and the out-of-distribution check on your own songs.

    python -m genre_classifier.evaluate --ood-dir Data/ood
OOD folder layout: either Data/ood/<genre>/<song>.mp3, or flat files named <genre>__<song>.mp3.
Genre names must come from config.GENRES (use e.g. `pop` for The Weeknd).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import config
from .infer import GenrePredictor


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def selective_stats(probs: np.ndarray, y_true: np.ndarray, threshold: float) -> dict:
    conf = probs.max(axis=1)
    accepted = conf >= threshold
    coverage = float(accepted.mean())
    acc = float((probs.argmax(axis=1)[accepted] == y_true[accepted]).mean()) if accepted.any() else float("nan")
    return {"threshold": float(threshold), "coverage": coverage, "accuracy_on_accepted": acc}


def plot_confusion(cm: np.ndarray, classes: list[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Test-set confusion matrix (row-normalised, track level)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_reliability(probs: np.ndarray, y_true: np.ndarray, path: Path, n_bins: int = 10) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    xs, ys = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() >= 3:
            xs.append(conf[m].mean())
            ys.append(correct[m].mean())
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfectly calibrated")
    ax.plot(xs, ys, "o-", label="model")
    ax.set_xlabel("Stated confidence")
    ax.set_ylabel("Actual accuracy")
    ax.set_title("Reliability diagram (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _ood_files(root: Path):
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in {".mp3", ".wav", ".flac", ".ogg", ".m4a"}:
            continue
        genre = p.parent.name if p.parent != root else p.name.split("__")[0]
        yield p, genre.lower()


def run_ood(ood_dir: Path, predictor: GenrePredictor | None = None) -> dict:
    import csv

    from .audio import AudioError, load_audio

    predictor = predictor or GenrePredictor()
    rows = []
    for path, expected in _ood_files(ood_dir):
        if expected not in config.GENRES:
            print(f"skip {path.name}: '{expected}' is not one of {config.GENRES}")
            continue
        try:
            pred = predictor.predict(load_audio(path))
        except AudioError as exc:
            print(f"skip {path.name}: {exc}")
            continue
        rows.append({"file": path.name, "expected": expected, "predicted": pred.label,
                     "confidence": round(pred.confidence, 3), "agreement": round(pred.agreement, 3),
                     "uncertain": pred.uncertain,
                     "top3": " | ".join(f"{g} {p:.0%}" for g, p in pred.top)})
        print(f"{path.name:40s} expected={expected:10s} -> {pred.label:10s} {pred.confidence:5.0%}"
              f"{'  [UNCERTAIN]' if pred.uncertain else ''}")
    if not rows:
        raise SystemExit("No usable files found.")
    config.REPORTS_DIR.mkdir(exist_ok=True)
    with open(config.REPORTS_DIR / "ood_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n = len(rows)
    correct = sum(r["expected"] == r["predicted"] for r in rows)
    confident_wrong = sum(r["expected"] != r["predicted"] and not r["uncertain"] for r in rows)
    summary = {"n": n, "accuracy": correct / n, "confidently_wrong": confident_wrong,
               "abstained": sum(r["uncertain"] for r in rows)}
    (config.REPORTS_DIR / "ood_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ood-dir", type=Path, required=True)
    run_ood(ap.parse_args().ood_dir)
