"""Train the classifier head on cached embeddings.

    python -m genre_classifier.train
Selects C and the temperature on the validation split; the test split is evaluated once at the end.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

from . import config
from .data import check_no_leakage
from .evaluate import expected_calibration_error, plot_confusion, plot_reliability, selective_stats
from .extract import cache_path
from .infer import FullHead, full_logits, softmax

MIN_AGREEMENT = 0.34


def load_split(df: pd.DataFrame, split: str):
    """Window-level matrix plus the track index each window came from."""
    xs, ys, tracks = [], [], []
    rows = df[df["split"] == split].reset_index(drop=True)
    kept = []
    for i, r in rows.iterrows():
        p = cache_path(r["track_id"])
        if not p.exists():
            continue
        emb = np.load(p)
        xs.append(emb)
        ys += [config.GENRES.index(r["genre"])] * len(emb)
        tracks += [len(kept)] * len(emb)
        kept.append(r)
    return np.concatenate(xs), np.array(ys), np.array(tracks), pd.DataFrame(kept).reset_index(drop=True)


def track_logits(clf, scaler, X, tracks, n_tracks):
    logits = clf.decision_function(scaler.transform(X))
    return np.stack([logits[tracks == t].mean(axis=0) for t in range(n_tracks)])


def main() -> None:
    df = pd.read_csv(config.CACHE_DIR / "index.csv")
    check_no_leakage(df)
    Xtr, ytr, _, _ = load_split(df, "train")
    Xva, yva, tva, mva = load_split(df, "val")
    Xte, yte, tte, mte = load_split(df, "test")
    print(f"windows: train {len(Xtr)}, val {len(Xva)}, test {len(Xte)}; "
          f"tracks: val {len(mva)}, test {len(mte)}")

    scaler = StandardScaler().fit(Xtr)
    present = np.unique(ytr)
    yva_t = mva["genre"].map(config.GENRES.index).to_numpy()
    yte_t = mte["genre"].map(config.GENRES.index).to_numpy()

    best = None
    for C in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
        clf.fit(scaler.transform(Xtr), ytr)
        lv = full_logits(clf.classes_, track_logits(clf, scaler, Xva, tva, len(mva)), len(config.GENRES))
        acc = float((lv.argmax(1) == yva_t).mean())
        print(f"C={C:<6} val track accuracy {acc:.3f}")
        if best is None or acc > best[0]:
            best = (acc, C, clf, lv)
    _, C, clf, lv = best

    def nll(T):
        p = softmax(lv / T)
        return -np.log(p[np.arange(len(yva_t)), yva_t] + 1e-12).mean()

    T = float(minimize_scalar(nll, bounds=(0.3, 10), method="bounded").x)
    pv = softmax(lv / T)
    # Smallest confidence threshold that keeps >=90% accuracy on accepted validation tracks and >=50% coverage.
    tau = 0.5
    for cand in np.arange(0.30, 0.96, 0.01):
        s = selective_stats(pv, yva_t, cand)
        if s["coverage"] >= 0.5 and s["accuracy_on_accepted"] >= 0.9:
            tau = float(cand)
            break
    print(f"C={C}, temperature={T:.2f}, abstain below {tau:.2f}")

    # ---- final, single look at the test split ----
    lt = full_logits(clf.classes_, track_logits(clf, scaler, Xte, tte, len(mte)), len(config.GENRES))
    pt = softmax(lt / T)
    pred = pt.argmax(1)
    labels = list(range(len(config.GENRES)))
    cm = confusion_matrix(yte_t, pred, labels=labels)
    report = classification_report(yte_t, pred, labels=labels, target_names=config.GENRES,
                                   output_dict=True, zero_division=0)
    metrics = {
        "test_tracks": int(len(mte)),
        "accuracy": float((pred == yte_t).mean()),
        "macro_f1": float(f1_score(yte_t, pred, average="macro", labels=labels, zero_division=0)),
        "ece": expected_calibration_error(pt, yte_t),
        "selective": selective_stats(pt, yte_t, tau),
        "per_source_accuracy": {s: float((pred[mte["source"] == s] == yte_t[mte["source"] == s]).mean())
                                for s in mte["source"].unique()},
        "per_class": {g: {k: report[g][k] for k in ("precision", "recall", "f1-score", "support")}
                      for g in config.GENRES},
        "chance_accuracy": 1 / len(present),
        "train_genres": [config.GENRES[i] for i in present],
    }
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_confusion(cm, config.GENRES, config.REPORTS_DIR / "confusion_matrix.png")
    plot_reliability(pt, yte_t, config.REPORTS_DIR / "reliability.png")

    # Re-fit the head with all 13 genre slots so the app never needs to know which were present.
    bundle = {"classes": config.GENRES, "scaler": scaler, "clf": FullHead(clf, len(config.GENRES)),
              "temperature": T, "min_confidence": tau, "min_agreement": MIN_AGREEMENT,
              "metrics": {k: metrics[k] for k in ("accuracy", "macro_f1", "ece", "test_tracks", "selective")}}
    config.MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, config.MODEL_PATH, compress=3)
    print(json.dumps({k: metrics[k] for k in ("accuracy", "macro_f1", "ece", "selective", "per_source_accuracy")}, indent=2))
    print("saved", config.MODEL_PATH)


if __name__ == "__main__":
    main()
