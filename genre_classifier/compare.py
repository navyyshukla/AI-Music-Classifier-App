"""Score a saved head on one data source's held-out test tracks, using cached embeddings.

    python -m genre_classifier.compare --source jamendo --out reports/jamendo_before.json
Run it once with the old head ("before") and again after retraining ("after").
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from . import config
from .extract import cache_path
from .infer import GenrePredictor


def run(source: str, model_path=config.MODEL_PATH) -> dict:
    idx = pd.read_csv(config.CACHE_DIR / "index.csv")
    rows = idx[(idx["split"] == "test") & (idx["source"] == source)]
    p = GenrePredictor(model_path)
    recs = []
    for r in rows.itertuples():
        f = cache_path(r.track_id)
        if not f.exists():
            continue
        pred = p.predict_embeddings(np.load(f))
        recs.append({"true": r.genre, "pred": pred.label, "conf": pred.confidence, "unsure": pred.uncertain,
                     "ood": pred.out_of_distribution})
    d = pd.DataFrame(recs)
    d["correct"] = d["true"] == d["pred"]
    answered = d[~d["unsure"]]
    per_class = {g: {"n": int(len(x)), "recall": round(float(x["correct"].mean()), 3)} for g, x in d.groupby("true")}
    return {
        "source": source, "n_tracks": int(len(d)),
        "accuracy_forced": round(float(d["correct"].mean()), 4),                 # always answer with the top genre
        "coverage": round(float(len(answered) / len(d)), 4),                      # share it was willing to answer
        "accuracy_when_answering": round(float(answered["correct"].mean()), 4) if len(answered) else None,
        "confidently_wrong": int(((~d["correct"]) & (~d["unsure"]) & (d["conf"] >= 0.8)).sum()),
        "rejected_as_not_music": int(d["ood"].sum()),
        "per_class": per_class,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="jamendo")
    ap.add_argument("--model", default=str(config.MODEL_PATH))
    ap.add_argument("--out")
    a = ap.parse_args()
    res = run(a.source, a.model)
    print(json.dumps(res, indent=2))
    if a.out:
        config.REPORTS_DIR.mkdir(exist_ok=True)
        open(a.out, "w").write(json.dumps(res, indent=2))
