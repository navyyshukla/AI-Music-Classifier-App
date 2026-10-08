"""Embed every track once and cache the window embeddings to disk.

    python -m genre_classifier.extract --per-class 600
Resumable: tracks that already have a cache file are skipped.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from tqdm import tqdm

from . import config
from .audio import AudioError, load_audio, split_windows
from .data import build_index
from .embeddings import Encoder


def cache_path(track_id: str):
    return config.CACHE_DIR / "embeddings" / f"{track_id}.npy"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=600, help="max FMA tracks per genre")
    ap.add_argument("--no-fma", action="store_true")
    ap.add_argument("--no-gtzan", action="store_true")
    ap.add_argument("--no-jamendo", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only process N tracks (smoke test)")
    args = ap.parse_args()

    df = build_index(args.per_class, use_gtzan=not args.no_gtzan, use_fma=not args.no_fma,
                     use_jamendo=not args.no_jamendo)
    if args.limit:
        df = df.sample(args.limit, random_state=0)
    (config.CACHE_DIR / "embeddings").mkdir(parents=True, exist_ok=True)
    df.to_csv(config.CACHE_DIR / "index.csv", index=False)

    enc = Encoder()
    failed, t0 = [], time.time()
    for row in tqdm(df.itertuples(), total=len(df), desc="embedding"):
        out = cache_path(row.track_id)
        if out.exists():
            continue
        try:
            y = load_audio(row.path)
            windows, _ = split_windows(y, max_windows=config.TRAIN_WINDOWS_PER_TRACK)
            np.save(out, enc.embed(windows))
        except (AudioError, RuntimeError, ValueError) as exc:
            failed.append((row.track_id, str(exc)))
    print(f"done in {(time.time() - t0) / 60:.1f} min, {len(failed)} tracks skipped")
    for tid, msg in failed[:20]:
        print("  skipped", tid, msg[:100])


if __name__ == "__main__":
    main()
