"""Dataset indexes with track-level (and, for FMA, artist-level) splits.

Expected layout under Data/:
  Data/genres_original/<genre>/<genre>.NNNNN.wav          (GTZAN)
  Data/fma_metadata/tracks.csv                            (FMA metadata)
  Data/fma_medium/<NNN>/<NNNNNN>.mp3                      (FMA-medium audio)
  Data/jamendo/...                                        (optional, see jamendo.py)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config

GTZAN_LABELS = {g: g for g in
                ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]}

# FMA top-level genres -> unified labels. Anything not listed is dropped: "Experimental",
# "Instrumental", "International", "Easy Listening", "Old-Time / Historic" and "Spoken"
# are not genres in the same sense and would blur the classes.
FMA_LABELS = {
    "Blues": "blues", "Classical": "classical", "Country": "country", "Electronic": "electronic",
    "Folk": "folk", "Hip-Hop": "hiphop", "Jazz": "jazz", "Pop": "pop", "Rock": "rock",
    "Soul-RnB": "soul_rnb",
}


def gtzan_index(root: Path = config.DATA_DIR / "genres_original", seed: int = 42) -> pd.DataFrame:
    """One row per track, split 80/10/10 stratified by genre. Splitting is per track, never per clip."""
    rows = []
    for genre_dir in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        label = GTZAN_LABELS.get(genre_dir.name)
        if label is None:
            continue
        for wav in sorted(genre_dir.glob("*.wav")):
            rows.append({"track_id": f"gtzan_{wav.stem}", "path": str(wav), "genre": label,
                         "source": "gtzan", "artist": f"gtzan_{wav.stem}"})
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f"No GTZAN wav files found under {root}")
    rng = np.random.default_rng(seed)
    df["split"] = "train"
    for _, idx in df.groupby("genre").groups.items():
        idx = rng.permutation(np.asarray(idx))
        n = len(idx)
        df.loc[idx[: int(0.1 * n)], "split"] = "test"
        df.loc[idx[int(0.1 * n) : int(0.2 * n)], "split"] = "val"
    return df


def fma_index(root: Path = config.DATA_DIR, per_class: int = 600, seed: int = 42) -> pd.DataFrame:
    """FMA-medium rows using FMA's official split, capped at about `per_class` tracks per genre."""
    tracks = pd.read_csv(Path(root) / "fma_metadata" / "tracks.csv", index_col=0, header=[0, 1])
    t = pd.DataFrame({
        "genre_top": tracks[("track", "genre_top")],
        "split": tracks[("set", "split")],
        "subset": tracks[("set", "subset")],
        "artist": tracks[("artist", "id")],
    })
    t = t[t["subset"].isin(["small", "medium"]) & t["genre_top"].isin(FMA_LABELS)].copy()
    t["split"] = t["split"].replace({"training": "train", "validation": "val"})
    t["genre"] = t["genre_top"].map(FMA_LABELS)
    t["path"] = [str(Path(root) / "fma_medium" / f"{i:06d}"[:3] / f"{i:06d}.mp3") for i in t.index]
    t["track_id"] = [f"fma_{i}" for i in t.index]
    t["artist"] = "fma_artist_" + t["artist"].astype(str)
    t["source"] = "fma"
    t = t[t["path"].map(lambda p: Path(p).exists())]
    # Balance: cap each genre, sampling inside each (genre, split) so split proportions are kept.
    parts = []
    for (genre, split), g in t.groupby(["genre", "split"]):
        n = min(len(g), max(1, int(per_class * _SPLIT_SHARE[split])))
        parts.append(g.sample(n, random_state=seed))
    t = pd.concat(parts)
    return t[["track_id", "path", "genre", "split", "source", "artist"]].reset_index(drop=True)


_SPLIT_SHARE = {"train": 0.8, "val": 0.1, "test": 0.1}


def jamendo_index() -> pd.DataFrame:
    """MTG-Jamendo tracks that have been unpacked under Data/jamendo/audio (see jamendo.py)."""
    from . import jamendo

    return pd.DataFrame(jamendo.index_rows())


def build_index(per_class: int = 600, use_gtzan: bool = True, use_fma: bool = True,
                use_jamendo: bool = True) -> pd.DataFrame:
    parts = []
    if use_gtzan:
        parts.append(gtzan_index())
    if use_fma:
        parts.append(fma_index(per_class=per_class))
    if use_jamendo and (config.DATA_DIR / "jamendo" / "audio").exists():
        jam = jamendo_index()
        if len(jam):
            parts.append(jam)
    if not parts:
        raise ValueError("Enable at least one dataset")
    df = pd.concat(parts, ignore_index=True)
    check_no_leakage(df)
    return df


def check_no_leakage(df: pd.DataFrame) -> None:
    """Fail loudly if a track or artist appears in more than one split."""
    assert df["track_id"].is_unique, "duplicate track ids"
    per_artist = df.groupby("artist")["split"].nunique()
    leaking = per_artist[per_artist > 1]
    if len(leaking):
        raise AssertionError(f"{len(leaking)} artists appear in more than one split, e.g. {list(leaking.index[:3])}")
    unknown = set(df["genre"]) - set(config.GENRES)
    if unknown:
        raise AssertionError(f"labels outside config.GENRES: {unknown}")
