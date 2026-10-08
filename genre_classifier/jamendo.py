"""MTG-Jamendo (Creative Commons, mostly 2010s-2020s independent releases) as extra training data.

Layout under Data/jamendo/:
  autotagging_genre.tsv                      metadata from github.com/MTG/mtg-jamendo-dataset
  tars/raw_30s_audio-low-NN.tar              downloaded archives (tar NN holds track ids with id % 100 == NN)
  audio/NN/<id>.mp3                          the tracks we use, unpacked from the tars

    python -m genre_classifier.jamendo prepare     # list selected tracks, unpack only those from the tars
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
import tarfile
from collections import Counter
from pathlib import Path

from . import config

ROOT = config.DATA_DIR / "jamendo"

# Jamendo genre tag -> unified label.
TAG_MAP = {
    "rnb": "soul_rnb", "soul": "soul_rnb",
    "disco": "disco", "country": "country", "reggae": "reggae",
    "pop": "pop", "electropop": "pop",
    "hiphop": "hiphop", "rap": "hiphop",
    "electronic": "electronic", "dance": "electronic", "house": "electronic",
    "techno": "electronic", "trance": "electronic",
    "rock": "rock", "hardrock": "rock", "punkrock": "rock", "grunge": "rock",
    "metal": "metal", "folk": "folk",
    "jazz": "jazz", "jazzfusion": "jazz",
    "blues": "blues", "bluesrock": "blues",
    "classical": "classical",
}


def read_metadata(path: Path = ROOT / "autotagging_genre.tsv") -> list[dict]:
    """One dict per track. Tags are spread over any number of trailing tab-separated columns."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)                                   # header
        for r in reader:
            rows.append({"id": int(r[0].split("_")[1]), "artist": r[1], "path": r[3],
                         "tags": [t.replace("genre---", "") for t in r[5:] if t]})
    return rows


# Tracks often carry several genres (R&B tracks are nearly always also tagged pop). When more than one
# unified label applies, the more specific one wins: pop + R&B -> soul_rnb, pop + electronic -> pop.
PRIORITY = ["soul_rnb", "hiphop", "disco", "reggae", "country", "blues", "jazz", "metal", "folk",
            "classical", "rock", "pop", "electronic"]
MAX_LABELS = 2   # tracks tagged with 3+ different unified genres are too ambiguous to use


def unique_label(tags: list[str]) -> str | None:
    labels = {TAG_MAP[t] for t in tags if t in TAG_MAP}
    if not labels or len(labels) > MAX_LABELS:
        return None
    return min(labels, key=PRIORITY.index)


def artist_split(artist: str) -> str:
    """Stable 80/10/10 split by artist, so one artist's tracks never straddle splits."""
    h = int(hashlib.sha1(artist.encode()).hexdigest(), 16) % 100
    return "test" if h < 10 else "val" if h < 20 else "train"


def low_name(path: str) -> str:
    """Metadata says NN/ID.mp3; the low-bitrate archives store NN/ID.low.mp3."""
    return path[:-4] + ".low.mp3" if path.endswith(".mp3") else path


def select_tracks(max_tar: int = 17, per_class: int = 600, seed: int = 42) -> list[dict]:
    """Single-label tracks from tars 00..max_tar, capped per genre (and per artist, to avoid one band dominating)."""
    import random

    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = {}
    for r in read_metadata():
        folder = int(r["path"].split("/")[0])
        label = unique_label(r["tags"])
        if label is None or folder > max_tar:
            continue
        by_label.setdefault(label, []).append({**r, "genre": label, "split": artist_split(r["artist"])})
    out = []
    for label, items in by_label.items():
        rng.shuffle(items)
        per_artist: Counter = Counter()
        kept = []
        for it in items:
            if per_artist[it["artist"]] >= 6:      # at most 6 tracks per artist per genre
                continue
            per_artist[it["artist"]] += 1
            kept.append(it)
            if len(kept) >= per_class:
                break
        out += kept
    return out


def prepare(max_tar: int, per_class: int) -> None:
    sel = select_tracks(max_tar, per_class)
    print(f"{len(sel)} tracks selected")
    print(Counter(s["genre"] for s in sel).most_common())
    wanted = {low_name(s["path"]) for s in sel}
    audio = ROOT / "audio"
    audio.mkdir(exist_ok=True)
    for tar_path in sorted((ROOT / "tars").glob("raw_30s_audio-low-*.tar")):
        nn = tar_path.stem.rsplit("-", 1)[1]
        need = [p for p in wanted if p.split("/")[0].zfill(2) == nn and not (audio / p).exists()]
        if not need:
            continue
        need_set, got, total = set(need), 0, 0
        try:
            with tarfile.open(tar_path) as tar:
                for m in tar:                      # stream; a truncated download still yields its readable prefix
                    total += 1
                    name = m.name.split("/", 1)[-1] if m.name.count("/") > 1 else m.name
                    if m.isfile() and (m.name in need_set or name in need_set):
                        tar.extract(m, audio)
                        got += 1
        except (tarfile.TarError, EOFError) as exc:
            print(f"{tar_path.name}: truncated after {total} entries ({exc}); kept what was readable")
        print(f"{tar_path.name}: unpacked {got} of {len(need)} wanted")


def index_rows(max_tar: int = 17, per_class: int = 600) -> list[dict]:
    """Rows for data.build_index: only tracks whose audio file is on disk."""
    rows = []
    for s in select_tracks(max_tar, per_class):
        p = ROOT / "audio" / low_name(s["path"])
        if p.exists():
            rows.append({"track_id": f"jamendo_{s['id']}", "path": str(p), "genre": s["genre"], "split": s["split"],
                         "source": "jamendo", "artist": f"jamendo_{s['artist']}"})
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["prepare", "stats"])
    ap.add_argument("--max-tar", type=int, default=17)
    ap.add_argument("--per-class", type=int, default=600)
    a = ap.parse_args()
    if a.action == "stats":
        sel = select_tracks(a.max_tar, a.per_class)
        print(len(sel), "tracks;", Counter(s["split"] for s in sel))
        for g, n in Counter(s["genre"] for s in sel).most_common():
            print(f"  {g:11s}{n}")
    else:
        prepare(a.max_tar, a.per_class)
