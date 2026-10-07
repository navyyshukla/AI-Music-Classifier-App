"""Shared constants. Everything that must match between training and inference lives here."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.environ.get("GENRE_MODEL_PATH", ROOT / "models" / "genre_head.joblib"))
REPORTS_DIR = ROOT / "reports"
DATA_DIR = ROOT / "Data"
CACHE_DIR = DATA_DIR / "cache"

ENCODER_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
SAMPLE_RATE = 16_000          # AST expects 16 kHz mono
WINDOW_SECONDS = 10.0         # AST's native context is ~10.24 s
MAX_WINDOWS = 18              # windows are spread evenly over the whole song
MAX_LOAD_SECONDS = 600        # never decode more than 10 minutes
MIN_AUDIO_SECONDS = 3.0
SILENCE_RMS = 0.005           # windows quieter than this (~-46 dBFS) are skipped
TRAIN_WINDOWS_PER_TRACK = 3   # 30 s clips -> 3 non-overlapping windows

# Unified label set. GTZAN and FMA label names are mapped onto these in data.py.
GENRES = [
    "blues", "classical", "country", "disco", "electronic", "folk",
    "hiphop", "jazz", "metal", "pop", "reggae", "rock", "soul_rnb",
]
