"""Audio loading and windowing. Pure functions, no model code."""
from __future__ import annotations

import numpy as np

from . import config


class AudioError(ValueError):
    """Raised for audio that cannot be classified (too short, silent, undecodable)."""


def load_audio(path, sr: int = config.SAMPLE_RATE, max_seconds: float = config.MAX_LOAD_SECONDS) -> np.ndarray:
    """Decode any supported file to mono float32 at `sr`."""
    import librosa

    try:
        y, _ = librosa.load(str(path), sr=sr, mono=True, duration=max_seconds)
    except Exception as exc:  # decoder errors come in many types
        raise AudioError(f"Could not decode audio: {exc}") from exc
    return np.asarray(y, dtype=np.float32)


def split_windows(
    y: np.ndarray,
    sr: int = config.SAMPLE_RATE,
    window_seconds: float = config.WINDOW_SECONDS,
    max_windows: int = config.MAX_WINDOWS,
    silence_rms: float = config.SILENCE_RMS,
) -> tuple[list[np.ndarray], list[float]]:
    """Cut the *whole* signal into up to `max_windows` evenly spread windows.

    Returns (windows, start_times_in_seconds). Near-silent windows are dropped.
    Clips shorter than one window are zero-padded into a single window.
    """
    if y.ndim != 1:
        raise AudioError("expected mono audio")
    if len(y) < config.MIN_AUDIO_SECONDS * sr:
        raise AudioError(f"Audio is shorter than {config.MIN_AUDIO_SECONDS:.0f} seconds.")

    size = int(window_seconds * sr)
    if len(y) <= size:
        padded = np.pad(y, (0, size - len(y)))
        starts = [0]
        chunks = [padded]
    else:
        n = min(max_windows, max(1, int(np.ceil(len(y) / size))))
        starts = np.unique(np.linspace(0, len(y) - size, n).astype(int)).tolist()
        chunks = [y[s : s + size] for s in starts]

    kept, times = [], []
    for s, c in zip(starts, chunks):
        if float(np.sqrt(np.mean(np.square(c)))) >= silence_rms:
            kept.append(c)
            times.append(s / sr)
    if not kept:
        raise AudioError("The audio is silent or too quiet to analyse.")
    return kept, times
