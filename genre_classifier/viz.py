"""Display helpers for the app: cheap, downsampled views of the audio."""
from __future__ import annotations

import numpy as np


def waveform_envelope(y: np.ndarray, sr: int, n_points: int = 1500) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Min/max envelope with at most `n_points` buckets: (times, lo, hi)."""
    n = max(1, min(n_points, len(y)))
    edges = np.linspace(0, len(y), n + 1).astype(int)
    lo = np.array([y[a:b].min() if b > a else 0.0 for a, b in zip(edges[:-1], edges[1:])])
    hi = np.array([y[a:b].max() if b > a else 0.0 for a, b in zip(edges[:-1], edges[1:])])
    times = (edges[:-1] + edges[1:]) / 2 / sr
    return times, lo, hi


def mel_spectrogram_db(y: np.ndarray, sr: int, n_mels: int = 96, max_frames: int = 1200):
    """Log-mel spectrogram (dB) with its time and mel-frequency axes, downsampled for plotting."""
    import librosa

    hop = 512
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=hop, n_mels=n_mels, fmax=sr / 2)
    S_db = librosa.power_to_db(S, ref=np.max, top_db=80.0)
    times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr, hop_length=hop)
    freqs = librosa.mel_frequencies(n_mels=n_mels, fmax=sr / 2)
    if S_db.shape[1] > max_frames:
        idx = np.linspace(0, S_db.shape[1] - 1, max_frames).astype(int)
        S_db, times = S_db[:, idx], times[idx]
    return times, freqs, S_db
