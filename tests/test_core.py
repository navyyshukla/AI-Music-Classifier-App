import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from genre_classifier import config
from genre_classifier.audio import AudioError, split_windows
from genre_classifier.data import FMA_LABELS, GTZAN_LABELS, check_no_leakage
from genre_classifier.infer import FullHead, GenrePredictor, aggregate, decide, softmax
from genre_classifier.viz import mel_spectrogram_db, waveform_envelope

SR = config.SAMPLE_RATE


def tone(seconds, freq=440.0, amp=0.3):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ---- windowing -------------------------------------------------------------------------
def test_windows_cover_whole_song_not_just_the_intro():
    y = tone(120)
    windows, times = split_windows(y)
    assert len(windows) == config.MAX_WINDOWS or len(windows) == 12
    assert all(len(w) == int(config.WINDOW_SECONDS * SR) for w in windows)
    assert times[0] == 0 and times[-1] > 100          # last window is near the end


def test_short_clip_is_padded_to_one_window():
    windows, times = split_windows(tone(5))
    assert len(windows) == 1 and len(windows[0]) == int(config.WINDOW_SECONDS * SR)


def test_too_short_and_silent_audio_rejected():
    with pytest.raises(AudioError):
        split_windows(tone(1))
    with pytest.raises(AudioError):
        split_windows(np.zeros(30 * SR, dtype=np.float32))


def test_silent_sections_are_dropped():
    y = np.concatenate([np.zeros(30 * SR, dtype=np.float32), tone(30)])
    windows, times = split_windows(y)
    assert 0 < len(windows) and min(times) > 15


# ---- aggregation and abstain logic -----------------------------------------------------
def test_aggregate_is_a_distribution():
    p = aggregate(np.random.default_rng(0).normal(size=(5, 4)), temperature=2.0)
    assert p.shape == (4,) and np.isclose(p.sum(), 1)


def test_decide_abstains_on_low_confidence_or_disagreement():
    classes = ["a", "b", "c"]
    confident = np.array([0.9, 0.05, 0.05])
    agree = np.tile(confident, (4, 1))
    assert decide(confident, agree, classes, 0.5, 0.34)[3] is False
    flat = np.array([0.4, 0.35, 0.25])
    assert decide(flat, np.tile(flat, (4, 1)), classes, 0.5, 0.34)[3] is True
    split = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9], [0.05, 0.9, 0.05]])
    label, conf, agreement, unc, why = decide(np.array([0.6, 0.3, 0.1]), split, classes, 0.5, 0.5)
    assert unc and "agree" in why


# ---- label map and leakage -------------------------------------------------------------
def test_every_label_maps_into_config_genres():
    assert set(GTZAN_LABELS.values()) <= set(config.GENRES)
    assert set(FMA_LABELS.values()) <= set(config.GENRES)


def test_leakage_check_catches_artist_in_two_splits():
    df = pd.DataFrame({"track_id": ["a", "b"], "artist": ["x", "x"], "split": ["train", "test"],
                       "genre": ["pop", "pop"]})
    with pytest.raises(AssertionError):
        check_no_leakage(df)
    df["artist"] = ["x", "y"]
    check_no_leakage(df)


# ---- end to end with a fake encoder ----------------------------------------------------
class FakeEncoder:
    """Maps a window's dominant frequency to a stable 768-d embedding."""

    def embed(self, windows):
        out = []
        for w in windows:
            f = np.argmax(np.abs(np.fft.rfft(w))) * SR / len(w)
            rng = np.random.default_rng(int(f // 100))
            out.append(rng.normal(size=768) + 0.01 * w[:768].std())
        return np.stack(out)


def test_predictor_roundtrip_through_joblib(tmp_path):
    rng = np.random.default_rng(0)
    n = len(config.GENRES)
    X = rng.normal(size=(200, 768))
    y = np.repeat(np.arange(n), 200 // n + 1)[:200]
    X += np.eye(n, 768)[y] * 4
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=500).fit(scaler.transform(X), y)
    path = tmp_path / "head.joblib"
    joblib.dump({"classes": config.GENRES, "scaler": scaler, "clf": FullHead(clf, n),
                 "temperature": 1.5, "min_confidence": 0.3, "min_agreement": 0.3}, path)
    pred = GenrePredictor(path, encoder=FakeEncoder()).predict(tone(60))
    assert pred.label in config.GENRES
    assert np.isclose(pred.probabilities.sum(), 1)
    assert pred.window_probabilities.shape[1] == n
    assert len(pred.top) == 3 and pred.top[0][1] >= pred.top[1][1]


# ---- display helpers -------------------------------------------------------------------
def test_waveform_envelope_is_small_and_bounded():
    t, lo, hi = waveform_envelope(tone(60), SR, 1500)
    assert len(t) == 1500 and (hi >= lo).all() and hi.max() <= 0.31


def test_spectrogram_shapes():
    t, f, S = mel_spectrogram_db(tone(10), SR)
    assert S.shape == (len(f), len(t)) and S.max() <= 0.0 + 1e-6
