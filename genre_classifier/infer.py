"""Whole-song inference: window -> embed -> classify -> aggregate -> (maybe) abstain."""
from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np

from . import config
from .audio import split_windows


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def full_logits(classes_idx: np.ndarray, raw: np.ndarray, n_total: int) -> np.ndarray:
    """Expand logits to every genre slot (genres absent from training get a very low logit)."""
    out = np.full((raw.shape[0], n_total), -30.0)
    out[:, classes_idx] = raw
    return out


class FullHead:
    """Wraps a fitted LogisticRegression so decision_function returns one column per genre.

    Lives here (not in train.py) so the pickled head can be loaded by the app.
    """

    def __init__(self, clf, n_total: int):
        self.clf, self.n_total = clf, n_total

    def decision_function(self, X):
        return full_logits(self.clf.classes_, self.clf.decision_function(X), self.n_total)


@dataclass
class Prediction:
    classes: list[str]
    probabilities: np.ndarray            # (n_classes,) calibrated, aggregated over windows
    window_probabilities: np.ndarray     # (n_windows, n_classes)
    window_times: list[float]
    label: str                           # argmax genre
    confidence: float                    # probability of `label`
    agreement: float                     # share of windows whose top genre equals `label`
    uncertain: bool
    reason: str = ""
    top: list[tuple[str, float]] = field(default_factory=list)


def aggregate(window_logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Mean of window logits (geometric pooling), temperature-scaled -> (n_classes,) probabilities."""
    return softmax(window_logits.mean(axis=0) / temperature)


def decide(
    probs: np.ndarray,
    window_probs: np.ndarray,
    classes: list[str],
    min_confidence: float,
    min_agreement: float,
) -> tuple[str, float, float, bool, str]:
    """Pick a label and decide whether the model should abstain."""
    top = int(np.argmax(probs))
    confidence = float(probs[top])
    agreement = float(np.mean(np.argmax(window_probs, axis=1) == top))
    reasons = []
    if confidence < min_confidence:
        reasons.append(f"top genre probability {confidence:.0%} is below {min_confidence:.0%}")
    if agreement < min_agreement:
        reasons.append(f"only {agreement:.0%} of the song's sections agree")
    return classes[top], confidence, agreement, bool(reasons), "; ".join(reasons)


class GenrePredictor:
    """Loads the trained head and (lazily) the encoder."""

    def __init__(self, model_path=config.MODEL_PATH, encoder=None):
        bundle = joblib.load(model_path)
        self.classes: list[str] = list(bundle["classes"])
        self.scaler = bundle["scaler"]
        self.clf = bundle["clf"]
        self.temperature: float = float(bundle["temperature"])
        self.min_confidence: float = float(bundle["min_confidence"])
        self.min_agreement: float = float(bundle["min_agreement"])
        self.metrics: dict = bundle.get("metrics", {})
        self._encoder = encoder

    @property
    def encoder(self):
        if self._encoder is None:
            from .embeddings import Encoder

            self._encoder = Encoder()
        return self._encoder

    def predict_embeddings(self, emb: np.ndarray, window_times: list[float] | None = None) -> Prediction:
        logits = self.clf.decision_function(self.scaler.transform(emb))
        probs = aggregate(logits, self.temperature)
        window_probs = softmax(logits / self.temperature)
        label, conf, agree, uncertain, reason = decide(
            probs, window_probs, self.classes, self.min_confidence, self.min_agreement
        )
        order = np.argsort(probs)[::-1][:3]
        return Prediction(
            classes=self.classes,
            probabilities=probs,
            window_probabilities=window_probs,
            window_times=list(window_times) if window_times is not None else list(range(len(emb))),
            label=label,
            confidence=conf,
            agreement=agree,
            uncertain=uncertain,
            reason=reason,
            top=[(self.classes[i], float(probs[i])) for i in order],
        )

    def predict(self, y: np.ndarray, sr: int = config.SAMPLE_RATE, progress=None) -> Prediction:
        windows, times = split_windows(y, sr)
        if progress:
            progress(f"Embedding {len(windows)} sections of the song…")
        emb = self.encoder.embed(windows)
        return self.predict_embeddings(emb, times)
