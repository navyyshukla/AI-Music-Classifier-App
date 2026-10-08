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


class OODDetector:
    """Flags audio that does not resemble the training music (noise, tones, speech, odd recordings).

    Mahalanobis distance, in a PCA space of the scaled embeddings, to the nearest genre mean with one
    shared covariance. A track's score is the median over its windows; the threshold is the 99th
    percentile of validation tracks, so about 1% of in-distribution tracks are rejected.
    """

    N_COMPONENTS = 64
    SHRINKAGE = 0.1

    def __init__(self, pca, means: np.ndarray, precision: np.ndarray, threshold: float = np.inf):
        self.pca, self.means, self.precision, self.threshold = pca, means, precision, float(threshold)

    @classmethod
    def fit(cls, X_scaled: np.ndarray, y: np.ndarray) -> "OODDetector":
        from sklearn.decomposition import PCA

        pca = PCA(n_components=cls.N_COMPONENTS, random_state=0).fit(X_scaled)
        Z = pca.transform(X_scaled)
        labels = np.unique(y)
        means = np.stack([Z[y == c].mean(axis=0) for c in labels])
        centred = Z - means[np.searchsorted(labels, y)]
        cov = np.cov(centred, rowvar=False)
        cov = (1 - cls.SHRINKAGE) * cov + cls.SHRINKAGE * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
        return cls(pca, means, np.linalg.inv(cov))

    def window_scores(self, X_scaled: np.ndarray) -> np.ndarray:
        d = self.pca.transform(X_scaled)[:, None, :] - self.means[None]
        return np.einsum("nkd,de,nke->nk", d, self.precision, d).min(axis=1)

    def track_score(self, X_scaled: np.ndarray) -> float:
        return float(np.median(self.window_scores(X_scaled)))

    def is_ood(self, score: float) -> bool:
        return score > self.threshold


class MusicGate:
    """AudioSet "Music" score from the encoder's own pretrained classifier head (numpy, no torch).

    The AST checkpoint ships a LayerNorm + Linear head over the same 768-d pooled embedding we cache.
    Only the "Music" row is kept. Noise, tones, clicks and speech score near zero; real tracks do not.
    A track's score is the median over its windows; the threshold is the 1st percentile of validation tracks.
    """

    def __init__(self, ln_weight, ln_bias, ln_eps, w, b, threshold: float = 0.0):
        self.ln_weight, self.ln_bias, self.ln_eps = np.asarray(ln_weight), np.asarray(ln_bias), float(ln_eps)
        self.w, self.b, self.threshold = np.asarray(w), float(b), float(threshold)

    @classmethod
    def from_checkpoint(cls, name: str) -> "MusicGate":
        import json

        from huggingface_hub import hf_hub_download

        cfg = json.load(open(hf_hub_download(name, "config.json")))
        idx = next(int(k) for k, v in cfg["id2label"].items() if v == "Music")
        try:
            from safetensors.torch import load_file

            sd = load_file(hf_hub_download(name, "model.safetensors"))
        except Exception:
            import torch

            sd = torch.load(hf_hub_download(name, "pytorch_model.bin"), map_location="cpu")
        return cls(sd["classifier.layernorm.weight"].numpy(), sd["classifier.layernorm.bias"].numpy(),
                   cfg.get("layer_norm_eps", 1e-12),
                   sd["classifier.dense.weight"][idx].numpy(), sd["classifier.dense.bias"][idx].item())

    def window_probs(self, emb: np.ndarray) -> np.ndarray:
        mu = emb.mean(axis=1, keepdims=True)
        var = emb.var(axis=1, keepdims=True)
        z = (emb - mu) / np.sqrt(var + self.ln_eps) * self.ln_weight + self.ln_bias
        return 1.0 / (1.0 + np.exp(-(z @ self.w + self.b)))

    def track_prob(self, emb: np.ndarray) -> float:
        return float(np.median(self.window_probs(emb)))

    def is_not_music(self, prob: float) -> bool:
        return prob < self.threshold


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
    out_of_distribution: bool = False
    ood_score: float = 0.0
    music_prob: float = 1.0
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
        self.ood: OODDetector | None = bundle.get("ood")
        self.gate: MusicGate | None = bundle.get("gate")
        self._encoder = encoder

    @property
    def encoder(self):
        if self._encoder is None:
            from .embeddings import Encoder

            self._encoder = Encoder()
        return self._encoder

    def predict_embeddings(self, emb: np.ndarray, window_times: list[float] | None = None) -> Prediction:
        scaled = self.scaler.transform(emb)
        logits = self.clf.decision_function(scaled)
        probs = aggregate(logits, self.temperature)
        window_probs = softmax(logits / self.temperature)
        label, conf, agree, uncertain, reason = decide(
            probs, window_probs, self.classes, self.min_confidence, self.min_agreement
        )
        ood_score, music_prob, is_ood, why = 0.0, 1.0, False, []
        if self.gate is not None:
            music_prob = self.gate.track_prob(emb)
            if self.gate.is_not_music(music_prob):
                why.append(f"it does not sound like music (music score {music_prob:.0%})")
        if self.ood is not None:
            ood_score = self.ood.track_score(scaled)
            if self.ood.is_ood(ood_score):
                why.append("it does not resemble the music this model was trained on")
        if why:
            is_ood, uncertain, reason = True, True, "; ".join(why)
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
            out_of_distribution=is_ood,
            ood_score=ood_score,
            music_prob=music_prob,
            top=[(self.classes[i], float(probs[i])) for i in order],
        )

    def predict(self, y: np.ndarray, sr: int = config.SAMPLE_RATE, progress=None) -> Prediction:
        windows, times = split_windows(y, sr)
        if progress:
            progress(f"Embedding {len(windows)} sections of the song…")
        emb = self.encoder.embed(windows)
        return self.predict_embeddings(emb, times)
