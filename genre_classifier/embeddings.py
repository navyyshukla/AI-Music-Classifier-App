"""Pretrained audio encoder (Audio Spectrogram Transformer, AudioSet)."""
from __future__ import annotations

import numpy as np

from . import config


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Encoder:
    """Turns 16 kHz mono windows into 768-d embeddings. Loads weights lazily."""

    def __init__(self, device: str | None = None, batch_size: int = 4):
        self.device = device or pick_device()
        self.batch_size = batch_size
        self._model = None
        self._extractor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import ASTFeatureExtractor, ASTModel

        torch.set_grad_enabled(False)
        self._extractor = ASTFeatureExtractor.from_pretrained(config.ENCODER_NAME)
        self._model = ASTModel.from_pretrained(config.ENCODER_NAME).eval().to(self.device)

    def embed(self, windows: list[np.ndarray]) -> np.ndarray:
        """Return an (n_windows, 768) float32 array."""
        import torch

        self._load()
        out = []
        with torch.inference_mode():
            for i in range(0, len(windows), self.batch_size):
                batch = windows[i : i + self.batch_size]
                inputs = self._extractor(batch, sampling_rate=config.SAMPLE_RATE, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                out.append(self._model(**inputs).pooler_output.float().cpu().numpy())
        return np.concatenate(out, axis=0)
