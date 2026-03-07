from __future__ import annotations
import numpy as np
import torch
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy
from voxis.config import VoxISConfig


class ECAPAEmbedder:
    def __init__(self, config: VoxISConfig) -> None:
        self.config = config
        self.device = config.device

        self.model = EncoderClassifier.from_hparams(
            source=config.ecapa_source,
            savedir=config.ecapa_savedir,
            run_opts={"device": self.device},
            local_strategy=LocalStrategy.COPY,
        )

    def extract(self, waveform: torch.Tensor) -> np.ndarray:
        if waveform.ndim != 2:
            raise ValueError(
                f"Expected waveform shape [1, T], but got {tuple(waveform.shape)}"
            )

        with torch.no_grad():
            embedding = self.model.encode_batch(waveform.to(self.device))

        embedding = embedding.squeeze().detach().cpu().numpy().astype(np.float32)
        return embedding