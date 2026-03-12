from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np
import torch

from voxis.audio import load_audio
from voxis.embedding import ECAPAEmbedder
from voxis.transform import (
    generate_orthogonal_matrix,
    protect_embedding,
    tenant_seed,
)


@dataclass
class EnrollmentResult:
    protected_template: np.ndarray
    num_segments_used: int
    segment_duration_sec: float


def _split_waveform_into_segments(
    waveform: torch.Tensor,
    sample_rate: int,
    segment_duration_sec: float = 3.0,
) -> List[torch.Tensor]:
    """
    Split waveform [1, T] into non-overlapping fixed-length segments.
    Segments shorter than the target length are discarded.
    """
    if waveform.ndim != 2 or waveform.shape[0] != 1:
        raise ValueError(f"Expected waveform shape [1, T], got {tuple(waveform.shape)}")

    segment_samples = int(sample_rate * segment_duration_sec)
    total_samples = waveform.shape[1]

    segments: List[torch.Tensor] = []
    start = 0

    while start + segment_samples <= total_samples:
        segment = waveform[:, start : start + segment_samples]
        segments.append(segment)
        start += segment_samples

    return segments


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("Cannot normalize a zero vector.")
    return (vec / norm).astype(np.float32)


class EnrollmentService:
    def __init__(
        self,
        embedder: ECAPAEmbedder,
        sample_rate: int = 16000,
        segment_duration_sec: float = 3.0,
    ) -> None:
        self.embedder = embedder
        self.sample_rate = sample_rate
        self.segment_duration_sec = segment_duration_sec

    def build_reference_embedding(self, audio_paths: List[str]) -> tuple[np.ndarray, int]:
        """
        Build a stable enrollment embedding by averaging segment embeddings
        across one or more enrollment audio files.
        """
        if not audio_paths:
            raise ValueError("audio_paths cannot be empty.")

        segment_embeddings: List[np.ndarray] = []

        for audio_path in audio_paths:
            waveform = load_audio(audio_path, target_sr=self.sample_rate)
            segments = _split_waveform_into_segments(
                waveform=waveform,
                sample_rate=self.sample_rate,
                segment_duration_sec=self.segment_duration_sec,
            )

            for segment in segments:
                emb = self.embedder.extract(segment)
                segment_embeddings.append(emb)

        if not segment_embeddings:
            raise ValueError(
                "No valid enrollment segments were found. "
                "Provide longer audio or reduce segment_duration_sec."
            )

        emb_matrix = np.stack(segment_embeddings, axis=0)   # [N, D]
        mean_embedding = np.mean(emb_matrix, axis=0).astype(np.float32)
        mean_embedding = _l2_normalize(mean_embedding)

        return mean_embedding, len(segment_embeddings)

    def enroll(self, audio_paths: List[str], tenant_id: str) -> EnrollmentResult:
        """
        Build the protected enrollment template z = Rf from enrollment audio.
        """
        reference_embedding, num_segments_used = self.build_reference_embedding(audio_paths)

        seed = tenant_seed(tenant_id)
        R = generate_orthogonal_matrix(dim=reference_embedding.shape[0], seed=seed)
        protected_template = protect_embedding(reference_embedding, R)

        return EnrollmentResult(
            protected_template=protected_template,
            num_segments_used=num_segments_used,
            segment_duration_sec=self.segment_duration_sec,
        )