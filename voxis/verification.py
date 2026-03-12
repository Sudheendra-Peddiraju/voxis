from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from voxis.audio import load_audio
from voxis.embedding import ECAPAEmbedder
from voxis.similarity import cosine_similarity
from voxis.storage import TemplateStore
from voxis.transform import (
    generate_orthogonal_matrix,
    protect_embedding,
    tenant_seed,
)


@dataclass
class VerificationResult:
    user_id: str
    tenant_id: str
    score: float
    threshold: float
    verified: bool
    embedding_dim: int


class VerificationService:
    def __init__(
        self,
        embedder: ECAPAEmbedder,
        template_store: TemplateStore,
        sample_rate: int = 16000,
        threshold: float = 0.65,
    ) -> None:
        self.embedder = embedder
        self.template_store = template_store
        self.sample_rate = sample_rate
        self.threshold = threshold

    def build_probe_template(self, probe_audio_path: str, tenant_id: str) -> np.ndarray:
        waveform = load_audio(probe_audio_path, target_sr=self.sample_rate)
        probe_embedding = self.embedder.extract(waveform)

        seed = tenant_seed(tenant_id)
        R = generate_orthogonal_matrix(dim=probe_embedding.shape[0], seed=seed)
        protected_probe = protect_embedding(probe_embedding, R)

        return protected_probe

    def verify(self, user_id: str, tenant_id: str, probe_audio_path: str) -> VerificationResult:
        stored = self.template_store.get_template(user_id=user_id, tenant_id=tenant_id)
        if stored is None:
            raise ValueError(f"No enrolled template found for user_id={user_id}, tenant_id={tenant_id}")

        protected_probe = self.build_probe_template(
            probe_audio_path=probe_audio_path,
            tenant_id=tenant_id,
        )

        score = cosine_similarity(protected_probe, stored.protected_template)
        verified = score >= self.threshold

        return VerificationResult(
            user_id=user_id,
            tenant_id=tenant_id,
            score=score,
            threshold=self.threshold,
            verified=verified,
            embedding_dim=stored.embedding_dim,
        )