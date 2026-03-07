from __future__ import annotations
import numpy as np
from voxis.audio import load_audio
from voxis.embedding import ECAPAEmbedder
from voxis.transform import tenant_seed, generate_orthogonal_matrix, protect_embedding
from voxis.similarity import cosine_similarity


class VoxISPipeline:
    def __init__(self, embedder: ECAPAEmbedder) -> None:
        self.embedder = embedder

    def build_template(self, audio_path: str, tenant_id: str) -> np.ndarray:
        waveform = load_audio(audio_path)
        f = self.embedder.extract(waveform)
        seed = tenant_seed(tenant_id)
        R = generate_orthogonal_matrix(dim=f.shape[0], seed=seed)
        z = protect_embedding(f, R)
        return z

    def verify_pair(self, audio_path_1: str, audio_path_2: str, tenant_id: str) -> float:
        z1 = self.build_template(audio_path_1, tenant_id)
        z2 = self.build_template(audio_path_2, tenant_id)
        return cosine_similarity(z1, z2)