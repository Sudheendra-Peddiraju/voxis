from __future__ import annotations
import hashlib
import numpy as np
from scipy.stats import ortho_group


def tenant_seed(tenant_id: str) -> int:
    digest = hashlib.sha256(tenant_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def generate_orthogonal_matrix(dim: int, seed: int) -> np.ndarray:
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        R = ortho_group.rvs(dim=dim).astype(np.float32)
    finally:
        np.random.set_state(state)
    return R


def protect_embedding(embedding: np.ndarray, R: np.ndarray) -> np.ndarray:
    if embedding.ndim != 1:
        raise ValueError("Embedding must be a 1D vector.")

    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError("R must be a square matrix.")

    if R.shape[1] != embedding.shape[0]:
        raise ValueError(
            f"Shape mismatch: R has shape {R.shape}, embedding has shape {embedding.shape}"
        )

    return (R @ embedding).astype(np.float32)