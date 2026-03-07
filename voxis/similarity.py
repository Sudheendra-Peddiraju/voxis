from __future__ import annotations
import numpy as np


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y)

    if x_norm == 0 or y_norm == 0:
        raise ValueError("Zero-norm vector encountered.")

    return float(np.dot(x, y) / (x_norm * y_norm))