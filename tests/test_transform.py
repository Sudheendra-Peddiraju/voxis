import numpy as np

from voxis.similarity import cosine_similarity
from voxis.transform import generate_orthogonal_matrix


def test_orthogonal_transform_preserves_cosine() -> None:
    dim = 192
    rng = np.random.RandomState(42)

    x = rng.randn(dim).astype(np.float32)
    y = rng.randn(dim).astype(np.float32)

    R = generate_orthogonal_matrix(dim=dim, seed=123)

    cos_original = cosine_similarity(x, y)
    cos_transformed = cosine_similarity(R @ x, R @ y)

    assert abs(cos_original - cos_transformed) < 1e-4