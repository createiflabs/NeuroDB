import numpy as np

from neurodb.metrics import compute_scores


def test_cosine_identical_is_one():
    matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    query = np.array([1, 0, 0], dtype=np.float32)
    scores = compute_scores(matrix, query, "cosine")
    assert np.isclose(scores[0], 1.0)
    assert np.isclose(scores[1], 0.0)


def test_cosine_is_scale_invariant():
    matrix = np.array([[2, 0, 0]], dtype=np.float32)
    query = np.array([5, 0, 0], dtype=np.float32)
    assert np.isclose(compute_scores(matrix, query, "cosine")[0], 1.0)


def test_dot_product():
    matrix = np.array([[1, 2, 3]], dtype=np.float32)
    query = np.array([1, 1, 1], dtype=np.float32)
    assert np.isclose(compute_scores(matrix, query, "dot")[0], 6.0)


def test_euclidean_is_negative_distance():
    matrix = np.array([[0, 0], [3, 4]], dtype=np.float32)
    query = np.array([0, 0], dtype=np.float32)
    scores = compute_scores(matrix, query, "euclidean")
    assert np.isclose(scores[0], 0.0)
    assert np.isclose(scores[1], -5.0)


def test_empty_matrix_returns_empty():
    matrix = np.zeros((0, 4), dtype=np.float32)
    query = np.zeros(4, dtype=np.float32)
    assert compute_scores(matrix, query, "cosine").shape == (0,)


def test_unknown_metric_raises():
    import pytest

    with pytest.raises(ValueError):
        compute_scores(np.ones((1, 2), dtype=np.float32), np.ones(2, dtype=np.float32), "nope")
