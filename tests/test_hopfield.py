import numpy as np

from neurodb.hopfield import attention_weights, retrieve, softmax


def test_softmax_sums_to_one_and_orders():
    p = softmax(np.array([1.0, 2.0, 3.0]))
    assert np.isclose(p.sum(), 1.0)
    assert p[2] > p[1] > p[0]


def test_high_beta_recovers_nearest_pattern():
    X = np.eye(3, dtype=np.float32)
    query = np.array([0.9, 0.1, 0.0], dtype=np.float32)
    recon, weights = retrieve(X, query, beta=50.0)
    assert int(np.argmax(weights)) == 0
    np.testing.assert_allclose(recon, X[0], atol=1e-2)


def test_low_beta_blends_patterns():
    X = np.eye(2, dtype=np.float32)
    query = np.array([0.5, 0.5], dtype=np.float32)
    _recon, weights = retrieve(X, query, beta=0.001)
    assert abs(float(weights[0]) - float(weights[1])) < 0.05


def test_pattern_completion_with_mask():
    # Two stored patterns; the query only knows dimension 0.
    X = np.array([[1, 2, 3], [-1, -2, -3]], dtype=np.float32)
    query = np.array([1, 0, 0], dtype=np.float32)
    mask = np.array([True, False, False])
    completed, weights = retrieve(X, query, beta=10.0, mask=mask)
    assert int(np.argmax(weights)) == 0          # attends to the positive pattern
    assert np.isclose(completed[0], 1.0)         # known field preserved
    assert completed[1] > 0 and completed[2] > 0  # unknown fields completed


def test_empty_memory_returns_empty_weights():
    X = np.zeros((0, 3), dtype=np.float32)
    weights = attention_weights(X, np.zeros(3, dtype=np.float32), 1.0)
    assert weights.shape == (0,)
