"""Independent arithmetic checks of the water allocation equations."""
import numpy as np


def test_releases_storage_loss_overflow_close():
    rng = np.random.default_rng(5)
    available = rng.uniform(0, 200, (200, 1))
    requests = rng.uniform(0, 100, (200, 3))
    requested_total = requests.sum(axis=-1, keepdims=True)
    reduction = np.minimum(1.0, np.divide(available, requested_total,
                                          out=np.zeros_like(available), where=requested_total > 0))
    losses = requests * reduction
    resource = np.maximum(available - losses.sum(axis=-1, keepdims=True), 0.0)
    propensity = rng.uniform(0, 10, (200, 4))
    storage_raw = resource / (1 + propensity.sum(axis=-1, keepdims=True))
    releases = storage_raw * propensity
    capacity = 5.0
    overflow = np.maximum(storage_raw - capacity, 0)
    storage = np.minimum(storage_raw, capacity)
    residual = available - losses.sum(axis=-1, keepdims=True) - releases.sum(axis=-1, keepdims=True) - overflow - storage
    assert np.max(np.abs(residual)) < 1e-12
    assert np.min(storage) >= 0


def test_routing_weights_stay_with_released_water():
    rng = np.random.default_rng(9)
    released = rng.uniform(0, 5, 100)
    fractions = rng.dirichlet(np.ones(4), size=100)
    arrivals = np.zeros(103)
    for t in range(100):
        arrivals[t:t+4] += released[t] * fractions[t]
    assert np.isclose(arrivals.sum(), released.sum(), atol=1e-12)
    assert np.min(arrivals) >= 0


def test_constant_routing_conservation():
    release = np.array([8, 0, 2, 0], dtype=float)
    kernel = np.array([0.25, 0.50, 0.25])
    arrival = np.convolve(release, kernel)
    assert np.isclose(arrival.sum(), release.sum())
    np.testing.assert_allclose(arrival[:3], [2.0, 4.0, 2.5])
