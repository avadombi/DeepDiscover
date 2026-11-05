"""Two three-process networks with distinct internal structure."""
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
from deepdiscover import CRCNet, DeepDiscover, DiscoverUnit, MultiDUModel


def build_candidate(kind):
    def net(n):
        return CRCNet([1] * n, [10.0, 100.0] if n == 2 else [10.0, 100.0, 10.0],
                      hidden_units=4)
    if kind == "direct":
        du0 = DiscoverUnit([net(2), net(2)], context_indices=[(0,), (0,)])
        du1 = DiscoverUnit([net(3)], context_indices=[(0,)], upstream_count=1)
        routes = [["outlet", 1], ["outlet"]]
    else:
        du0 = DiscoverUnit([net(2)], context_indices=[(0,)])
        du1 = DiscoverUnit([net(3), net(3)], context_indices=[(0,), (0,)], upstream_count=1)
        routes = [[1], ["outlet", "outlet"]]
    return DeepDiscover(MultiDUModel([du0, du1], routes))


@pytest.mark.parametrize("kind", ["direct", "downstream"])
def test_three_process_structures_conserve_water(kind):
    dd = build_candidate(kind)
    p = np.array([0., 2., 1., 0., 4., 0., 1., 3.], dtype=np.float32)
    water = np.stack([p, np.zeros_like(p)], axis=-1)[None]
    drivers = p[None, :, None]
    result = dd.simulate(water, drivers)
    assert result["outlet"].shape == (1, len(p), 1)
    assert result["storage"].shape == (1, len(p), 2)
    assert result["process_fluxes"].shape == (1, len(p), 3)
    assert dd.model.n_routes == 3
    assert np.max(np.abs(result["mass_residual"].numpy())) < 2e-5


def test_cross_topology_checkpoint_rejected(tmp_path):
    direct = build_candidate("direct")
    downstream = build_candidate("downstream")
    p = np.array([1., 2., 1., 3.], np.float32)
    water = np.stack([p, np.zeros_like(p)], -1)[None]
    drivers = p[None, :, None]
    direct.predict(water, drivers)
    downstream.predict(water, drivers)
    direct.save_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="architecture|topology|parameter"):
        downstream.load_checkpoint(tmp_path)


def make_unit(n_incoming, n_processes=1):
    n_context = 2 + n_incoming
    return DiscoverUnit(
        process_nets=[
            CRCNet([1] * n_context, [5.0, 100.0] + [5.0] * n_incoming,
                   hidden_units=4)
            for _ in range(n_processes)
        ],
        context_indices=[(0,)] * n_processes,
        upstream_count=n_incoming,
    )


def make_inputs(n_du):
    rain = np.array([2., 0., 3., 1., 4., 0., 2., 5., 1., 0.], np.float32)
    water = np.zeros((2, len(rain), n_du), np.float32)
    water[0, :, 0] = rain
    water[1, :, 0] = rain * 0.5
    drivers = water[:, :, :1].copy()
    return water, drivers


@pytest.mark.parametrize("n_du", [3, 4, 5])
def test_long_serial_cascade_conserves_water(n_du):
    units = [make_unit(0)] + [make_unit(1) for _ in range(n_du - 1)]
    destinations = [[i + 1] for i in range(n_du - 1)] + [["outlet"]]
    dd = DeepDiscover(MultiDUModel(units, destinations))
    water, drivers = make_inputs(n_du)
    output = dd.simulate(water, drivers)

    assert output["outlet"].shape == (2, water.shape[1], 1)
    assert output["storage"].shape == (2, water.shape[1], n_du)
    assert output["process_fluxes"].shape == (2, water.shape[1], n_du)
    assert dd.model.n_routes == n_du
    assert np.any(output["outlet"].numpy() > 0)
    assert np.min(output["storage"].numpy()) >= 0
    assert np.max(np.abs(output["mass_residual"].numpy())) < 1e-4

    # With instantaneous routing, the only water remaining is in DU storage.
    final_storage = output["final_storage"].numpy().sum(axis=1)
    total_outlet = output["outlet"].numpy().sum(axis=(1, 2))
    np.testing.assert_allclose(final_storage + total_outlet,
                               water.sum(axis=(1, 2)), rtol=1e-5, atol=1e-4)


def test_four_du_branching_conservation_and_gradients():
    units = [make_unit(0, 2), make_unit(1), make_unit(1), make_unit(2)]
    dd = DeepDiscover(MultiDUModel(units, [[1, 2], [3], [3], ["outlet"]]))
    water, drivers = make_inputs(4)

    with tf.GradientTape() as tape:
        output = dd.simulate(water, drivers)
        objective = tf.reduce_sum(output["outlet"])
    gradients = tape.gradient(objective, dd._variables())

    assert dd.model.n_routes == 5
    assert output["process_fluxes"].shape == (2, water.shape[1], 5)
    assert output["storage"].shape == (2, water.shape[1], 4)
    assert np.all(output["internal_inflow"].numpy()[:, 1:, 1:] >= 0)
    assert np.any(output["internal_inflow"].numpy()[:, :, 3] > 0)
    assert np.max(np.abs(output["mass_residual"].numpy())) < 1e-4
    assert any(grad is not None for grad in gradients)
    for grad in gradients:
        if grad is not None:
            assert np.isfinite(grad.numpy()).all()

    final_storage = output["final_storage"].numpy().sum(axis=1)
    total_outlet = output["outlet"].numpy().sum(axis=(1, 2))
    np.testing.assert_allclose(final_storage + total_outlet,
                               water.sum(axis=(1, 2)), rtol=1e-5, atol=1e-4)


def test_branching_requires_correct_incoming_component_count():
    units = [make_unit(0, 2), make_unit(1), make_unit(1), make_unit(1)]
    with pytest.raises(ValueError, match="upstream_count"):
        MultiDUModel(units, [[1, 2], [3], [3], ["outlet"]])


def test_backward_transfer_is_rejected():
    units = [make_unit(0), make_unit(1), make_unit(1)]
    with pytest.raises(ValueError, match="topological order"):
        MultiDUModel(units, [[1], [2], [0]])
