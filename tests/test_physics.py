import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
from deepdiscover import (
    DeepDiscover, CRCNet, DiscoverUnit, MultiDUModel, RoutingKernel,
    TrainableRoutingMultiDUModel, DynamicRoutingMultiDUModel,
    PrescribedSource, PrescribedSink, DemandLimitedSink,
    InformationalMemoryCell, DifferentiableDegreeDaySnow,
)


def make_unit(upstream=0, bounded=False, memory=False):
    extra = int(memory) + int(bounded) + upstream
    gamma = [1, -1, 1] + [1] * extra
    refs = [5.0, 3.0, 100.0] + [100.0] * extra
    cell = InformationalMemoryCell([0], gamma=[1, 1], x_ref=[5, 100]) if memory else None
    return DiscoverUnit(
        process_nets=[CRCNet(gamma, refs)], context_indices=[(0, 1)],
        upstream_count=upstream, smax=30.0 if bounded else None,
        include_deficit=bounded, memory_cell=cell,
    )


def make_model(mode="fixed", memory=False, bounded=False):
    units = [make_unit(memory=memory, bounded=bounded), make_unit(upstream=2 if bounded else 1)]
    routes = [[RoutingKernel([0.6, 0.4])], [RoutingKernel([0.2, 0.5, 0.3])]]
    overflow = [1, None] if bounded else None
    overflow_routes = [RoutingKernel([0.4, 0.6]), None] if bounded else None
    base = dict(discover_units=units, destinations=[[1], ["outlet"]],
                overflow_destinations=overflow, routing=routes, overflow_routing=overflow_routes)
    if mode == "fixed":
        return MultiDUModel(**base)
    if mode == "learned":
        return TrainableRoutingMultiDUModel(**base)
    if mode == "dynamic":
        return DynamicRoutingMultiDUModel(**base)
    raise ValueError(mode)


@pytest.mark.parametrize("mode", ["fixed", "learned", "dynamic"])
@pytest.mark.parametrize("bounded", [False, True])
def test_global_conservation_and_shapes(sample, mode, bounded):
    water, drivers = sample
    model = make_model(mode=mode, bounded=bounded)
    result = DeepDiscover(model).simulate(np.concatenate([water, np.zeros_like(water)], -1), drivers)
    assert result["storage"].shape == (3, 50, 2)
    assert result["route_arrivals"].shape[-1] == (3 if bounded else 2)
    assert result["transit"].shape == (3, 50, model.n_routes)
    assert result["routing_weights"].shape == (3, 50, model.n_routes, 3)
    assert np.max(np.abs(result["mass_residual"].numpy())) < 2e-4
    assert np.min(result["storage"].numpy()) >= 0
    if bounded:
        assert np.max(result["storage"].numpy()[:, :, 0]) <= 30.0001


def test_batch_independence(sample):
    water, drivers = sample
    water = np.concatenate([water, np.zeros_like(water)], -1)
    dd = DeepDiscover(make_model("dynamic"))
    whole = dd.simulate(water, drivers)["outlet"].numpy()
    individual = dd.simulate(water[1:2], drivers[1:2])["outlet"].numpy()
    np.testing.assert_allclose(whole[1:2], individual, rtol=1e-5, atol=1e-5)


def test_continue_from_final_states(sample):
    water, drivers = sample
    water = np.concatenate([water, np.zeros_like(water)], -1)
    dd = DeepDiscover(make_model("fixed"))
    all_at_once = dd.simulate(water, drivers)
    first = dd.simulate(water[:, :20], drivers[:, :20])
    second = dd.simulate(
        water[:, 20:], drivers[:, 20:],
        initial_storage=first["final_storage"],
        initial_transit=first["final_transit_state"],
        initial_memory=first["final_memory"],
    )
    np.testing.assert_allclose(second["outlet"], all_at_once["outlet"][:, 20:], atol=2e-5)


def test_external_sources_sinks_competition(sample):
    water, drivers = sample
    sink1 = PrescribedSink(driver_index=0, scale=12.0)
    sink2 = PrescribedSink(driver_index=1, scale=12.0)
    src = PrescribedSource(driver_index=1, scale=1.0)
    du = DiscoverUnit([CRCNet([1, 1], [1, 100])], context_indices=[(0,)],
                      exchanges=[src, sink1, sink2])
    dd = DeepDiscover(MultiDUModel([du], [["outlet"]]))
    output = dd.simulate(water, drivers)
    assert np.max(np.abs(output["mass_residual"].numpy())) < 3e-5
    assert np.min(output["storage"].numpy()) >= 0
    assert np.any(output["sink_total"].numpy() > 0)


def test_optional_memory(sample):
    water, drivers = sample
    dd = DeepDiscover(make_model("fixed", memory=True))
    output = dd.simulate(np.concatenate([water, np.zeros_like(water)], -1), drivers)
    assert np.min(output["memory"].numpy()) >= 0
    assert np.max(np.abs(output["mass_residual"].numpy())) < 1e-4


def test_snow_conserves_water(sample):
    water, drivers = sample
    # The second driver is temperature for this simulation.
    drivers = drivers.copy()
    temperature = np.where(np.arange(drivers.shape[1])[None, :] < 25, -3.0, 4.0)
    drivers = np.concatenate([drivers, np.broadcast_to(temperature[..., None], (3, 50, 1))], axis=-1)
    snow = DifferentiableDegreeDaySnow()
    dd = DeepDiscover(make_model("fixed"), snow=snow, snow_unit=0, temperature_index=2)
    out = dd.simulate(np.concatenate([water, np.zeros_like(water)], -1), drivers)
    assert np.max(np.abs(out["mass_residual"].numpy())) < 1e-4
    assert np.any(out["snow_storage"].numpy() > 0)


def test_finite_gradients(sample):
    water, drivers = sample
    dd = DeepDiscover(make_model("dynamic", bounded=True))
    with tf.GradientTape() as tape:
        out = dd.simulate(np.concatenate([water, np.zeros_like(water)], -1), drivers)
        loss = tf.reduce_mean(tf.square(out["outlet"]))
    gradients = tape.gradient(loss, dd._variables())
    assert any(g is not None for g in gradients)
    for grad in gradients:
        if grad is not None:
            assert np.isfinite(grad.numpy()).all()


def test_instantaneous_routing_matches_one_tap(sample):
    water, drivers = sample
    unit = make_unit()
    instant = MultiDUModel([unit], [["outlet"]])
    one_tap = MultiDUModel([make_unit()], [["outlet"]], routing=[[RoutingKernel([1.0])]])
    dd1, dd2 = DeepDiscover(instant), DeepDiscover(one_tap)
    o1 = dd1.simulate(water, drivers)
    dd2.simulate(water, drivers)
    for source, target in zip(dd1._variables(), dd2._variables()):
        target.assign(source)
    o2 = dd2.simulate(water, drivers)
    np.testing.assert_allclose(o1["outlet"], o2["outlet"], atol=1e-6)


def test_local_allocation_exact():
    from deepdiscover import conservative_allocation
    resource = tf.constant([[5.0], [12.0]], dtype=tf.float32)
    tendencies = tf.constant([[1.0, 2.0], [0.0, 3.0]], tf.float32)
    storage, processes, overflow = conservative_allocation(resource, tendencies, smax=1.2)
    np.testing.assert_allclose((storage + tf.reduce_sum(processes, axis=-1, keepdims=True)
                                + overflow).numpy(), resource.numpy(), atol=2e-6)


def test_route_kernel_conservation(sample):
    water, drivers = sample
    dd = DeepDiscover(make_model("dynamic", bounded=True))
    physical = dd.simulate(np.concatenate([water, np.zeros_like(water)], -1), drivers)
    weights = physical["routing_weights"].numpy()
    assert np.min(weights) >= 0
    np.testing.assert_allclose(weights.sum(-1), np.ones_like(weights.sum(-1)), atol=1e-6)
    assert np.min(physical["transit"].numpy()) >= 0


def test_trainable_capacity_gradients(sample):
    water, drivers = sample
    du = DiscoverUnit(
        [CRCNet([1, 1, 1], [5, 100, 100])],
        context_indices=[(0,)], include_deficit=True, smax_bounds=(5.0, 80.0),
    )
    dd = DeepDiscover(MultiDUModel([du], [["outlet"]], overflow_destinations=["outlet"]))
    with tf.GradientTape() as tape:
        physical = dd.simulate(water, drivers)
        objective = tf.reduce_sum(physical["storage"])
    gradients = tape.gradient(objective, dd._variables())
    assert np.isfinite(du.smax.numpy()).all()
    assert 5 <= du.smax.numpy().item() <= 80
    assert any(g is not None for g in gradients)
    assert np.max(np.abs(physical["mass_residual"].numpy())) < 1e-4


def test_custom_boundary_and_process(sample):
    import keras
    water, drivers = sample

    class CustomSource(keras.layers.Layer):
        kind = 'source'
        def call(self, information):
            return tf.nn.relu(information[:, 1:2])

    class CustomSink(keras.layers.Layer):
        kind = 'sink'
        def call(self, information, storage, available):
            return tf.minimum(available, 0.2 * information[:, 0:1])

    class CustomProcess(keras.layers.Layer):
        def call(self, context):
            return tf.nn.softplus(context[:, -1:] * 0.001)

    du = DiscoverUnit([CustomProcess()], context_indices=[(0, 1)],
                      exchanges=[CustomSource(), CustomSink()])
    output = DeepDiscover(MultiDUModel([du], [["outlet"]])).simulate(water, drivers)
    assert output["source_fluxes"].shape == (3, 50, 1)
    assert output["sink_fluxes"].shape == (3, 50, 1)
    assert np.max(np.abs(output["mass_residual"].numpy())) < 1e-4


def test_rejects_backward_route():
    with pytest.raises(ValueError):
        MultiDUModel([make_unit(), make_unit(upstream=1)], [["outlet"], [0]])
