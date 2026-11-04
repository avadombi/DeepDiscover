"""Driver selection and compiled simulation tests."""
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
from deepdiscover import (
    CRCNet, DeepDiscover, DemandLimitedSink, DifferentiableDegreeDaySnow,
    DiscoverUnit, HydroDataset, MultiDUModel,
)


def build_model():
    efficiency = CRCNet([1], [100], hidden_units=4, output="sigmoid")
    du0 = DiscoverUnit(
        [CRCNet([1, 1, 1], [10, 10, 100], hidden_units=4),
         CRCNet([1, 1, 1], [10, 10, 100], hidden_units=4)],
        context_indices=[(0, 1), (0, 1)],
        exchanges=[DemandLimitedSink(efficiency, demand_index=2)],
    )
    du1 = DiscoverUnit(
        [CRCNet([1, 1], [100, 10], hidden_units=4)],
        context_indices=[()], upstream_count=1,
    )
    return DeepDiscover(
        MultiDUModel([du0, du1], [["outlet", 1], ["outlet"]]),
        snow=DifferentiableDegreeDaySnow(),
        temperature_index=2,
        driver_channels=("rain", "snowmelt", 1),
    )


def test_snow_drivers_and_batch_independence():
    dd = build_model()
    days = 14
    p = np.tile(np.array([1, 0, 3, 2, 0, 1, 5], np.float32), 2)
    pet = np.full(days, 1.3, np.float32)
    temp = np.tile(np.array([-4, -3, 0, 3, 6, 2, -2], np.float32), 2)
    water = np.zeros((2, days, 2), np.float32)
    water[:, :, 0] = p
    drivers = np.stack([p, pet, temp], axis=-1)[None].repeat(2, axis=0)
    drivers[1, :, 0] *= 0.5
    water[1, :, 0] *= 0.5

    output = dd.simulate(water, drivers)
    assert output["process_drivers"].shape == (2, days, 3)
    np.testing.assert_allclose(output["process_drivers"][:, :, 0:1], output["rain"], atol=1e-6)
    np.testing.assert_allclose(output["process_drivers"][:, :, 1:2], output["snowmelt"], atol=1e-6)
    np.testing.assert_allclose(output["process_drivers"][:, :, 2:3], drivers[:, :, 1:2], atol=1e-6)
    assert float(tf.reduce_max(tf.abs(output["mass_residual"]))) < 1e-4
    solo = dd.simulate(water[:1], drivers[:1])
    np.testing.assert_allclose(output["outlet"][:1], solo["outlet"], rtol=1e-5, atol=1e-5)


def test_full_parallel_training_with_periodic_validation():
    dd = build_model()
    t = np.arange(65, dtype=np.float32)
    p = np.maximum(1.5 + np.sin(t / 3), 0.0).astype(np.float32)
    temp = (4 + 10 * np.sin(t / 8)).astype(np.float32)
    pet = np.full(len(t), 0.5, np.float32)
    water = np.column_stack([p, np.zeros_like(p)])
    drivers = np.column_stack([p, pet, temp])
    data = HydroDataset(water, drivers, {"streamflow": p * 0.3})
    train = data.make_windows(5, 10, stride=10, first_score=5, last_score=55)
    validation = data.continuous_batch(55, 65, warmup=5)
    history = dd.fit(
        train, validation, epochs=3, batch_size=None,
        validation_every=2, learning_rate=0.002,
        shuffle=False, normalize=False, verbose=0,
    )
    assert len(history) == 3
    assert "validation_loss" not in history[0]
    assert "validation_loss" in history[1]
    assert "validation_loss" in history[2]  # final epoch
    assert all(item["epoch_seconds"] >= 0 for item in history)
    assert np.isfinite(history[-1]["validation_loss"])


def test_invalid_driver_channels():
    model = MultiDUModel([DiscoverUnit([CRCNet([1], [100])], context_indices=[()])], [["outlet"]])
    with pytest.raises(ValueError, match="requires a snow model"):
        DeepDiscover(model, driver_channels=("rain",))
    with pytest.raises(ValueError, match="non-negative"):
        DeepDiscover(model, driver_channels=(-1,))


def test_compiled_snow_layer_and_float64_inputs():
    snow = DifferentiableDegreeDaySnow()
    precipitation = tf.constant([[[2.0], [0.0], [1.0], [0.0]]], tf.float64)
    temperature = tf.constant([[[-4.0], [-3.0], [5.0], [7.0]]], tf.float64)

    @tf.function
    def forward_and_gradient(p, temp):
        with tf.GradientTape() as tape:
            result = snow({"precipitation": p, "temperature": temp})
            objective = tf.reduce_sum(result["liquid_water"])
        gradients = tape.gradient(objective, snow.trainable_variables)
        return result, gradients

    result, gradients = forward_and_gradient(precipitation, temperature)
    assert result["liquid_water"].shape == (1, 4, 1)
    assert float(tf.reduce_max(tf.abs(result["mass_residual"]))) < 1e-5
    assert any(gradient is not None for gradient in gradients)
    for gradient in gradients:
        if gradient is not None:
            assert np.isfinite(gradient.numpy()).all()


def test_checkpoint_with_snow_and_driver_channels(tmp_path):
    days = 8
    precipitation = np.array([1.0, 0.0, 3.0, 2.0, 0.0, 4.0, 1.0, 0.0], np.float32)
    water = np.column_stack([precipitation, np.zeros(days, np.float32)])[None]
    drivers = np.column_stack([precipitation, np.ones(days, np.float32),
                               np.array([-3, -2, 3, 5, -1, 4, 0, -4], np.float32)])[None]

    trained = build_model()
    expected = trained.predict(water, drivers)["streamflow"].numpy()
    trained.save_checkpoint(tmp_path)

    restored = build_model()
    restored.predict(water, drivers)
    restored.load_checkpoint(tmp_path)
    actual = restored.predict(water, drivers)["streamflow"].numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
