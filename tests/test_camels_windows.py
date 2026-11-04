"""Temporal windows matching the complete CAMELS study periods."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import numpy as np

module_file = Path(__file__).resolve().parents[1] / "src" / "deepdiscover" / "dataset.py"
spec = spec_from_file_location("deepdiscover_dataset_windows", module_file)
module = module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
HydroDataset = module.HydroDataset


def test_full_period_shapes_and_warmup():
    t = 10957
    day = np.arange(t, dtype=np.float32)
    rain = np.ones(t, np.float32)
    data = HydroDataset(
        np.column_stack((rain, np.zeros(t, np.float32))),
        np.column_stack((rain, rain * 0.2, day * 0)),
        {"streamflow": rain * 0.5},
    )
    train_end = 365 + 6940
    val_end = train_end + 1826
    train = data.make_windows(365, 365, stride=365, first_score=365, last_score=train_end)
    validation = data.continuous_batch(train_end, val_end, warmup=365)
    test = data.continuous_batch(val_end, t, warmup=365)
    assert train.water.shape == (19, 730, 2)
    assert train.drivers.shape == (19, 730, 3)
    assert train.targets["streamflow"].shape == (19, 730, 1)
    assert validation.water.shape == (1, 2191, 2)
    assert test.water.shape == (1, 2191, 2)
    np.testing.assert_array_equal(train.masks["streamflow"][:, :365], 0)
    np.testing.assert_array_equal(train.masks["streamflow"][:, 365:], 1)
    assert np.sum(train.masks["streamflow"]) == 19 * 365
    assert np.sum(validation.masks["streamflow"]) == 1826
    assert np.sum(test.masks["streamflow"]) == 1826
    # The 365-day warm-up is taken from immediately before the scored period.
    np.testing.assert_array_equal(test.water[0, :365], data.water[val_end - 365:val_end])
