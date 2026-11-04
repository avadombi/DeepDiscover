"""Dataset checks independent of the TensorFlow runtime."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import numpy as np
import pytest

path = Path(__file__).resolve().parents[1] / "src" / "deepdiscover" / "dataset.py"
spec = spec_from_file_location("deepdiscover_dataset_test", path)
module = module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
HydroDataset = module.HydroDataset


def make_data():
    dates = np.arange("2020-01-01", "2020-03-01", dtype="datetime64[D]")
    p = np.maximum(np.sin(np.arange(len(dates)) / 4), 0).astype(np.float32)
    level = np.arange(len(p), dtype=np.float32)
    level[5] = np.nan
    return HydroDataset(p, np.stack([p, p + 1], axis=-1), {"level": level}, time=dates)


def test_window_shapes_and_warmup_mask():
    data = make_data()
    batch = data.make_windows(warmup=3, score_length=7, stride=3, first_score=5, last_score=46)
    assert batch.water.shape[1:] == (10, 1)
    assert batch.drivers.shape[1:] == (10, 2)
    assert np.all(batch.masks["level"][:, :3] == 0)
    assert batch.masks["level"].shape == batch.targets["level"].shape
    assert batch.water.shape[0] > 1


def test_missing_observation_is_masked():
    data = make_data()
    batch = data.continuous_batch(start=5, stop=15, warmup=2)
    assert batch.masks["level"][0, 2, 0] == 0
    assert np.isfinite(batch.targets["level"]).all()


def test_rejects_invalid_inputs_and_irregular_time():
    with pytest.raises(ValueError):
        HydroDataset([1, -2, 1], [[1], [1], [1]], {"q": [1, 2, 3]})
    with pytest.raises(ValueError):
        HydroDataset([1, 2, 3], [[1], [1], [1]], {"q": [1, 2, 3]},
                     time=np.array(["2020-01-01", "2020-01-02", "2020-01-04"], dtype="datetime64[D]"))
