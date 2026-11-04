import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
from deepdiscover import (
    DeepDiscover, CRCNet, DiscoverUnit, MultiDUModel, HydroDataset,
    TrainableStorageToHead, masked_mse,
)


def model_with_head():
    du = DiscoverUnit([CRCNet([1, 1], [5, 100])], context_indices=[(0,)])
    model = MultiDUModel([du], [["outlet"]])
    return DeepDiscover(model, {"flow": "outlet", "head": TrainableStorageToHead(0, 0.02, 10.0)})


def example():
    t = np.arange(36, dtype=np.float32)
    p = 3 + np.sin(t/3)
    driver = np.column_stack([p, np.ones_like(p)])
    y = np.column_stack([p])
    head = 10 + 0.5*np.sin(t/6)
    head[12] = np.nan
    return HydroDataset(p, driver, {"flow": y, "head": head})


def test_training_validation_custom_loss_and_checkpoint(tmp_path):
    dataset = example()
    train = dataset.make_windows(warmup=3, score_length=8, stride=4, first_score=3, last_score=27)
    valid = dataset.continuous_batch(27, 36, warmup=3)
    dd = model_with_head()
    def loss(y, sim, mask):
        return masked_mse(y, sim, mask)
    history = dd.fit(train, valid, epochs=2, batch_size=2, loss={"flow": loss, "head": "huber"}, verbose=0)
    assert len(history) == 2
    before = dd.predict(valid.water, valid.drivers)
    dd.save_checkpoint(tmp_path, metadata={"basin": "synthetic"})
    other = model_with_head()
    other.predict(valid.water, valid.drivers)
    restored = other.load_checkpoint(tmp_path)
    assert restored["metadata"]["basin"] == "synthetic"
    after = other.predict(valid.water, valid.drivers)
    for k in before:
        np.testing.assert_allclose(before[k], after[k], rtol=1e-5, atol=1e-5)
    summary = other.evaluate(valid)
    assert "flow" in summary["metrics"]
    assert np.isfinite(summary["loss"])
