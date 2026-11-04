"""Short installation check for a new DeepDiscover environment.

Run from the repository root:
    python check.py
    python -m pytest -q
"""
from pathlib import Path
import numpy as np
import tensorflow as tf
import keras
from deepdiscover import (
    CRCNet, DiscoverUnit, MultiDUModel, DeepDiscover,
    PrescribedSink, HydroDataset,
)


def main():
    print('TensorFlow', tf.__version__, 'Keras', keras.__version__)
    t = np.arange(45, dtype=np.float32)
    rain = 2.0 + np.sin(t / 5)
    pet = np.ones_like(rain)
    drivers = np.stack([rain, pet], axis=-1)[None]
    water = rain[None, :, None]
    du = DiscoverUnit(
        process_nets=[CRCNet([1, -1, 1], [5, 5, 100])],
        context_indices=[(0, 1)],
        exchanges=[PrescribedSink(driver_index=1, scale=0.1)],
    )
    model = DeepDiscover(MultiDUModel([du], [['outlet']]))
    out = model.simulate(water, drivers)
    error = float(tf.reduce_max(tf.abs(out['mass_residual'])).numpy())
    print('Shapes:', {k: tuple(out[k].shape) for k in
                       ('outlet', 'storage', 'process_fluxes', 'sink_total', 'mass_residual')})
    print('Maximum mass residual:', error)
    assert error < 1e-4
    with tf.GradientTape() as tape:
        y = model.simulate(water, drivers)['outlet']
        objective = tf.reduce_mean(tf.square(y))
    grad = tape.gradient(objective, model._variables())
    assert any(g is not None for g in grad)
    assert all(np.isfinite(g.numpy()).all() for g in grad if g is not None)
    dataset = HydroDataset(rain, np.stack([rain, pet], axis=-1), {'streamflow': rain})
    train = dataset.make_windows(warmup=5, score_length=10, stride=5)
    model.fit(train, epochs=1, batch_size=2, verbose=0)
    score = model.evaluate(train)
    print('Training score:', score['loss'])
    print('Installation check passed')


if __name__ == '__main__':
    main()
