# Custom components

DeepDiscover uses Keras layers for trainable model components and ordinary Python/TensorFlow callables for objectives and metrics.

## Custom sources and sinks

A source must return a non-negative `(B,1)` quantity from `call(drivers)`. A sink must return a non-negative request from `call(drivers, storage, available)`.

```python
import keras
import tensorflow as tf

class ReservoirImport(keras.layers.Layer):
    kind = "source"
    def call(self, drivers):
        return tf.nn.relu(drivers[:, 2:3])

class AdjustablePumping(keras.layers.Layer):
    kind = "sink"
    def call(self, drivers, storage, available):
        del available
        return tf.nn.relu(drivers[:, 3:4]) * tf.sigmoid(storage / 50.0)
```

Insert these layers into a `DiscoverUnit(exchanges=[...])`. Physical boundary sources must be declared once, and a sink's requested amount may be reduced when water is insufficient.

## Custom loss

```python
from deepdiscover import masked_mse, masked_mae

def combined_loss(observed, predicted, mask):
    return masked_mse(observed, predicted, mask) + 0.1 * masked_mae(observed, predicted, mask)

model.fit(train, loss={"streamflow": combined_loss})
```

Do not compute means over masked NaNs without first selecting valid observations. Define how constant series or zero denominators should be handled.

## Custom observation operator

Provide a callable or Keras layer accepting the physical output dictionary. The returned tensor must have the same shape as the target. Its trainable Keras variables will be included in calibration.

## Custom process network

A Keras layer may replace `CRCNet` if it returns a non-negative process propensity `(B,1)`. Conservation follows from the allocation formula. The customary CRC orientations are only assured by the standard `CRCNet`; custom networks require their own monotonicity checks and documentation.

## Custom routing

Fixed routing can be specified with `RoutingKernel`. Learned and dynamic routing are provided by the two built-in model classes. Additional routing models can override `route_weights_for_step` on `MultiDUModel`; the returned `(B,max_lag+1)` distribution must be non-negative and sum to one for every time step. Lag supports must be fixed at construction, and routed water must retain the kernel assigned at release.

## Reproducibility

Use stable seeds, document units and observation operators, and store the script that constructs the model. `save_checkpoint` stores parameters; custom Python definitions must still be available to reconstruct them.
