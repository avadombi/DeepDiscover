# Observation operators

Observation operators map hydrological outputs to measurable variables without modifying the water balance. Any callable that receives the physical output dictionary and returns a tensor `(B,T,n_outputs)` can serve as an operator.

Built-in operators include:

- `DirectObservation(key="outlet")`: select an existing physical output or its column.
- `StorageToHead(du_index=..., scale=..., offset=...)`: `head = offset + scale * storage`.
- `TrainableStorageToHead(...)`: learn scale and/or offset; scale remains positive.
- `StorageToDepth(du_index=..., reference_depth=..., scale=...)`: depth to water decreases with increasing storage.
- `StorageToMoisture(du_index=..., capacity=..., residual=..., saturated=...)`: bounded moisture proxy.

```python
from deepdiscover import DeepDiscover, StorageToHead
model = DeepDiscover(physical_model, observations={
    "flow": "outlet",
    "head": StorageToHead(du_index=1, scale=0.02, offset=110.0),
})
```

One objective can include both measurements. When a measured variable has a distinct physical dimension or scale, normalization and the chosen operator must be explained. An affine head operator may be useful without uniquely identifying a physical aquifer storage.

To add a custom operator:

```python
import keras
import tensorflow as tf

class MyHead(keras.layers.Layer):
    def call(self, physical):
        storage = physical["storage"][..., 0:1]
        return 100.0 + 0.05 * tf.sqrt(storage + 1.0)
```

Only differentiable TensorFlow/Keras operations can contribute gradients to the training objective. Portable checkpoint restoration requires the same operator definition and architecture.
