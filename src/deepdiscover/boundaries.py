"""External inputs and losses acting on a DiscoverUnit.

Every boundary contribution is a non-negative amount of water per model step.
Sources add water before losses; losses are constrained jointly by availability.
The classes here may be replaced with user-defined Keras layers declaring
``kind = 'source'`` or ``kind = 'sink'``.
"""
from __future__ import annotations

import keras
import tensorflow as tf

from .model import PrescribedSource, DemandLimitedSink


class PrescribedSink(keras.layers.Layer):
    """Request a non-negative extraction from a driver column.

    Examples include pumping, an imposed lateral loss, or a managed release
    outside the model domain. The DiscoverUnit limits all requested losses
    together to the available water.
    """

    kind = "sink"

    def __init__(self, driver_index: int, scale: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        if int(driver_index) < 0 or scale < 0:
            raise ValueError("driver_index must be non-negative and scale cannot be negative.")
        self.driver_index = int(driver_index)
        self.scale = float(scale)

    def call(self, drivers, storage, available):
        del storage, available
        request = drivers[:, self.driver_index:self.driver_index + 1] * self.scale
        tf.debugging.assert_greater_equal(request, 0.0)
        return request

    def get_config(self):
        return {**super().get_config(), "driver_index": self.driver_index, "scale": self.scale}


class FunctionalSource(keras.layers.Layer):
    """Source computed by a user-supplied differentiable Keras layer.

    ``layer(drivers)`` must return a non-negative tensor of shape ``(B, 1)``.
    For portable saved models, register the layer as a Keras serializable class.
    """
    kind = "source"

    def __init__(self, layer: keras.layers.Layer, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(layer, keras.layers.Layer):
            raise TypeError("layer must be a Keras layer.")
        self.layer = layer

    def call(self, drivers):
        result = self.layer(drivers)
        tf.debugging.assert_rank(result, 2)
        tf.debugging.assert_equal(tf.shape(result)[-1], 1)
        return result


class FunctionalSink(keras.layers.Layer):
    """Requested loss from a custom differentiable Keras layer.

    ``layer(drivers, storage, available)`` returns ``(B, 1)`` in water units
    per time step. Requests exceeding available water are reduced together
    with all other losses in the same DU.
    """
    kind = "sink"

    def __init__(self, layer: keras.layers.Layer, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(layer, keras.layers.Layer):
            raise TypeError("layer must be a Keras layer.")
        self.layer = layer

    def call(self, drivers, storage, available):
        request = self.layer(drivers, storage, available)
        tf.debugging.assert_rank(request, 2)
        tf.debugging.assert_equal(tf.shape(request)[-1], 1)
        return request


__all__ = ["PrescribedSource", "DemandLimitedSink", "PrescribedSink", "FunctionalSource", "FunctionalSink"]
