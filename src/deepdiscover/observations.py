"""Observation relationships between model quantities and measured variables."""
from __future__ import annotations

import keras
import tensorflow as tf

from .model import StorageToHead, TrainableStorageToHead, observe


class DirectObservation(keras.layers.Layer):
    """Read a named physical output, such as outlet, sink_total or overflow."""
    def __init__(self, key="outlet", index=None, **kwargs):
        super().__init__(**kwargs)
        self.key = str(key)
        self.index = None if index is None else int(index)

    def call(self, physical_output):
        if self.key not in physical_output:
            raise KeyError(self.key)
        value = physical_output[self.key]
        return value if self.index is None else value[..., self.index:self.index + 1]

    def get_config(self):
        return {**super().get_config(), "key": self.key, "index": self.index}


class StorageToDepth(keras.layers.Layer):
    """Depth to groundwater in length units, decreasing as storage increases."""
    def __init__(self, du_index=0, reference_depth=10.0, scale=1.0, **kwargs):
        super().__init__(**kwargs)
        if scale <= 0:
            raise ValueError("scale must be positive.")
        self.du_index = int(du_index)
        self.reference_depth = float(reference_depth)
        self.scale = float(scale)

    def call(self, physical_output):
        return self.reference_depth - self.scale * physical_output["storage"][..., self.du_index:self.du_index+1]


class StorageToMoisture(keras.layers.Layer):
    """Illustrative storage-to-moisture conversion with declared water capacity.

    This operator assumes the selected storage represents the measurement volume.
    It does not infer soil texture, porosity or observation depth.
    """
    def __init__(self, du_index=0, capacity=100.0, residual=0.0, saturated=1.0, **kwargs):
        super().__init__(**kwargs)
        if capacity <= 0 or not 0 <= residual < saturated:
            raise ValueError("capacity must be positive and residual < saturated.")
        self.du_index = int(du_index)
        self.capacity = float(capacity)
        self.residual = float(residual)
        self.saturated = float(saturated)

    def call(self, physical_output):
        storage = physical_output["storage"][..., self.du_index:self.du_index+1]
        fraction = tf.clip_by_value(storage / self.capacity, 0.0, 1.0)
        return self.residual + (self.saturated - self.residual) * fraction


__all__ = ["DirectObservation", "StorageToHead", "TrainableStorageToHead", "StorageToDepth", "StorageToMoisture", "observe"]
