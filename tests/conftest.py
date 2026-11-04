"""Test data and model configuration shared across numerical checks."""
from __future__ import annotations
import numpy as np
import pytest


@pytest.fixture
def sample():
    t = np.arange(50, dtype=np.float32)
    p = (3.0 + np.sin(t/3.0)).astype(np.float32)
    pet = np.full_like(p, 1.2)
    water = p[None, :, None].repeat(3, axis=0)
    drivers = np.stack([p, pet], axis=-1)[None].repeat(3, axis=0)
    drivers[1, :, 0] *= 1.3
    drivers[2, :, 1] *= 0.8
    return water, drivers
