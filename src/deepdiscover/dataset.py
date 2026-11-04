"""Water inputs, predictors, observations, and temporal training windows."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import numpy as np


def _array(value, label):
    a = np.asarray(value, dtype=np.float32)
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2:
        raise ValueError(f"{label} must have shape (time, channels).")
    return a


@dataclass
class HydroBatch:
    """Independent windows, all with shape (B, T, channels)."""
    water: np.ndarray
    drivers: np.ndarray
    targets: dict[str, np.ndarray]
    masks: dict[str, np.ndarray]

    def __post_init__(self):
        self.water = np.asarray(self.water, np.float32)
        self.drivers = np.asarray(self.drivers, np.float32)
        if self.water.ndim != 3 or self.drivers.ndim != 3:
            raise ValueError("water and drivers must have shape (B,T,channels).")
        if self.water.shape[:2] != self.drivers.shape[:2]:
            raise ValueError("water and drivers must share batch and time dimensions.")
        if not np.isfinite(self.water).all() or np.any(self.water < 0):
            raise ValueError("water inputs must be finite and non-negative.")
        if not np.isfinite(self.drivers).all():
            raise ValueError("drivers must be finite; transform or fill missing predictors explicitly.")
        self.targets = {name: np.asarray(values, np.float32) for name, values in self.targets.items()}
        self.masks = {name: np.asarray(values, np.float32) for name, values in self.masks.items()}
        for name, y in self.targets.items():
            if y.ndim != 3 or y.shape[:2] != self.water.shape[:2]:
                raise ValueError(f"Target {name!r} must have shape (B,T,channels).")
            if name not in self.masks or self.masks[name].shape != y.shape:
                raise ValueError(f"Target {name!r} requires a mask with the same shape.")
            mask = self.masks[name]
            if np.any((mask != 0) & (mask != 1)):
                raise ValueError(f"Mask {name!r} must contain 0 or 1.")
            if not np.isfinite(y[mask > 0]).all():
                raise ValueError(f"Scored values in {name!r} must be finite.")
        if not self.targets:
            raise ValueError("At least one target is required.")

    def subset(self, indices):
        return HydroBatch(
            water=self.water[indices], drivers=self.drivers[indices],
            targets={k: y[indices] for k, y in self.targets.items()},
            masks={k: m[indices] for k, m in self.masks.items()},
        )

    @property
    def batch_size(self):
        return self.water.shape[0]

    @property
    def length(self):
        return self.water.shape[1]


class HydroDataset:
    """Continuous hydrological record with explicit water and information inputs.

    Parameters
    ----------
    water
        Non-negative conserved water quantities per time step and DU, (T,N_DU).
        Convert rates into quantities per model step beforehand.
    drivers
        Informational predictors, (T,N_drivers). Values must be finite.
    observations
        Names mapped to measured time series (T,n_outputs); NaNs are masked.
    masks
        Optional extra masks applied to observations, 1 for scored values.
    time
        Optional dates/timestamps of strictly increasing, regular intervals.
        Intervals are checked if NumPy datetime64 values are supplied.
    """

    def __init__(self, water, drivers, observations: Mapping[str, np.ndarray], masks=None, time=None):
        self.water = _array(water, "water")
        self.drivers = _array(drivers, "drivers")
        length = len(self.water)
        if len(self.drivers) != length:
            raise ValueError("water and drivers must have matching time lengths.")
        if not np.isfinite(self.water).all() or np.any(self.water < 0):
            raise ValueError("water must be finite and non-negative.")
        if not np.isfinite(self.drivers).all():
            raise ValueError("drivers must be finite.")
        if not observations:
            raise ValueError("observations cannot be empty.")
        self.observations = {}
        self.masks = {}
        masks = masks or {}
        for name, values in observations.items():
            y = _array(values, f"observation {name!r}")
            if len(y) != length:
                raise ValueError(f"Observation {name!r} length does not match water.")
            valid = np.isfinite(y)
            if name in masks:
                explicit = np.broadcast_to(_array(masks[name], name), y.shape)
                if not np.isin(explicit, [0, 1]).all():
                    raise ValueError(f"Mask for {name!r} must contain 0 or 1.")
                valid &= (explicit == 1)
            self.observations[name] = np.where(valid, y, 0).astype(np.float32)
            self.masks[name] = valid.astype(np.float32)
        self.time = None if time is None else np.asarray(time)
        if self.time is not None:
            if len(self.time) != length:
                raise ValueError("time must have one value per sample.")
            if np.issubdtype(self.time.dtype, np.datetime64):
                diffs = np.diff(self.time.astype("datetime64[ns]").astype("int64"))
                if diffs.size and (np.any(diffs <= 0) or np.any(diffs != diffs[0])):
                    raise ValueError("The model requires a regular, strictly increasing time step.")

    def make_windows(self, warmup: int, score_length: int, stride: int | None = None,
                     first_score: int = 0, last_score: int | None = None) -> HydroBatch:
        """Create windows with a warm-up prefix that contributes zero training loss.

        ``first_score`` and ``last_score`` delimit scored indices in the original
        record. Earlier samples may be used as warm-up without being scored.
        The end index ``last_score`` is exclusive.
        """
        warmup, score_length = int(warmup), int(score_length)
        stride = score_length if stride is None else int(stride)
        if warmup < 0 or score_length < 1 or stride < 1:
            raise ValueError("warmup >= 0, score_length >= 1, and stride >= 1 are required.")
        last_score = len(self.water) if last_score is None else int(last_score)
        if not 0 <= first_score < last_score <= len(self.water):
            raise ValueError("Invalid scored interval.")
        starts = list(range(max(int(first_score), warmup), last_score - score_length + 1, stride))
        if not starts:
            raise ValueError("No complete windows available for these settings.")
        windows = [(s - warmup, s + score_length) for s in starts]
        def stack(values):
            return np.stack([values[a:b] for a, b in windows]).astype(np.float32)
        masks = {name: stack(mask) for name, mask in self.masks.items()}
        if warmup:
            for mask in masks.values():
                mask[:, :warmup, :] = 0.0
        return HydroBatch(
            water=stack(self.water), drivers=stack(self.drivers),
            targets={name: stack(y) for name, y in self.observations.items()}, masks=masks,
        )

    def continuous_batch(self, start: int = 0, stop: int | None = None, warmup: int = 0) -> HydroBatch:
        stop = len(self.water) if stop is None else int(stop)
        if start < warmup or stop <= start or stop > len(self.water):
            raise ValueError("Invalid interval or insufficient warm-up history.")
        a = start - warmup
        masks = {k: m[a:stop][None, ...].copy() for k, m in self.masks.items()}
        if warmup:
            for m in masks.values():
                m[:, :warmup] = 0
        return HydroBatch(
            water=self.water[a:stop][None], drivers=self.drivers[a:stop][None],
            targets={k: y[a:stop][None] for k, y in self.observations.items()}, masks=masks,
        )
