# `deepdiscover.dataset`

Public classes and functions of this module.

## `HydroBatch`

### `subset`

```python
subset(indices)
```

### `batch_size`

```python
batch_size()
```

### `length`

```python
length()
```

## `HydroDataset`

```python
HydroDataset(water, drivers, observations: Mapping[str, np.ndarray], masks=None, time=None)
```

### `make_windows`

```python
make_windows(warmup: int, score_length: int, stride: int | None=None, first_score: int=0, last_score: int | None=None)
```

### `continuous_batch`

```python
continuous_batch(start: int=0, stop: int | None=None, warmup: int=0)
```
