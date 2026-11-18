# `deepdiscover.api`

Public classes and functions of this module.

## `DeepDiscover`

```python
DeepDiscover(model: MultiDUModel, observations: dict | None=None, snow=None, snow_unit=0, temperature_index=None, driver_channels=None)
```

### `simulate`

```python
simulate(water, drivers, **initial_states)
```

### `predict`

```python
predict(water, drivers, **initial_states)
```

### `fit`

```python
fit(train: HydroBatch, validation: HydroBatch | None=None, *, epochs=100, batch_size=16, learning_rate=0.001, optimizer=None, validation_every=1, loss='mse', weights=None, normalize=True, penalties=(), shuffle=True, seed=42, patience=None, verbose=1, train_initial_states=None, validation_initial_states=None)
```

### `evaluate`

```python
evaluate(batch: HydroBatch, *, loss='mse', weights=None, normalization=None, penalties=(), initial_states=None)
```

### `describe`

```python
describe()
```

### `save_checkpoint`

```python
save_checkpoint(directory, metadata: dict | None=None)
```

### `load_checkpoint`

```python
load_checkpoint(directory)
```
