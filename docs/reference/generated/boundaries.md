# `deepdiscover.boundaries`

Public classes and functions of this module.

## `PrescribedSink`

```python
PrescribedSink(driver_index: int, scale: float=1.0, **kwargs)
```

### `call`

```python
call(drivers, storage, available)
```

### `get_config`

```python
get_config()
```

## `FunctionalSource`

```python
FunctionalSource(layer: keras.layers.Layer, **kwargs)
```

### `call`

```python
call(drivers)
```

## `FunctionalSink`

```python
FunctionalSink(layer: keras.layers.Layer, **kwargs)
```

### `call`

```python
call(drivers, storage, available)
```
