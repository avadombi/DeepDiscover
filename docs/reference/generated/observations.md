# `deepdiscover.observations`

Public classes and functions of this module.

## `DirectObservation`

```python
DirectObservation(key='outlet', index=None, **kwargs)
```

### `call`

```python
call(physical_output)
```

### `get_config`

```python
get_config()
```

## `StorageToDepth`

```python
StorageToDepth(du_index=0, reference_depth=10.0, scale=1.0, **kwargs)
```

### `call`

```python
call(physical_output)
```

## `StorageToMoisture`

```python
StorageToMoisture(du_index=0, capacity=100.0, residual=0.0, saturated=1.0, **kwargs)
```

### `call`

```python
call(physical_output)
```
