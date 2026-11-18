# `deepdiscover.snow`

Public classes and functions of this module.

## `DifferentiableDegreeDaySnow`

```python
DifferentiableDegreeDaySnow(snowfall_temperature=-1.5, melt_temperature=1.5, degree_day_factor=2.5, snowfall_temperature_bounds=(-3.0, 1.0), melt_temperature_bounds=(-1.0, 4.0), degree_day_factor_bounds=(0.05, 8.0), transition_sharpness=5.0, trainable_snowfall_temperature=True, trainable_melt_temperature=True, trainable_degree_day_factor=True, name=None)
```

### `build`

```python
build(input_shape)
```

### `snowfall_temperature`

```python
snowfall_temperature()
```

### `melt_temperature`

```python
melt_temperature()
```

### `degree_day_factor`

```python
degree_day_factor()
```

### `parameter_values`

```python
parameter_values()
```

### `call`

```python
call(inputs, initial_storage=None)
```

## `apply_snow`

```python
apply_snow(snow, precipitation, temperature=None, initial_storage=None)
```
