# `deepdiscover.model`

Public classes and functions of this module.

## `PositiveDense`

```python
PositiveDense(units: int, activation='linear', bias_init=0.0, **kwargs)
```

### `build`

```python
build(input_shape)
```

### `call`

```python
call(x)
```

## `scale_nonnegative`

```python
scale_nonnegative(context, x_ref)
```

## `CRCNet`

```python
CRCNet(gamma, x_ref, hidden_units=16, hidden_activation='tanh', output='softplus', name=None)
```

### `signed_context`

```python
signed_context(context)
```

### `call`

```python
call(context)
```

## `PrescribedSource`

```python
PrescribedSource(driver_index, scale=1.0, name=None)
```

### `call`

```python
call(drivers)
```

## `DemandLimitedSink`

```python
DemandLimitedSink(efficiency_net, demand_index, context_indices=(), name=None)
```

### `call`

```python
call(drivers, storage, available)
```

## `conservative_allocation`

```python
conservative_allocation(resource, propensities, smax=None)
```

## `InformationalMemoryCell`

```python
InformationalMemoryCell(driver_indices, gamma, x_ref, hidden_units=8, name=None)
```

### `call`

```python
call(memory, drivers, storage)
```

## `DiscoverUnit`

```python
DiscoverUnit(process_nets, smax=None, smax_bounds=None, context_indices=None, upstream_count=0, exchanges=None, memory_cell=None, include_deficit=False, name=None)
```

### `build`

```python
build(input_shape)
```

### `smax`

```python
smax()
```

### `has_memory`

```python
has_memory()
```

### `bounded`

```python
bounded()
```

### `call`

```python
call(storage, inflow, contexts, drivers=None)
```

### `step`

```python
step(storage, inflow, drivers, upstream=None, memory=None)
```

### `run_sequence`

```python
run_sequence(inflow, drivers, initial_storage, initial_memory=None)
```

## `StorageToHead`

```python
StorageToHead(du_index, scale=1.0, offset=0.0, name=None)
```

### `call`

```python
call(physical_output)
```

## `TrainableStorageToHead`

```python
TrainableStorageToHead(du_index, scale_init=1.0, offset_init=0.0, learn_scale=True, learn_offset=True, name=None)
```

### `build`

```python
build(input_shape)
```

### `scale`

```python
scale()
```

### `call`

```python
call(physical_output)
```

## `observe`

```python
observe(physical_output, observation=None, target='outlet')
```

## `RoutingKernel`

```python
RoutingKernel(weights)
```

### `max_lag`

```python
max_lag()
```

### `mean_delay`

```python
mean_delay()
```

## `MultiDUModel`

```python
MultiDUModel(discover_units, destinations, overflow_destinations=None, routing=None, overflow_routing=None, name=None)
```

### `mark_built_for_checkpoint`

```python
mark_built_for_checkpoint()
```

### `route_weight_matrix`

```python
route_weight_matrix()
```

### `route_weights_numpy`

```python
route_weights_numpy()
```

### `route_weights`

```python
route_weights()
```

### `route_support_weights_numpy`

```python
route_support_weights_numpy()
```

### `route_weights_for_step`

```python
route_weights_for_step(route_index, source_storage, source_inflow, route_flow, source_memory, drivers=None)
```

### `run_sequence`

```python
run_sequence(external_inflow, drivers, initial_storage, initial_transit=None, initial_memory=None)
```
