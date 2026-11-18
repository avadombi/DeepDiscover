# `deepdiscover.routing`

Public classes and functions of this module.

## `TrainableRoutingMultiDUModel`

```python
TrainableRoutingMultiDUModel(*args, **kwargs)
```

### `route_weight_matrix`

```python
route_weight_matrix()
```

### `initial_route_weights_numpy`

```python
initial_route_weights_numpy()
```

## `DynamicRoutingMultiDUModel`

```python
DynamicRoutingMultiDUModel(*args, routing_context=('storage', 'inflow', 'flow'), routing_hidden_units=8, routing_storage_ref=100.0, routing_flow_ref=1.0, routing_memory_ref=1.0, routing_seed=42, **kwargs)
```

### `route_weights_for_step`

```python
route_weights_for_step(route_index, source_storage, source_inflow, route_flow, source_memory, drivers=None)
```

### `route_weights_numpy`

```python
route_weights_numpy()
```

### `route_support_weights_numpy`

```python
route_support_weights_numpy()
```
