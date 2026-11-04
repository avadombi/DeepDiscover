# Temporal routing

Each process releases water from one DU and has exactly one declared destination. The route controls the time of arrival at this destination. All routes transport conserved water.

| Route type | Class or configuration | Description |
| --- | --- | --- |
| Instantaneous | `routing=None` | Entire release arrives in the same step |
| Fixed | `RoutingKernel([0.2,0.5,0.3])` | Prescribed fractions over delays 0,1,2 |
| Learned, time-invariant | `TrainableRoutingMultiDUModel` | One trainable softmax distribution per route |
| Dynamic | `DynamicRoutingMultiDUModel` | A separate state-dependent distribution at each release time |

The `routing` argument is a nested list with the same structure as `destinations`. The optional `overflow_routing` list configures overflow propagation. No new physical water is generated during routing.

```python
from deepdiscover import RoutingKernel, TrainableRoutingMultiDUModel
model = TrainableRoutingMultiDUModel(
    discover_units=[du0, du1],
    destinations=[[1], ["outlet"]],
    routing=[[RoutingKernel([0.4, 0.6])], [RoutingKernel([0.2, 0.3, 0.5])]],
)
```

For dynamic routing, use `DynamicRoutingMultiDUModel` with the same core configuration. Its default inputs are beginning-of-step storage, incoming water and the current process release. Optional informational memory is also available. Dynamic kernel fractions are evaluated when water is released; they remain attached to each released amount during propagation.

Delayed water is explicitly represented by `transit` and `final_transit_state`. The complete balance includes this transit quantity. A route with only one lag is always instantaneous, including in the dynamic class.

The DU graph is presently a directed acyclic graph. Index DUs in topological order. A route from DU 2 to DU 0 is rejected. Temporal kernels are measured in **integer model steps**; they do not imply a physical time unit by themselves.
