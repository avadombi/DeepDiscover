# Physical water sources and losses

DeepDiscover distinguishes **boundary exchanges** from **training loss functions**. A boundary exchange changes the physical water balance and must be accounted for in the simulated system.

## Sources

`water` supplies the external conserved inputs directly to DUs. Additional sources can be specified in a DU's `exchanges` list:

- `PrescribedSource(driver_index, scale=1)` adds a non-negative amount from a driver column.
- `FunctionalSource(my_layer)` adds a non-negative amount computed by a differentiable Keras layer.
- A custom `keras.layers.Layer` declaring `kind = "source"` can implement the same `call(drivers)` contract.

Applications include precipitation, prescribed recharge, irrigation, lateral inflow and water transferred into the modeled domain. A physical source cannot be silently included in both `water` and `exchanges`, as that would count the same water twice.

## Losses

- `PrescribedSink(driver_index, scale=1)` requests an externally specified amount, such as pumping or managed extraction.
- `DemandLimitedSink(efficiency_net, demand_index, context_indices)` requests a loss bounded by demand and an efficiency between zero and one.
- `FunctionalSink(my_layer)` uses a custom layer returning a non-negative requested amount.
- A custom `keras.layers.Layer` with `kind = "sink"` implements `call(drivers, storage, available)`.

Loss requests from the same DU are combined. If demand exceeds water availability, each request is reduced by the same fraction. Simultaneous evaporation and groundwater pumping are therefore admissible and conserve water.

## Example

```python
from deepdiscover import (
    CRCNet, DiscoverUnit, PrescribedSource, PrescribedSink,
    DemandLimitedSink,
)

et_efficiency = CRCNet(gamma=[1, 1], x_ref=[10, 100], output="sigmoid")
du = DiscoverUnit(
    process_nets=[CRCNet([1, 1], [10, 100])],
    context_indices=[(0,)],
    exchanges=[
        PrescribedSource(driver_index=2),            # lateral inflow
        DemandLimitedSink(et_efficiency, demand_index=1, context_indices=[0]),
        PrescribedSink(driver_index=3),              # pumping
    ],
)
```

Here the driver columns represent precipitation information, PET, lateral inflow and pumping request. The user must decide which entries are conserved in `water`, and which are introduced as boundary sources. This choice determines the physical system boundary.

## Requirements for custom exchanges

Sources and sinks return one non-negative `(B,1)` tensor in quantity-per-step units. Non-finite or negative values raise errors. Custom sinks return **requested losses**; the DU applies the available-water constraint. These exchanges may carry trainable Keras parameters. If they introduce water from a separate physical reservoir outside the graph, that reservoir is not included in the system storage balance.
