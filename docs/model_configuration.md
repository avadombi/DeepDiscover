# Model configuration

A DU contains one or more CRC networks. One list of destination names is provided for each DU. The number of destinations must match the number of processes in that DU.

```python
from deepdiscover import CRCNet, DiscoverUnit, MultiDUModel

# Driver columns are [precipitation, PET].
du0 = DiscoverUnit(
    process_nets=[
        CRCNet([1, -1, 1], [5, 3, 100]),
        CRCNet([1, -1, 1], [5, 3, 100]),
    ],
    context_indices=[(0, 1), (0, 1)],
)

du1 = DiscoverUnit(
    process_nets=[CRCNet([1, -1, 1, 1], [5, 3, 100, 5])],
    context_indices=[(0, 1)],
    upstream_count=1,
)

model = MultiDUModel(
    discover_units=[du0, du1],
    destinations=[["outlet", 1], ["outlet"]],
)
```

One process from `du0` goes directly to the outlet. Another transfers water to `du1`, where it becomes an incoming component. Each DU uses one storage. The incoming-component count is explicit in the constitutive context of the receiving DU.

## Fixed capacity and overflow

Pass `smax=100.0` for a fixed storage limit or `smax_bounds=(50.0, 200.0)` for a learned capacity. A bounded DU requires an overflow destination in `overflow_destinations`. Overflow can use its own temporal kernel. Include an additional upstream context channel for an overflow route entering a downstream DU.

## Informational memory and deficit

Set `memory_cell=InformationalMemoryCell(...)` to add an optional memory state. Use `include_deficit=True` only when the DU has a finite storage capacity. The process context order is:

1. Selected drivers in the declared order.
2. Beginning-of-step storage.
3. Optional memory.
4. Optional deficit `Smax - S`.
5. Incoming routed components, in route order.

The `gamma` and `x_ref` vectors must match this order. Use a negative orientation for a non-increasing influence.

## Custom constitutive networks

A process can be a Keras layer that returns a non-negative `(B,1)` propensity. DeepDiscover checks the non-negativity and the conservation equations. CRC monotonicity is guaranteed only by the standard `CRCNet` or by a user implementation with independently verified monotonicity.
