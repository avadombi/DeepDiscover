# Defining model topologies

Each release process has one destination. The destination can be a downstream DiscoverUnit or the outlet. Temporal routing is configured separately.

The following two networks contain two DUs and three release processes. Both are admissible in DeepDiscover and represent distinct structural hypotheses.

## Direct outlet pathway and internal transfer

In this configuration, DU 0 releases one process directly to the outlet and a second process to DU 1. DU 1 releases one process to the outlet.

```python
from deepdiscover import CRCNet, DiscoverUnit, MultiDUModel

first_du = DiscoverUnit(
    process_nets=[
        CRCNet([1, 1], [10, 100]),
        CRCNet([1, 1], [10, 100]),
    ],
    context_indices=[(0,), (0,)],
)
second_du = DiscoverUnit(
    process_nets=[CRCNet([1, 1, 1], [10, 100, 10])],
    context_indices=[(0,)],
    upstream_count=1,
)
model = MultiDUModel(
    [first_du, second_du],
    destinations=[["outlet", 1], ["outlet"]],
)
```

## Internal transfer followed by two outlet processes

DU 0 transfers water to DU 1. DU 1 has two processes that both contribute to the outlet.

```python
first_du = DiscoverUnit(
    process_nets=[CRCNet([1, 1], [10, 100])],
    context_indices=[(0,)],
)
second_du = DiscoverUnit(
    process_nets=[
        CRCNet([1, 1, 1], [10, 100, 10]),
        CRCNet([1, 1, 1], [10, 100, 10]),
    ],
    context_indices=[(0,), (0,)],
    upstream_count=1,
)
model = MultiDUModel(
    [first_du, second_du],
    destinations=[[1], ["outlet", "outlet"]],
)
```

The order of process networks matches the order of destinations. The DUs must appear in topological order, and routes to upstream units are not permitted. Both examples use one informational precipitation channel per process; additional drivers, losses, capacities, snow, and temporal routing can be configured separately.

These configurations are candidates within the Multiple-DU structural analyses. Similar predictive scores do not establish the uniqueness of their internal states or fluxes. Compare their optimization adequacy, validation performance, conservation, and storage behaviour using the procedures in [structural comparison](structural_comparison.md). Reserve independent test observations for the final evaluation after model selection.
