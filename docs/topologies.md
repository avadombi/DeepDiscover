# Defining model topologies

The number of DiscoverUnits (DUs) and release processes is configurable. Each release process has exactly one destination: a downstream DU or the outlet. Multiple processes may reach the same DU or outlet. Temporal routing is specified separately from the network connections.

The first two configurations contain two DUs and three release processes. They represent distinct structural hypotheses, not a limit on the number of DUs or the complexity of the network.

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

## Four-DU serial cascade

This network passes water from DU 0 through three downstream DUs before it reaches the outlet:

`DU 0 -> DU 1 -> DU 2 -> DU 3 -> outlet`

The following helper creates a DU with one precipitation driver, its beginning-of-step storage, and one context variable for each incoming transfer. All four DUs have one release process in this example.

```python
from deepdiscover import CRCNet, DiscoverUnit, MultiDUModel


def make_unit(n_incoming, n_processes=1):
    n_context = 2 + n_incoming  # precipitation, storage, incoming transfers
    return DiscoverUnit(
        process_nets=[
            CRCNet([1] * n_context, [5.0, 100.0] + [5.0] * n_incoming)
            for _ in range(n_processes)
        ],
        context_indices=[(0,)] * n_processes,
        upstream_count=n_incoming,
    )


units = [make_unit(0), make_unit(1), make_unit(1), make_unit(1)]
model = MultiDUModel(
    discover_units=units,
    destinations=[[1], [2], [3], ["outlet"]],
)
```

Additional DUs can be appended to the same pattern without changing the governing conservation equations. The computational cost still grows with the number of DUs, processes, and routing states.

## Four-DU branching network

The model also supports converging branches:

```text
          -> DU 1 -
DU 0 -<              >- DU 3 -> outlet
          -> DU 2 -
```

Using the `make_unit` function above:

```python
units = [
    make_unit(0, n_processes=2),  # one transfer to each branch
    make_unit(1),                 # DU 1 receives one transfer
    make_unit(1),                 # DU 2 receives one transfer
    make_unit(2),                 # DU 3 receives two transfers
]
model = MultiDUModel(
    discover_units=units,
    destinations=[[1, 2], [3], [3], ["outlet"]],
)
```

Here, DU 3 must declare `upstream_count=2`, since it receives two distinct release processes. Its CRC network needs four context variables: the selected driver, storage, and the two incoming routed components. Each incoming **route** contributes one component, including routes that originate in the same upstream DU.

For either four-DU configuration, physical water inputs have shape `(B, T, 4)`, where `B` is the batch size and `T` is the number of time steps. For example, if precipitation enters only DU 0, the other three input columns are zero. The shared driver array can independently contain precipitation, PET, and other explanatory variables. Inputs to a DU need not match its process-driver channels.

## Topological constraints and interpretation

The order of the process networks in each DU matches the order of their destinations. The DUs are indexed from zero and must be supplied in topological order: an internal transfer from DU `i` can only lead to DU `j` if `j > i`. The model therefore supports directed acyclic networks; feedback transfers from downstream to upstream DUs are not part of this topology interface.

The `upstream_count` of each DU must equal the number of incoming routed components, including any capacity-overflow routes. The CRC networks must have the corresponding number of context variables. Additional drivers, prescribed sources, boundary losses, finite capacities, snow, and temporal routing can be configured separately. Temporal routing can delay transfers but does not create topological feedback.

These configurations are candidate hydrological structures. Similar predictive scores do not establish the uniqueness of their internal states or fluxes. Compare optimization adequacy, validation performance, conservation, and storage behaviour following [structural comparison](structural_comparison.md). Reserve independent test observations for final evaluation after model selection.

