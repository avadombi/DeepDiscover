# Model configuration

A model consists of DiscoverUnits (DUs), the candidate release processes within each unit, and the destinations of the released water. The `MultiDUModel` object groups this physical network, while `DeepDiscover` supplies the simulation, observation, and training interface.

## Defining a unit

A minimal model has one DU and one outlet process:

```python
from deepdiscover import CRCNet, DiscoverUnit, MultiDUModel

release = CRCNet(gamma=[1, -1, 1], x_ref=[5, 3, 100])
unit = DiscoverUnit(
    process_nets=[release],
    context_indices=[(0, 1)],
)
physical_model = MultiDUModel(
    [unit], destinations=[['outlet']]
)
```

The **process_nets** list contains the constitutive networks of the unit. Each tuple in **context_indices** selects the driver channels used by the corresponding process. The neural context also includes physical state information required by the configured implementation.

## Causal Relationship Constraints

The `CRCNet` class assigns the permitted direct response direction of each input through its **gamma** specification. Common values are `+1` for a non-decreasing direct response, `-1` for a non-increasing direct response, and `0` to exclude a direct input. The reference scales **x_ref** are positive constants used in the input transformations.

Selecting directions is part of model specification. The fitted response still depends on the available observations, the input variables and the trainable parameters.

## Interconnected units

For several DUs, pass an ordered collection of units and one destination per process. A destination is either a downstream unit index or `'outlet'`.

If DU 0 has two candidate processes, one going to DU 1 and another to the outlet, its destinations would begin with `[1, 'outlet']`.

The number of declared incoming pathways must agree with the receiving unit's `upstream_count`. Each incoming route supplies context to the downstream DU.

## Storage limits, sources and losses

Each DU can use unbounded storage, a prescribed capacity, or a capacity constrained by learnable bounds. A finite-capacity DU needs an explicit overflow destination. External source and sink components are configured separately from the candidate hydrological release networks.

## Observation interface

The outer `DeepDiscover` object connects the model to the observed variables. For example:

```python
from deepdiscover import DeepDiscover

model = DeepDiscover(
    physical_model,
    observations={'streamflow': 'outlet'},
)
```

Different observation operators can be supplied for additional variables. See [Observation operators](observations.md).
