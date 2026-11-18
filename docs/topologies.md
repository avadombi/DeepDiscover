# Model topologies

DeepDiscover accepts a configurable number of DiscoverUnits and candidate release processes. Water transfers can form serial cascades, parallel branches, or general directed acyclic networks. Each process has exactly one declared destination.

```{figure} _static/figures/Fig4.svg
:alt: Examples of serial and branching connections between DiscoverUnits.
:width: 95%
:align: center
:class: dd-science-figure

**Figure 4.** Examples of connections between DiscoverUnits. An outlet destination releases water from the represented network; an internal destination transfers it to another DU.
```

## Three DiscoverUnits in series

The model below illustrates the destination list for a serial topology:

```python
# Each element of this list gives the destination of one DU's processes.
destinations = [[1], [2], ['outlet']]
```

DU 0 releases water to DU 1, DU 1 to DU 2, and DU 2 to the outlet. Each unit must be constructed with a compatible number of release processes, neural inputs and incoming-transfer channels. The downstream units each receive one upstream route.

## Four DiscoverUnits with a branching connection

An upstream unit can have two distinct processes feeding different destinations:

```python
destinations = [
    [1, 2],       # DU 0 has two release processes
    [3],          # DU 1 contributes to DU 3
    [3],          # DU 2 also contributes to DU 3
    ['outlet'],   # DU 3 contributes to the outlet
]
```

The first DU requires two process networks. The final DU has two incoming routes and needs a compatible upstream context configuration. A single process is not assigned multiple destinations under this convention; branches are represented by separate processes.

## Declared model structure

The topology is specified before calibration. DeepDiscover can be used to compare distinct, user-defined candidate structures under common training and validation conditions. The software does not assign a unique physical meaning to a topology solely from its predictive performance.

See [Model configuration](model_configuration.md) for the DU interfaces and [Structural comparison](structural_comparison.md) for the model comparison workflow.
