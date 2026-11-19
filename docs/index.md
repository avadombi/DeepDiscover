# DeepDiscover

<div class="dd-subtitle">Hydrological model discovery with constrained neural networks</div>

<p class="dd-lede">DeepDiscover is an open-source Python package for constructing, calibrating, and inspecting data-driven conceptual hydrological models. The modeller defines candidate water stores, possible release processes, and their connections. Neural networks learn the associated constitutive relationships under Causal Relationship Constraints, while each DiscoverUnit conserves water explicitly.</p>

## Install

```console
python -m pip install deepdiscover-hydro
```

DeepDiscover supports Python 3.11 and 3.12. See [Installation](installation.md) for environment setup and optional notebook dependencies.

## Begin with a model

- [Getting started](getting_started.md) introduces the inputs, storages, process releases, and simulation outputs.
- [Your first DiscoverUnit](tutorials/first_model.md) builds a small executable Python example.
- [How DeepDiscover works](how_it_works.md) follows water from an input to an outlet, with diagrams and equations.
- [Model configuration](model_configuration.md) describes networks with several DiscoverUnits.
- [Scientific formulation](scientific_formulation.md) defines the quantity balance and routing conventions.

## Learn more

The [model topologies](topologies.md) page illustrates connected and branched networks. [Routing](routing.md) describes instantaneous and delayed transfers, and [Observation operators](observations.md) explains how simulated quantities can be compared with measurements. The [Python API](reference/index.md) gives an overview of the public interfaces.

Source code: [GitHub](https://github.com/avadombi/DeepDiscover). Published package: [PyPI](https://pypi.org/project/deepdiscover-hydro/).

```{toctree}
:caption: Getting started
:maxdepth: 2
:hidden:

installation
getting_started
tutorials/first_model
```

```{toctree}
:caption: Concepts and equations
:maxdepth: 2
:hidden:

how_it_works
scientific_formulation
boundaries
routing
observations
limitations
```

```{toctree}
:caption: Model development
:maxdepth: 2
:hidden:

model_configuration
topologies
data
training_and_evaluation
structural_comparison
customization
camels
```

```{toctree}
:caption: Reference
:maxdepth: 2
:hidden:

reference/index
```
