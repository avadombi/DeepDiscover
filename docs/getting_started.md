# Getting started

A DeepDiscover model begins with a water storage and a set of candidate release processes. The process relationships are learned from observed data through neural networks. The modeller prescribes the number of processes, the variables each process can use, and the destination of each release.

## Water and explanatory variables

Every simulation receives two main arrays:

- **water** gives the water amounts entering individual DiscoverUnits (DUs).
- **drivers** gives the variables used by the neural relationships, such as precipitation and potential evapotranspiration (PET).

These arrays may share a variable, but their physical roles differ. A variable supplied only as a driver does not add or remove water from the model.

For a batch of $\mathrm{B}$ sequences containing $\mathrm{T}$ time steps:

```text
water   : (B, T, number_of_DUs)
drivers : (B, T, number_of_drivers)
```

For example, precipitation can enter the water balance and also inform the constitutive neural networks. PET usually describes atmospheric demand. Any resulting physical water loss must be represented through a declared boundary process.

## Build, simulate, inspect

The [first model tutorial](tutorials/first_model.md) illustrates a complete minimal configuration. It uses a **CRCNet** for a process relationship, a **DiscoverUnit** for storage and conservative release, a **MultiDUModel** to connect model components, and **DeepDiscover** to simulate and evaluate them.

Typical outputs include the modelled outlet discharge, storage trajectories, individual process fluxes, and a numerical water-balance residual. Details and available keys depend on the configured model.

## Continue with several DUs

A downstream unit can receive released water from one or several upstream processes. Connections are defined by **destinations**, one per process. See [Model topologies](topologies.md) for examples of the supported acyclic networks.

## Calibration

For training, create a **HydroDataset** with forcing data and observations, divide the record into temporal windows, and supply the windows to `DeepDiscover.fit`. Warm-up steps evolve the hydrological states while the objective is evaluated only over the scored portion of each window.
