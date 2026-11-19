# Scientific formulation

This page describes the water-quantity accounting implemented by the DeepDiscover framework. It focuses on model construction and interpretation. For the published methodological basis, see [Adombi (2026)](https://doi.org/10.1016/j.jhydrol.2026.135249).

## A network of DiscoverUnits

Physical variables are set in upright letters. Indices such as $i$ and $j$, and the time index $t$, are italic. Storage uses a lowercase upright **s**. Time is denoted by a parenthesized superscript, for example $\mathrm{s}_i^{(t)}$ for the storage at step $t$ and $\mathrm{s}_i^{(t-1)}$ for the preceding step. Subscripts identify DiscoverUnits and candidate processes.

Each DiscoverUnit (DU) represents a conceptual water storage connected to possible release processes. The connections form a directed acyclic network. Each process is assigned one destination: another DU or the system outlet.

Let $\mathrm{s}_i^{(t-1)}$ denote the water retained by DU $i$ at the end of the previous step, and $\mathrm{I}_i^{(t)}$ the water arriving during the current step. The amount available before external losses is

```{math}
:label: scientific-formulation-1

\mathrm{A}_i^{(t)}=\mathrm{s}_i^{(t-1)}+\mathrm{I}_i^{(t)}.
```

The incoming water may include external input and transfers from other DUs. Units must be consistent throughout the model. For catchment-average simulations, water is commonly expressed as an equivalent depth.

## Boundary losses

A physical sink requests a non-negative loss $\mathrm{L}_{i,\mathrm{request}}^{(t)}$ from the water available at a DU. One demand-limited example is

```{math}
:label: scientific-formulation-2

\mathrm{L}_{i,\mathrm{request}}^{(t)}=\eta_i^{(t)}\min\left[\mathrm{D}_i^{(t)},\mathrm{A}_i^{(t)}\right],
\qquad 0\le\eta_i^{(t)}\le1.
```

Here $\mathrm{D}_i^{(t)}$ is an external demand, and the coefficient $\eta_i^{(t)}$ may depend on hydrological conditions. When several losses compete for the same limited resource, the total demand is capped by available water. Let $\mathrm{L}_i^{(t)}$ denote the resulting total realized physical loss. The water remaining for release and storage is

```{math}
:label: scientific-formulation-3

\mathrm{B}_i^{(t)}=\mathrm{A}_i^{(t)}-\mathrm{L}_i^{(t)},\qquad \mathrm{B}_i^{(t)}\ge0.
```

The loss $\mathrm{L}_i$ is a physical flux. A training objective called a *loss function* is a separate numerical quantity.

## Neural constitutive relationships

For a DU with $\mathrm{n}_i$ candidate release processes, process $j$ has a non-negative propensity

```{math}
:label: scientific-formulation-4

\mathrm{h}_{ij}^{(t)}=\mathcal{N}_{ij}\left(\mathbf{c}_{ij}^{(t)};\boldsymbol{\theta}_{ij}\right)\ge0.
```

The context $\mathbf{c}_{ij}^{(t)}$ may contain chosen meteorological variables, antecedent storage, and selected upstream transfers. The trainable parameters $\boldsymbol{\theta}_{ij}$ determine the response within the imposed architecture and Causal Relationship Constraints.

## Conservative allocation

The physical release of process $j$ is

```{math}
:label: scientific-formulation-5

\mathrm{u}_{ij}^{(t)}=\mathrm{B}_i^{(t)}\frac{\mathrm{h}_{ij}^{(t)}}{1+\sum_{k=1}^{\mathrm{n}_i}\mathrm{h}_{ik}^{(t)}}.
```

Water retained before applying a finite storage capacity is

```{math}
:label: scientific-formulation-6

\mathrm{s}_{i,\mathrm{raw}}^{(t)}=\frac{\mathrm{B}_i^{(t)}}{1+\sum_{k=1}^{\mathrm{n}_i}\mathrm{h}_{ik}^{(t)}}.
```

Consequently,

```{math}
:label: scientific-formulation-7

\mathrm{s}_{i,\mathrm{raw}}^{(t)}+\sum_{j=1}^{\mathrm{n}_i}\mathrm{u}_{ij}^{(t)}=\mathrm{B}_i^{(t)}.
```

This identity follows directly from the allocation rule. It does not depend on the fitted values of the network weights.

## Storage capacity and overflow

If a capacity $\mathrm{s}_{i,\max}$ is provided, the retained water becomes

```{math}
:label: scientific-formulation-8

\mathrm{s}_i^{(t)}=\min\left(\mathrm{s}_{i,\mathrm{raw}}^{(t)},\mathrm{s}_{i,\max}\right),
```

and excess water is released through an explicit overflow path:

```{math}
:label: scientific-formulation-9

\mathrm{o}_i^{(t)}=\max\left(\mathrm{s}_{i,\mathrm{raw}}^{(t)}-\mathrm{s}_{i,\max},0\right).
```

The local balance is then

```{math}
:label: scientific-formulation-10

\mathrm{s}_i^{(t)}+\sum_{j=1}^{\mathrm{n}_i}\mathrm{u}_{ij}^{(t)}+\mathrm{o}_i^{(t)}+\mathrm{L}_i^{(t)}=\mathrm{A}_i^{(t)}.
```

An unbounded DU has no capacity-related overflow. For a learned capacity, its prescribed bounds delimit the admissible value.

## Process destinations and temporal routing

Each release enters a declared route. For a prescribed kernel $\mathrm{g}_e(\tau)$ with non-negative weights summing to one, the arrival at the destination may be written as

```{math}
:label: scientific-formulation-11

\mathrm{F}_e^{(t)}=\sum_{\tau=0}^{\mathrm{L}_e}\mathrm{g}_e(\tau)\mathrm{x}_e^{(t-\tau)},
\qquad \mathrm{g}_e(\tau)\ge0,\quad \sum_{\tau=0}^{\mathrm{L}_e}\mathrm{g}_e(\tau)=1,
```

where $\mathrm{x}_e^{(t)}$ is the release entering route $e$. The water still in transit is explicitly accounted for. An instantaneous route uses a single unit weight at zero delay.

## Global balance and observation

The change in total modelled water storage, including the water held inside routing pathways and any configured upstream snow storage, is balanced by incoming water, physical losses, and water arriving at the system outlet. Internal transfers cancel when the complete network is considered.

Observed discharge, groundwater level, and other measured variables are connected to the simulated state through observation operators. These operators affect the quantities compared in calibration, without changing the water-conservation identities above.
