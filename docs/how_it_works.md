# How DeepDiscover works

DeepDiscover builds conceptual hydrological models from water-conserving components. Each DiscoverUnit stores water, represents possible processes that release it, and passes released water to a declared destination. Constitutive neural networks determine the strength of each process response from the available information.

## 1. Water and information are different inputs

A DU receives external water and, where applicable, transfers from upstream units. Explanatory variables inform its neural processes. The model keeps these roles explicit so that the water balance remains interpretable.

```{figure} _static/figures/Fig1.svg
:alt: DiscoverUnit water balance with incoming water, physical losses, process releases and retained storage.
:width: 95%
:align: center
:class: dd-science-figure

**Figure 1.** Water balance of a DiscoverUnit. The schematic distinguishes conserved water exchanges from information supplied to the constitutive networks.
```

## 2. The amount of water available is explicit

For DU $i$, the pre-loss water amount is

```{math}
:label: how-it-works-1

\mathrm{A}_i^{(t)} = \mathrm{s}_i^{(t-1)} + \mathrm{I}_i^{(t)}.
```

Here $\mathrm{s}_i^{(t-1)}$ is the stored water from the preceding time step and $\mathrm{I}_i^{(t)}$ combines assigned external inputs and incoming transfers. Physical boundary losses are removed before process allocation, leaving an available amount $\mathrm{B}_i^{(t)}$.

## 3. Neural networks learn process propensities

Each candidate process has a neural relationship that produces a non-negative **propensity** $\mathrm{h}_{ij}^{(t)}$. This quantity describes the strength of the process response under the supplied conditions. It is not yet a physical water flux.

```{figure} _static/figures/Fig2.svg
:alt: Constitutive neural networks using explanatory variables under Causal Relationship Constraints.
:width: 95%
:align: center
:class: dd-science-figure

**Figure 2.** Learned constitutive relationships and Causal Relationship Constraints (CRC). The inputs and prescribed response directions are kept distinct from the learned network parameters.
```

The CRC signs prescribe the direct response direction of selected inputs. A positive sign indicates a non-decreasing direct response, and a negative sign a non-increasing direct response, holding the remaining inputs fixed.

## 4. A conservative rule allocates the water

The learned propensities determine the fraction of the available water assigned to each process:

```{math}
:label: how-it-works-2

 \mathrm{u}_{ij}^{(t)}=\mathrm{B}_i^{(t)}\frac{\mathrm{h}_{ij}^{(t)}}{1+\sum_{k=1}^{\mathrm{n}_i}\mathrm{h}_{ik}^{(t)}}.
```

The remaining amount before any storage overflow is

```{math}
:label: how-it-works-3

 \mathrm{s}_{i,\mathrm{raw}}^{(t)}=\frac{\mathrm{B}_i^{(t)}}{1+\sum_{k=1}^{\mathrm{n}_i}\mathrm{h}_{ik}^{(t)}}.
```

```{figure} _static/figures/Fig3.svg
:alt: Conservative allocation between water released through candidate processes and water remaining in storage.
:width: 95%
:align: center
:class: dd-science-figure

**Figure 3.** Conservative water allocation. Each release and the retained storage are calculated from the same available resource.
```

The additional unit in the denominator represents the option of retaining water in storage. A finite capacity, when prescribed, may cause overflow through a separate declared route.

## 5. Released water follows declared routes

Each release process has one destination: another DU or the outlet. A routing kernel determines when the released water arrives. Delayed water remains accounted for in the model until its arrival.

## 6. Calibration connects the model to observations

Observation operators map simulated quantities, such as outlet discharge or selected storage states, to quantities that can be compared with measurements. Gradient-based optimization estimates the trainable relationships using the configured observations and loss functions.

## What to read next

The [scientific formulation](scientific_formulation.md) gives the governing quantities and equations. [Model configuration](model_configuration.md) shows how to define more than one storage, while [Training and evaluation](training_and_evaluation.md) explains calibration and diagnostics.
