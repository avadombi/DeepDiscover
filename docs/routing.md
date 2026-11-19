# Temporal routing

A process describes how much water leaves a DiscoverUnit. Its destination specifies where the water goes. Temporal routing describes when it arrives. These are independent choices in DeepDiscover.

```{figure} _static/figures/Fig5.svg
:alt: Instantaneous and delayed routes, with observed quantities linked through observation operators.
:width: 95%
:align: center
:class: dd-science-figure

**Figure 5.** Routing and observation operators. Water released by a process can arrive immediately or after a distributed delay; simulated quantities are subsequently related to measurements through selected operators.
```

## Instantaneous transfer

In instantaneous routing, a released amount is delivered to its destination within the same model time step. This is suitable for model structures in which no explicit delay is required along that connection.

## Prescribed temporal routing

A routing kernel allocates each release across a finite set of possible arrival delays. Its weights are non-negative and sum to one:

```{math}
:label: routing-1

 \mathrm{g}_e(\tau)\ge0,\qquad \sum_{\tau=0}^{\mathrm{L}_e}\mathrm{g}_e(\tau)=1.
```

For a time-invariant kernel, the arrival quantity is

```{math}
:label: routing-2

 \mathrm{F}_e^{(t)}=\sum_{\tau=0}^{\mathrm{L}_e}\mathrm{g}_e(\tau)\mathrm{x}_e^{(t-\tau)}.
```

Released water that has not yet arrived remains in the routing state. The remaining quantity is part of the total system storage for numerical conservation.

## Learned routing

The package supports routing kernels whose parameters can be estimated during calibration. It also provides state-dependent kernels, where the weight distribution is evaluated from the conditions at the time a quantity of water enters the route.

The interpretation of a learned delay depends on the selected model time step and the routing architecture.

## Continuing a simulation

If a long record is simulated in consecutive segments, the second segment must receive the ending storage and routing states from the first. Starting the second segment with empty transit states can change its outlet response when a delayed route is present.

See [Scientific formulation](scientific_formulation.md) for the conservation equations and [Model configuration](model_configuration.md) for destination declarations.
