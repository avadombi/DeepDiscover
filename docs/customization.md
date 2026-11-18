# Custom components

DeepDiscover provides standard neural process networks, physical sources and sinks, observation operators, and routing configurations. Users can extend these elements when the standard options do not represent an application adequately.

## Constitutive neural networks

A custom Keras component can replace a standard process network if it accepts the expected context tensor and returns a **non-negative release propensity** with the required shape. The surrounding DiscoverUnit applies its conservative water-allocation equation.

Directional CRC properties do not automatically carry over to arbitrary custom neural layers. Verify any physical response conditions required by the scientific application.

## Physical boundary components

A custom source adds water to an assigned DU; a custom sink removes water from it. Their outputs must have compatible units, time dimensions and non-negative physical quantities. The model accounts for realized physical exchanges separately from its training objective.

## Observation operators

Custom operators can map physical model states and fluxes into measured quantities. Trainable mappings must remain differentiable to be optimized jointly with the hydrological model.

## Implementation checklist

Check the input and output shapes on a short synthetic example, confirm numerical conservation, verify that gradients are finite, and test a simulation split into consecutive time segments when routing state is involved. Keep custom model components and their versions with the saved experiment.
