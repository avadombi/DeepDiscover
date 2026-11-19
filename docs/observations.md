# Observation operators

Hydrological measurements rarely correspond one-to-one to every internal variable of a conceptual model. DeepDiscover separates the simulated physical quantities from the operators that relate them to observations.

## Direct observations

Observed streamflow can be compared with the sum of the water arriving at the model outlet. A configured observation name, for example `streamflow`, identifies the corresponding calibration target.

## Storage-related observations

Groundwater hydraulic head, water-table depth, and soil-moisture representations may be linked to selected model storages through predefined or user-defined operators. An illustrative affine relation is

```{math}
:label: observations-1

\mathrm{H}_i^{(t)}=\mathrm{a}_i\mathrm{s}_i^{(t)}+\mathrm{b}_i,\qquad \mathrm{a}_i>0,
```

where $\mathrm{a}_i$ and $\mathrm{b}_i$ represent an observation scale and offset. These parameters can be specified or estimated when the selected operator permits it.

A fitted head response does not by itself establish that conceptual storage is uniquely determined. The interpretation depends on the observation operator and the information available for calibration.

## Several observation types

DeepDiscover can fit several variables using separate objective contributions and optional weights. Observations with different units may need variance normalization to prevent one numerical scale from dominating the objective.

Missing measurements are excluded through observation masks. Missing physical forcing is a different matter: the user must prepare a valid input series before simulation.

## Custom operators

A custom operator must transform the selected simulated quantities into an output with the dimensions expected by the observation interface. If it contains trainable parameters, its implementation must remain differentiable for gradient-based calibration.
