# Limitations

DeepDiscover models **water quantities**, with observations linked to simulated conserved storages and fluxes.

- A prescribed topology can fit observed data without being uniquely identifiable. Physical labels for latent storages or internal fluxes require supporting information.
- The DU graph is acyclic and uses one destination per process. Cycles, distributed spatial grids and non-hydrological conservation laws require additional models.
- The standard `CRCNet` requires non-negative physical context inputs. Signed predictors need a meaningful transformation and a documented orientation.
- The model works in quantities per regular time step. Rate data and observations at irregular times require explicit conversion or temporal aggregation.
- Default training-window states are zero. Warm-up and suitable initial conditions matter for reservoirs and long-memory pathways.
- Observation masks handle missing targets. Missing physical forcing data must be handled before simulation.
- Dynamic routing weights depend on release-time conditions, with finite lag support fixed at construction.
- For physical sinks with several competing demands, requests are reduced proportionally. Users should confirm that this allocation represents their intended physical process.
- Training custom constitutive networks may reproduce predictions without satisfying CRC signs. The user must test monotonicity separately.
- Checkpoints store parameter arrays but not automatically executable Python definitions of customized components.
- The package does not predict precipitation or meteorological forcings by itself. Such forecasts require an upstream forecasting model.
- Isotope, tracer and water-age transport are not included in the quantity model.
- GPU, multi-worker and platform-specific runtime performance should be verified for each environment.
