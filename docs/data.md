# Data and units

The model receives non-negative **water quantities per time step** and exogenous **informational predictors**.

| Variable | Shape | Meaning |
| --- | --- | --- |
| `water` | `(B,T,n_DU)` | External conserved water entering each DU per time step |
| `drivers` | `(B,T,n_drivers)` | Predictors used in networks and boundary exchanges |
| `storage` | `(B,T,n_DU)` | Water remaining in each DU after the step |
| `process_fluxes` | `(B,T,n_processes)` | Quantities released by learned constitutive processes |
| `route_arrivals` | `(B,T,n_routes)` | Water arriving at the destination after routing |
| `transit` | `(B,T,n_routes)` | Water still travelling on each route |
| `memory` | `(B,T,n_DU)` | Optional informational state, not a water quantity |
| `observations[name]` | `(B,T,n_y)` | Measured variable used in fitting |

`B` is the number of independent windows in a batch; `T` is time within each window. No loop over batch elements is required by the core. Time-varying units must be converted to amounts for the model step. For example, a discharge rate can be converted to an equivalent depth per time step using catchment area.

`HydroDataset` accepts a continuous `(T, channels)` record, masks missing observations and checks regular datetime sampling if dates are supplied. `make_windows(warmup=..., score_length=..., stride=...)` returns `HydroBatch`. Warm-up time is simulated with zero observation masks.

Use `first_score` and `last_score` to restrict training windows to the desired training period. Validation and test periods should be separated chronologically and must not influence model selection before their intended use. Predictors and water inputs cannot contain missing values; imputation is an explicit user decision.

CRC inputs must be **non-negative**. For signed temperatures, provide a physically meaningful non-negative transform as a separate driver, or use temperature only in the snow module. Do not apply an arbitrary offset without documenting its interpretation.

## Long sequences and state continuation

To split a continuous run into parts, pass the first segment's `final_storage`, `final_memory` and `final_transit_state` into the second segment. With snow, also pass `snow_final_snow_storage`. Training windows start from the user-provided state or zero by default; a sufficiently long warm-up is necessary to limit state-initialization effects.
