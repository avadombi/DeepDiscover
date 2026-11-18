# Physical sources, sinks and storage limits

A DeepDiscover model may receive water through the main **water** input or through additional source components assigned to individual DiscoverUnits. Physical sources and sinks appear in the water balance; explanatory drivers do not.

## Sources

A source may be prescribed from a time series or implemented through a user-defined component. Examples include lateral inflow and managed recharge. Its units and timing must match those used for the rest of the model.

When a physical input enters DU $i$, it contributes to that DU's incoming water $\mathrm{I}_i^{(t)}$ before the local release and retention calculations.

## Sinks

A sink removes water from the modelled system. An atmospheric demand-limited sink, for example, cannot remove more water than is physically available. The realized flux is reported separately from the model's discharge-generating processes.

When multiple sinks operate on the same available resource, the requested losses are reduced if their total exceeds available water. Their realized amounts are included in the conservation accounting.

## Fixed capacity and overflow

A storage limit can be provided with the corresponding constructor parameter, for example `smax=100.0`, or set as a learnable value constrained by prescribed bounds, for example `smax_bounds=(50.0, 200.0)`.

For a bounded DU, its overflow destination must be specified in `overflow_destinations`. Overflow can be routed through its own temporal kernel. When overflow enters a downstream DU, the receiving unit needs the corresponding incoming context channel.

## Diagnostics

Inspect individual source and sink fluxes together with the storage change and process releases. The numerical water-balance residual checks the consistency of the simulated quantities. A small residual confirms numerical conservation, not necessarily the physical validity of the selected process representation.
