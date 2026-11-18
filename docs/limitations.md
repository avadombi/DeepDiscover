# Model limitations

DeepDiscover provides a configurable computational framework. Its outputs require the same scientific scrutiny as those of any conceptual hydrological model.

The main points to consider are:

- Candidate processes, explanatory variables, observation operators, and destinations are selected by the modeller. These choices define the representations explored by calibration.
- Physical conservation restricts admissible water exchanges, but does not establish that each learned process corresponds uniquely to a real hydrological mechanism.
- Current DU connections are represented through directed acyclic networks. Cyclic exchanges and distributed spatial grids are outside this interface.
- The properties imposed by the standard CRCNet architecture are not automatically guaranteed when a custom neural component is substituted.
- A model may reproduce the observed discharge well while representing internal storages and fluxes differently from other acceptable models.
- Numerical conservation diagnostics verify bookkeeping consistency. They do not replace evaluation against independent hydrological evidence.
- Execution time depends on the number of DUs, neural processes, sequence length, routing configuration, and computational platform.

A useful model assessment therefore combines predictive performance, physical diagnostics, and a transparent account of the candidate structures and observations used for calibration.
