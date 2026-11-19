# Structural comparison

A useful model-development question is whether different assumptions about water storage, release processes, and network connections produce meaningfully different hydrological responses. DeepDiscover provides utilities for comparing user-defined candidate structures under a common evaluation procedure.

## Define candidates

Each candidate is a model constructor that returns an independent, freshly initialized model. Candidates may differ in the number of DUs, the process connections, or selected process representations.

For a meaningful comparison, use consistent input data, observation operators, calibration periods, and training budgets unless these are the deliberate subject of the experiment.

## Compare on validation data

Train each candidate using the training observations and compare its performance on a separate validation period. Repeating the procedure with different initializations helps distinguish systematic differences from optimization variability.

The package includes public utilities named `Candidate`, `compare_structures`, and `equivalent_candidates` for setting up and examining declared comparisons. Consult their installed signatures when configuring a particular experiment, because data and model-constructor arguments depend on the package version.

## Examine physical consistency

A small predictive difference is rarely sufficient to interpret a structure. Also examine conservation residuals, storage behavior, physical boundary exchanges, and the complexity of the candidate network.

Model comparisons characterize the configurations that were actually evaluated. They do not prove that the selected topology is the unique physical representation of a catchment.

## Reproducibility

Record the exact package version, random seeds, forcing and observation partitions, objective functions, model constructors, and selected checkpoints. An independent evaluation period should remain separate from structural and hyperparameter selection.
