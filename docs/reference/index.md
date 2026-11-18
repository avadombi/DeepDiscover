# Python API reference

This page identifies the main public software interfaces in DeepDiscover 1.0.1. It is an overview, not a substitute for the installed Python signatures or the release-specific source code.

| Component | Purpose |
| --- | --- |
| `CRCNet` | Neural constitutive relationship with Causal Relationship Constraints |
| `DiscoverUnit` | Conceptual storage with trainable release processes |
| `MultiDUModel` | Connections and simulation of the physical DU network |
| `DeepDiscover` | Simulation, observation mapping, calibration, evaluation and prediction |
| `HydroDataset` | Hydrological arrays and temporal-window preparation |
| `DifferentiableDegreeDaySnow` | Optional differentiable snow module |
| `Candidate` | Candidate description for structural comparisons |
| `compare_structures` | Fit and evaluate declared model structures |
| `equivalent_candidates` | Examine candidates under declared validation tolerances |

The [source repository](https://github.com/avadombi/DeepDiscover/tree/main/src/deepdiscover) contains the authoritative implementation. See [Your first DiscoverUnit](../tutorials/first_model.md) for a minimal example with tested public constructor patterns.

## Generate detailed API pages

When this documentation is installed in the source repository, `tools/generate_api.py` can generate detailed Markdown pages directly from `src/deepdiscover/`. It parses function signatures and class docstrings without importing TensorFlow. The generated documentation will then be available under [the detailed API pages](generated/index.md).

```{toctree}
:maxdepth: 1
:hidden:

generated/index
```
