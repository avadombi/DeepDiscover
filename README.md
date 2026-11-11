# DeepDiscover

DeepDiscover is an open-source Python library for hydrological model discovery. It learns constitutive relationships for water release processes within a differentiable, mass-conserving framework. Models comprise one or more DiscoverUnits (DUs), each with a water storage and candidate processes. Each process has one declared destination, with temporal routing configured separately.

The same hydrological system can be calibrated against streamflow, groundwater levels, soil moisture and other measured variables using observation operators. Multiple observation types can be used together.

## Installation

Python 3.11 or 3.12 is recommended. DeepDiscover uses TensorFlow and Keras.

Install the published package from [PyPI](https://pypi.org/project/deepdiscover-hydro/):

```bash
python -m pip install deepdiscover-hydro
```

To run the notebooks or build the documentation from a source checkout:

```bash
python -m pip install -e ".[examples,docs]"
```

## Example

```python
import numpy as np
from deepdiscover import CRCNet, DeepDiscover, DiscoverUnit, MultiDUModel

n_days = 365
precipitation = np.maximum(2 + np.sin(np.arange(n_days) / 9), 0).astype("float32")
pet = np.ones(n_days, dtype="float32")

water = precipitation[:, None]
drivers = np.column_stack([precipitation, pet])

release = CRCNet(gamma=[1, -1, 1], x_ref=[5, 3, 100])
unit = DiscoverUnit(process_nets=[release], context_indices=[(0, 1)])
physical_model = MultiDUModel([unit], destinations=[["outlet"]])
model = DeepDiscover(physical_model, observations={"streamflow": "outlet"})

prediction = model.predict(water[None], drivers[None])["streamflow"]
print(prediction.shape)  # (1, 365, 1)
```

To calibrate the model, provide observed discharge as an array and create training windows:

```python
from deepdiscover import HydroDataset

data = HydroDataset(water, drivers, {"streamflow": observed_discharge})
train = data.make_windows(warmup=30, score_length=60, stride=30)
model.fit(train, epochs=30, batch_size=4, loss="mse")
```

Here `observed_discharge` is a measured series containing one value per day.

## Main features

| Component | Options |
| --- | --- |
| DiscoverUnits | Multiple storages, processes and fixed or learned storage capacities |
| Constitutive relationships | Causal Relationship Constraints (CRC), custom Keras layers |
| Sources | Prescribed water inputs, lateral inflow, managed recharge, custom source layers |
| Losses | Evapotranspiration, pumping, prescribed extraction, custom sink layers |
| Routes | Instantaneous, prescribed, learned and state-dependent temporal routing |
| Additional states | Optional information memory and storage deficit |
| Snow | Differentiable degree-day snow model |
| Observations | Direct outputs, groundwater-head operators, soil-moisture operators, custom layers |
| Training | Time windows, warm-up, observation masks, multivariable and custom losses |
| Diagnostics | States, fluxes, routing storage, numerical water balance and model scores |
| Model comparison | Multi-seed fitting and validation-based comparison of declared structures |

Water losses belong to the physical balance. Training losses measure the agreement between simulations and observations. Both can be configured independently.

## Model topologies

DeepDiscover supports a configurable number of DiscoverUnits and release processes. Internal transfers can form serial cascades, parallel branches, or other acyclic networks, and several processes may share a downstream destination or contribute to the outlet. Each process has exactly one declared destination. The [model topologies](https://github.com/avadombi/DeepDiscover/blob/main/docs/topologies.md) page includes two-DU configurations, a four-DU cascade, and a branched four-DU network. See [structural comparison](https://github.com/avadombi/DeepDiscover/blob/main/docs/structural_comparison.md) for validation-based comparisons of alternative models.

## Examples

The `examples/` directory contains four notebooks:

- `01_streamflow.ipynb`: simulation and calibration of streamflow.
- `02_groundwater_level.ipynb`: joint calibration using streamflow and groundwater levels.
- `03_sources_losses_routing.ipynb`: custom physical boundaries and delayed routing.
- `04_camels_us.ipynb`: a CAMELS-US case study with snow and chronological data partitions.

To run the three self-contained notebooks:

```bash
python run_examples.py
```

To run the CAMELS-US example, obtain the original forcing and streamflow files and follow [the data instructions](https://github.com/avadombi/DeepDiscover/blob/main/docs/camels.md). Three execution modes are provided:

```bash
# Short run: loading, simulation, calibration and evaluation
python run_examples.py --only 04_camels_us

# Timing on full-length training windows (three epochs)
python run_examples.py --only 04_camels_us --camels-mode benchmark

# Full calibration with the complete training windows
python run_examples.py --only 04_camels_us --camels-mode full
```

The short check uses a limited number of training windows and two epochs. Its scores should not be used as scientific results. The benchmark and full modes use 19 parallel windows with 365 warm-up and 365 scored days. Full mode uses up to 400 epochs and validates every five epochs. It saves the validation-selected model and its training history under `results/camels/01013500/two_du_serial_seed42/`. During notebook execution, progress is printed for each code cell and each optimization batch.

Output notebooks and execution reports are written to `results/notebooks/`, which is excluded from version control.

A saved CAMELS-US checkpoint can be verified independently, without repeating model fitting or consulting the test period:

```bash
python verify_camels.py
```

This command requires the CAMELS-US files and the `checkpoint/` and `history.json` created by the full example.

## Documentation

The `docs/` directory covers the [scientific equations](https://github.com/avadombi/DeepDiscover/blob/main/docs/scientific_formulation.md), [model configuration](https://github.com/avadombi/DeepDiscover/blob/main/docs/model_configuration.md), [topologies](https://github.com/avadombi/DeepDiscover/blob/main/docs/topologies.md), [data](https://github.com/avadombi/DeepDiscover/blob/main/docs/data.md), [routing](https://github.com/avadombi/DeepDiscover/blob/main/docs/routing.md), [observation operators](https://github.com/avadombi/DeepDiscover/blob/main/docs/observations.md), [custom components](https://github.com/avadombi/DeepDiscover/blob/main/docs/customization.md), [calibration](https://github.com/avadombi/DeepDiscover/blob/main/docs/training_and_evaluation.md) and [model comparison](https://github.com/avadombi/DeepDiscover/blob/main/docs/structural_comparison.md).

To build the documentation locally:

```bash
python -m pip install ".[docs]"
mkdocs build
```

## Tests

```bash
python -m pip install ".[test]"
python check.py
python -m pytest -q
```

The tests cover conservation, routing, gradients, observation operators, training, model saving, temporal windows and CAMELS data preparation.

## Reporting bugs

Report bugs and request features through [GitHub Issues](https://github.com/avadombi/DeepDiscover/issues). Include the Python, TensorFlow and Keras versions, the model configuration, a minimal example and the complete error message. Avoid sharing private datasets or credentials.

See [CONTRIBUTING.md](https://github.com/avadombi/DeepDiscover/blob/main/CONTRIBUTING.md) for contribution guidelines.

## Scientific references and license

DeepDiscover is intended for water-quantity modeling. Its latent states and processes require scientific interpretation in the context of the available observations. See [limitations](https://github.com/avadombi/DeepDiscover/blob/main/docs/limitations.md).

The methodological article is identified in [CITATION.cff](https://github.com/avadombi/DeepDiscover/blob/main/CITATION.cff). Source code is distributed under the [BSD 3-Clause License](https://github.com/avadombi/DeepDiscover/blob/main/LICENSE). CAMELS-US data must be obtained separately from their original provider.

## Citation

The Multiple-DU methodology is described in:

Adombi, A. V. D. P. (2026). *Can discharge identify internal hydrological processes? Neural equifinality in the Multiple-DU extension of DeepDiscover*. Manuscript submitted.

The original DeepDiscover formulation is published in:

Adombi, A. V. D. P. (2026). *DeepDiscover: towards autonomous discovery of bucket-type conceptual models – a proof of concept applied to hydrology*. *Journal of Hydrology*, 671, 135249. https://doi.org/10.1016/j.jhydrol.2026.135249

