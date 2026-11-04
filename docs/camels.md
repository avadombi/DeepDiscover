# CAMELS-US data

The CAMELS-US example uses the forcing and observed streamflow files for basin `01013500`. Download the original data from the CAMELS-US provider. The files are not part of this repository.

Arrange the files as follows:

```text
data/
    camels/
        01013500/
            01013500_lump_cida_forcing_leap.txt
            01013500_streamflow_qc.txt
```

```python
from deepdiscover import load_camels_dataset

data = load_camels_dataset(
    ".", "01013500", n_du=2,
    start="1980-10-01", end="2010-09-30",
)
print(data.water.shape, data.drivers.shape)
```

The data reader computes mean temperature, Hamon potential evapotranspiration and catchment-equivalent streamflow depth. Predictor order is precipitation, potential evapotranspiration and temperature. External precipitation enters DU 0 by default.

A value of `-999` or quality flag `M` in the CAMELS streamflow file indicates a missing observation. These values are excluded from calibration through the observation mask. Other negative discharge values cause an error. Missing precipitation or meteorological drivers must be handled explicitly before simulation.

The published case-study chronology uses 365 warm-up days from 1980-10-01, training from 1981-10-01 to 2000-09-30, validation from 2000-10-01 to 2005-09-30 and testing from 2005-10-01 to 2010-09-30. The notebook selects these dates explicitly.

Review the original CAMELS-US data documentation and quality flags before using alternative basins or periods. Cite the original data sources in scientific publications.

## Running the notebook

For a short execution check:

```bash
python run_examples.py --only 04_camels_us
```

To measure training speed on the complete set of 19 windows:

```bash
python run_examples.py --only 04_camels_us --camels-mode benchmark
```

For full calibration:

```bash
python run_examples.py --only 04_camels_us --camels-mode full
```

The physical model in this example has two DUs and three release processes: one direct outlet pathway, one transfer to the second DU, and one downstream outlet pathway. The snow module separates precipitation into rain and snowmelt. These quantities and PET form the process drivers, in that order. The temperature series is used by the snow module.

Training data have shapes `water=(19,730,2)`, `drivers=(19,730,3)` and `streamflow=(19,730,1)`. Each window contains 365 unscored warm-up days and 365 scored days. The 19 complete scored blocks cover 6,935 of the 6,940 available training days; the final five days are not included in these nonoverlapping blocks. Full calibration fits all 19 windows together in one gradient update per epoch, validates every five epochs, and uses an upper limit of 400 epochs with early stopping. The notebook uses a learning rate of 0.01. The benchmark mode uses the same windows but only three epochs to report execution times.

The check and benchmark modes verify execution and measure timing. Their calibration scores should not be used in scientific analysis. The independent test period is evaluated only in full mode, after checkpoint selection on validation data. Do not use test results for model selection. For comparisons between models, report the random seeds, model parameters, calibration periods and evaluation conventions.


## Verifying a saved checkpoint

After a full notebook run, the selected weights and validation history are saved. To check their consistency without fitting again, run:

```bash
python verify_camels.py
```

This procedure reconstructs the notebook architecture, loads the saved parameters, evaluates the validation period, and checks the water balance. The recorded validation objective and selected epoch must agree with the saved model. It does not evaluate the independent test period. The checkpoint format requires the same model configuration and a compatible DeepDiscover version.
