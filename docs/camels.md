# CAMELS-US example

The public repository includes `04_camels_us.ipynb`, a worked example using daily meteorological forcing and observed streamflow from the CAMELS-US dataset.

The notebook illustrates the preparation of precipitation, potential evapotranspiration and temperature inputs, the construction of temporal windows, and the calibration and examination of a model with connected DiscoverUnits.

## Data

Obtain the original forcing and streamflow files according to the [repository's CAMELS instructions](https://github.com/avadombi/DeepDiscover/blob/main/docs/camels.md). Verify the basin identifier, date alignment, units, and missing-value handling before running the notebook.

The package contains utilities for loading CAMELS-US forcing and observed discharge. The loader uses additional dependencies supplied through the `examples` installation extra.

## Notebook workflow

The example follows the same general sequence as the synthetic tutorial:

1. Load and check the forcing and observations.
2. Prepare physical water inputs and explanatory drivers.
3. Select training and validation periods.
4. Construct the DUs, their processes, and their connections.
5. Fit the model and examine the predicted discharge and internal outputs.

The notebook is an application example. Its calibration scores depend on the data partitions, model structure, random initialization, and optimization settings used in a particular run.

## Original data and citation

CAMELS-US is described by [Newman et al. (2015)](https://doi.org/10.5194/hess-19-209-2015) and [Addor et al. (2017)](https://doi.org/10.5194/hess-21-5293-2017). Consult the dataset distribution for use and citation details.
