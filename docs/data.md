# Data preparation

DeepDiscover uses explicit arrays for physical water inputs, neural explanatory variables, and observations. The **HydroDataset** interface stores these quantities and constructs temporal windows for training and evaluation.

## Shapes and conventions

A batch of $\mathrm{B}$ windows, each containing $\mathrm{T}$ time steps, uses

```text
water        (B, T, number_of_DUs)
drivers      (B, T, number_of_drivers)
observations (B, T, number_of_targets)
```

Physical inputs and explanatory drivers need separate dimensions. Both arrays must have the same number of time steps and use the same temporal ordering.

## Continuous data

Before windowing, a single catchment record can be stored in two-dimensional arrays, one row per time step. For example:

```python
import numpy as np
from deepdiscover import HydroDataset

water = precipitation.astype('float32')[:, None]
drivers = np.column_stack([precipitation, pet]).astype('float32')
observations = {'streamflow': measured_discharge}

data = HydroDataset(water, drivers, observations)
```

This example assumes that `precipitation`, `pet`, and `measured_discharge` have already been prepared with consistent units and aligned dates.

## Warm-up and scored periods

Water storage depends on preceding inputs, so an initial period is usually needed to evolve the internal states. Training windows may include a **warm-up** prefix, followed by a **scored** part contributing to the optimization objective.

```text
|--------- warm-up ---------|------- scored observations -------|
                           ^ loss starts here
```

An illustrative call is

```python
train = data.make_windows(
    warmup=30,
    score_length=60,
    stride=30,
)
```

Choose the warm-up duration and window length according to the hydrological memory, available record, and modeling objective. A batch represents several independent temporal windows from the supplied record.

## Data partitions

Training, validation, and independent evaluation intervals should be defined chronologically. Data preprocessing parameters estimated from observations should use the training interval. The validation period supports model selection, while the independent evaluation period is reserved for assessing the selected procedure.

## Missing values

Missing observations can be handled through observation masks. The package does not automatically reconstruct missing physical forcing records. Check and prepare the inputs before simulation.
