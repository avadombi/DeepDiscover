# Your first DiscoverUnit

This example follows the public DeepDiscover interface. It uses synthetic inputs to demonstrate simulation without fitting the neural network to observations.

## Prepare the inputs

```python
import numpy as np
from deepdiscover import CRCNet, DeepDiscover, DiscoverUnit, MultiDUModel

n_days = 365
precipitation = np.maximum(
    2.0 + np.sin(np.arange(n_days) / 9.0), 0.0
).astype('float32')
pet = np.ones(n_days, dtype='float32')

water = precipitation[:, None]
drivers = np.column_stack([precipitation, pet])

print('water:', water.shape)
print('drivers:', drivers.shape)
```

**water** is the conserved input to the DU. **drivers** contains the two variables used as information by the neural process. Their shapes are `(365, 1)` and `(365, 2)` before batching.

## Define a release process

```python
release = CRCNet(gamma=[1, -1, 1], x_ref=[5, 3, 100])
unit = DiscoverUnit(
    process_nets=[release],
    context_indices=[(0, 1)],
)
physical_model = MultiDUModel(
    [unit], destinations=[['outlet']]
)
model = DeepDiscover(
    physical_model,
    observations={'streamflow': 'outlet'},
)
```

The process receives two selected driver variables and the antecedent storage. The CRC signs express the prescribed directions of their direct influences. The release amount remains limited by available water through the conservative allocation rule.

## Run a simulation

```python
prediction = model.predict(
    water[None], drivers[None]
)['streamflow']
print(prediction.shape)
```

Expected shape:

```text
(1, 365, 1)
```

The leading dimension denotes the batch. Here it contains one synthetic sequence.

## Examine the result

The current neural weights have not been calibrated, so this simulation illustrates software operation, not a hydrologically fitted model. The predicted discharge can be plotted using any standard Python plotting library. Inspecting storage and water-balance diagnostics requires accessing the simulation outputs through the configured DeepDiscover interface.

## Next steps

To calibrate a model, supply observed discharge through a **HydroDataset**:

```python
from deepdiscover import HydroDataset

# observed_discharge must be a measured series of n_days values.
data = HydroDataset(water, drivers, {'streamflow': observed_discharge})
train = data.make_windows(warmup=30, score_length=60, stride=30)
model.fit(train, epochs=30, batch_size=4, loss='mse')
```

The training snippet assumes `observed_discharge` has already been prepared. See [Training and evaluation](../training_and_evaluation.md) before running a calibration on real data.
