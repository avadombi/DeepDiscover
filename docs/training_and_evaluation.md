# Training and evaluation

The `HydroDataset` class holds continuous water inputs, predictors and measured values. Missing measurements are masked. Train/validation/test periods must be specified by the user.

```python
train = data.make_windows(warmup=60, score_length=120, stride=60,
                          first_score=60, last_score=730)
validation = data.continuous_batch(start=730, stop=1000, warmup=60)

history = model.fit(
    train, validation,
    epochs=100, batch_size=8, learning_rate=0.001,
    loss={"flow": "mse", "head": "huber"},
    weights={"flow": 1.0, "head": 0.5},
    normalize=True, patience=15,
)
summary = model.evaluate(validation)
```

For one gradient update over all windows in a batch, set `batch_size=None`. To validate every five epochs, use `validation_every=5`. The `patience` parameter counts consecutive validation checks without improvement. The training history records `epoch_seconds`; when validation is run it also records `validation_seconds`. The hydrological simulation is compiled with TensorFlow and supports a batch of independent temporal windows.

Training and validation sequences simulate the warm-up. The masks exclude warm-up observations from scoring. With `normalize=True`, each observation loss is divided by the variance of the scored training observations. These variances are held fixed during validation. Choose `normalize=False` when your loss function already handles scaling.

Supported loss names: `mse`, `mae`, `rmse`, `huber`. Custom losses use the signature `loss(observed, predicted, mask)` and return a differentiable scalar. You may pass a dictionary with distinct loss functions for each observation name. Extra scalar regularizers use `penalties=[function]` with each function accepting the dictionary of physical outputs.

`model.evaluate` reports the objective, per-variable NSE, KGE, RMSE, MAE, PBIAS, correlation and maximum absolute numerical mass residual. Undefined metrics are reported as NaN when the scored record is degenerate.

The model assumes that batch members are independent temporal windows. Initial water storage, route transit and informational memory are zero in `fit` unless physically initialized through a preceding warm-up. For continuous prediction, pass explicit initial states and retain final states between segments.

## Saving and reloading

```python
model.save_checkpoint("results/experiment_01", metadata={"case": "my_basin"})

# In a fresh process, rebuild the same physical model and observation operators.
restored = DeepDiscover(physical_model, observations=operators)
restored.predict(train.water[:1], train.drivers[:1])   # initializes weights
restored.load_checkpoint("results/experiment_01")
```

The checkpoint contains trained parameter arrays and normalization values. The same Python model configuration and custom layer definitions must be available during reloading. Store the configuration and relevant input normalizations alongside the checkpoint for reproduction.

## User-defined initial states

For hydrological systems with important initial storage, `fit` accepts optional `train_initial_states` and `validation_initial_states` dictionaries. Each training array must have one state row per window. Supported keys are `initial_storage`, `initial_memory`, `initial_transit` and `initial_snow_storage`. Use `evaluate(batch, initial_states={...})` for evaluation with prescribed initial states. This provides a direct way to calibrate pre-existing water stores or resume known physical states without silently resetting their initial quantities.

The metric dictionary also includes `nrmse_std`, `pearson_r2` and streamflow-oriented flow-duration-curve diagnostics `fhv_percent`, `fms_percent`, `flv_percent`. The FDC metrics should not be interpreted as physical groundwater-level diagnostics. They are reported only as numerical summaries when calculated for other variables.

## Snow and constitutive drivers

When snow is enabled, the input data can preserve the original `(P, PET, T)` driver columns while the constitutive networks use the snow-module outputs. For example:

```python
model = DeepDiscover(
    physical_model,
    snow=snow_module,
    temperature_index=2,
    driver_channels=("rain", "snowmelt", 1),
)
```

The input column 1 is PET. The resulting network drivers are `(rain, snowmelt, PET)`. The names `rain`, `snowmelt`, `snowfall` and `liquid_water` refer to water quantities produced by the snow module. Numeric entries select columns in the original driver array. If `driver_channels` is omitted, networks receive the original driver array unchanged.
