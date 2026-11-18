# Training and evaluation

The `DeepDiscover` interface fits trainable model components by comparing simulated observation variables with measurements.

## Build training windows

Prepare a `HydroDataset` and divide its continuous record into temporal windows. Keep a warm-up prefix for state evolution and a scored interval for the objective.

```python
train = data.make_windows(warmup=30, score_length=60, stride=30)
```

## Fit the model

The public interface supports gradient-based optimization, with Adam available as a default optimizer. An illustrative calibration call is

```python
history = model.fit(
    train,
    epochs=30,
    batch_size=4,
    loss='mse',
)
```

When several observation variables are configured, loss functions and optional weights can be supplied for the individual targets. Different variable units should be considered when interpreting the resulting objective.

## Validation and model selection

Use a chronological validation period that does not overlap the scored training period. The package supports periodic validation and selection of model parameters using the configured validation criterion.

A fitted model may be sensitive to random initialization, optimizer settings, and the selected model structure. For comparisons, report the model-building choices, training budget, seeds, and validation procedure.

## Examine predictive performance

Depending on the observed variable, common metrics include Nash-Sutcliffe efficiency (NSE), Kling-Gupta efficiency (KGE), root mean squared error (RMSE), and bias. These metrics describe different aspects of the agreement with observations.

NSE for observations $\mathrm{q}^{(t)}$ and simulations $\widehat{\mathrm{q}}^{(t)}$ is

```{math}
:label: training-and-evaluation-1

\mathrm{NSE}=1-\frac{\sum_{t}(\mathrm{q}^{(t)}-\widehat{\mathrm{q}}^{(t)})^2}{\sum_{t}(\mathrm{q}^{(t)}-\overline{\mathrm{q}})^2}.
```

A high metric value does not establish that an unobserved internal process has been uniquely identified.

## Water balance and internal processes

Inspect the model's conservation residual alongside predictive scores. The residual should remain small relative to the scale and floating-point precision of the simulated quantities. Storage, process release, routing transit, and physical losses are useful additional diagnostics.

## Save and restore

DeepDiscover provides functions for saving and restoring trainable parameters and associated configuration information. To restore a trained model, reconstruct the compatible physical model and observation operators before loading the saved parameters. Custom components must remain available in Python.
