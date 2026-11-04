"""Public interface for fitting and evaluating DeepDiscover models."""
from __future__ import annotations

import json
import time
from pathlib import Path
import numpy as np
import tensorflow as tf
import keras

from .dataset import HydroBatch
from .model import MultiDUModel, observe
from .snow import apply_snow
from .metrics import masked_mse, masked_mae, masked_rmse, masked_huber, hydro_metrics


LOSSES = {"mse": masked_mse, "mae": masked_mae, "rmse": masked_rmse, "huber": masked_huber}


class DeepDiscover:
    """Training and prediction around one conservative MultiDUModel.

    The user supplies the physical model, observation operators and training
    data. Every observation may have its own differentiable loss. Raw physical
    outputs remain accessible through ``simulate()``.

    Parameters
    ----------
    model
        Configured MultiDUModel, TrainableRoutingMultiDUModel, or
        DynamicRoutingMultiDUModel.
    observations
        Mapping from observation name to a callable accepting the physical
        output dictionary. A string selects a physical output directly.
    snow
        Optional differentiable snow model applied to the specified water
        input DU, using the specified temperature driver index.
    driver_channels
        Optional tuple describing the channels passed to the constitutive
        networks. Integer entries select original driver columns. With snow,
        "rain", "snowmelt", "snowfall", and "liquid_water" select snow-module
        outputs. If omitted, original driver columns are used unchanged.
    """

    def __init__(self, model: MultiDUModel, observations: dict | None = None,
                 snow=None, snow_unit=0, temperature_index=None,
                 driver_channels=None):
        if not isinstance(model, MultiDUModel):
            raise TypeError("model must be a MultiDUModel.")
        self.model = model
        self.observations = dict(observations or {"streamflow": "outlet"})
        if not self.observations:
            raise ValueError("At least one observation is required.")
        for name, operator in self.observations.items():
            if not isinstance(operator, str) and not callable(operator):
                raise TypeError(f"Observation {name!r} needs a physical key or callable operator.")
        self.snow = snow
        self.snow_unit = int(snow_unit)
        self.temperature_index = temperature_index
        self.driver_channels = None if driver_channels is None else tuple(driver_channels)
        known_snow_channels = {"rain", "snowmelt", "snowfall", "liquid_water"}
        if self.driver_channels is not None:
            if not self.driver_channels:
                raise ValueError("driver_channels must contain at least one channel.")
            for channel in self.driver_channels:
                if isinstance(channel, (int, np.integer)) and not isinstance(channel, (bool, np.bool_)):
                    if channel < 0:
                        raise ValueError("Driver indices must be non-negative.")
                elif isinstance(channel, str) and channel in known_snow_channels:
                    if snow is None:
                        raise ValueError(f"Driver channel {channel!r} requires a snow model.")
                else:
                    raise ValueError("driver_channels entries must be non-negative indices or "
                                     f"one of {sorted(known_snow_channels)}.")
        if snow is not None:
            if temperature_index is None or not 0 <= self.snow_unit < model.n_du:
                raise ValueError("snow requires a valid snow_unit and temperature_index.")
        self.normalization = None
        self.history = []

    @tf.function(reduce_retracing=True)
    def _forward(self, water, drivers, initial_storage=None, initial_transit=None,
                 initial_memory=None, initial_snow_storage=None):
        water = tf.cast(tf.convert_to_tensor(water), tf.float32)
        drivers = tf.cast(tf.convert_to_tensor(drivers), tf.float32)
        tf.debugging.assert_rank(water, 3)
        tf.debugging.assert_rank(drivers, 3)
        tf.debugging.assert_all_finite(water, "Water inputs must be finite.")
        tf.debugging.assert_all_finite(drivers, "Drivers must be finite.")
        tf.debugging.assert_greater_equal(water, 0.0, message="Water inputs cannot be negative.")
        tf.debugging.assert_equal(tf.shape(water)[-1], self.model.n_du)
        batch = tf.shape(water)[0]
        if initial_storage is None:
            initial_storage = tf.zeros((batch, self.model.n_du), tf.float32)
        else:
            initial_storage = tf.cast(tf.convert_to_tensor(initial_storage), tf.float32)
        if self.snow is None:
            liquid_input = water
            snow_result = None
        else:
            snow_result = apply_snow(
                self.snow,
                water[..., self.snow_unit:self.snow_unit + 1],
                drivers[..., self.temperature_index:self.temperature_index + 1],
                initial_storage=initial_snow_storage,
            )
            units = tf.unstack(water, axis=-1)
            units[self.snow_unit] = snow_result["liquid_water"][..., 0]
            liquid_input = tf.stack(units, axis=-1)
        if self.driver_channels is None:
            process_drivers = drivers
        else:
            channels = []
            for channel in self.driver_channels:
                if isinstance(channel, str):
                    channels.append(snow_result[channel])
                else:
                    channels.append(drivers[..., channel:channel + 1])
            process_drivers = tf.concat(channels, axis=-1)
        output = self.model.run_sequence(
            liquid_input, process_drivers, initial_storage,
            initial_transit=initial_transit, initial_memory=initial_memory,
        )
        if snow_result is not None:
            output.update({f"snow_{name}": value for name, value in snow_result.items()})
            output["snow_storage"] = snow_result["snow_storage"]
            output["snowmelt"] = snow_result["snowmelt"]
            output["rain"] = snow_result["rain"]
            output["snowfall"] = snow_result["snowfall"]
            initial_snow = (tf.zeros((batch, 1), tf.float32) if initial_snow_storage is None
                            else tf.reshape(tf.cast(tf.convert_to_tensor(initial_snow_storage), tf.float32), (batch, 1)))
            snow_states = snow_result["snow_storage"]
            snow_previous = tf.concat([initial_snow[:, None, :], snow_states[:, :-1, :]], axis=1)
            snow_change = snow_states - snow_previous
            original_minus_liquid = tf.reduce_sum(water-liquid_input, axis=-1, keepdims=True)
            output["mass_residual"] = output["mass_residual"] + original_minus_liquid - snow_change
        output["external_water"] = water
        output["drivers"] = drivers
        output["process_drivers"] = process_drivers
        return output

    def simulate(self, water, drivers, **initial_states):
        """Return physical trajectories as TensorFlow tensors.

        Accepted state keywords: initial_storage, initial_transit, initial_memory,
        initial_snow_storage. Continuation uses the corresponding final states.
        """
        return self._forward(water, drivers, **initial_states)

    def predict(self, water, drivers, **initial_states):
        """Return a dictionary of named observation trajectories."""
        physical = self.simulate(water, drivers, **initial_states)
        return {
            name: observe(physical, observation=None if isinstance(op, str) else op,
                          target=op if isinstance(op, str) else "outlet")
            for name, op in self.observations.items()
        }

    def _variables(self):
        variables = list(self.model.trainable_variables)
        if self.snow is not None:
            variables += list(self.snow.trainable_variables)
        for op in self.observations.values():
            if isinstance(op, keras.layers.Layer):
                variables += list(op.trainable_variables)
        unique = []
        for variable in variables:
            if not any(variable is item for item in unique):
                unique.append(variable)
        return unique

    @staticmethod
    def _loss_functions(loss):
        if isinstance(loss, str):
            if loss not in LOSSES:
                raise ValueError(f"Available losses are {tuple(LOSSES)} or a custom callable.")
            return LOSSES[loss]
        if not callable(loss):
            raise TypeError("Each loss must be a name or a callable (observed, predicted, mask).")
        return loss

    def _objective(self, batch: HydroBatch, output, losses, weights, variances,
                   penalties=()):
        total = tf.constant(0.0, tf.float32)
        contributions = {}
        for name, operator in self.observations.items():
            observed = tf.convert_to_tensor(batch.targets[name], tf.float32)
            mask = tf.convert_to_tensor(batch.masks[name], tf.float32)
            predicted = observe(output, observation=None if isinstance(operator, str) else operator,
                                target=operator if isinstance(operator, str) else "outlet")
            tf.debugging.assert_equal(tf.shape(observed), tf.shape(predicted),
                                      message=f"Observation {name} and prediction shapes differ.")
            value = losses[name](observed, predicted, mask)
            tf.debugging.assert_all_finite(value, f"Non-finite loss for {name}.")
            weighted = tf.cast(weights[name] / variances[name], tf.float32) * value
            total = total + weighted
            contributions[name] = weighted
        for penalty in penalties:
            val = tf.convert_to_tensor(penalty(output), tf.float32)
            tf.debugging.assert_rank(val, 0)
            total = total + val
        return total, contributions

    @staticmethod
    def _prepare_weights(batch, observation_names, normalize, weights):
        variances = {}
        for name in observation_names:
            mask = batch.masks[name] > 0
            values = batch.targets[name][mask].astype(np.float64)
            if not values.size:
                raise ValueError(f"No scored training samples for {name}.")
            var = float(np.var(values))
            if normalize and var <= 0:
                raise ValueError(f"Cannot normalize constant training observations for {name}.")
            variances[name] = var if normalize else 1.0
        return variances, {name: float(weights.get(name, 1.0)) for name in observation_names}

    def fit(self, train: HydroBatch, validation: HydroBatch | None = None, *,
            epochs=100, batch_size=16, learning_rate=1e-3, optimizer=None,
            validation_every=1,
            loss="mse", weights=None, normalize=True, penalties=(),
            shuffle=True, seed=42, patience=None, verbose=1,
            train_initial_states=None, validation_initial_states=None):
        """Fit several independent temporal windows in parallel.

        Warm-up is excluded by the supplied masks. Variances are estimated from
        scored training values only. Validation never updates model parameters.
        With batch_size=None, all windows are fitted in parallel in one update.
        validation_every controls how often validation is evaluated; patience
        counts validation checks, not epochs, when validation_every > 1.
        """
        if not isinstance(train, HydroBatch) or (validation is not None and not isinstance(validation, HydroBatch)):
            raise TypeError("fit expects HydroBatch objects; use HydroDataset.make_windows().")
        if train.water.shape[-1] != self.model.n_du:
            raise ValueError("water has the wrong number of DU channels.")
        if not set(self.observations) <= set(train.targets):
            raise ValueError("All declared observations must be present in training data.")
        if validation is not None and not set(self.observations) <= set(validation.targets):
            raise ValueError("All declared observations must be present in validation data.")
        if int(epochs) < 1:
            raise ValueError("epochs must be positive.")
        batch_size = train.batch_size if batch_size is None else int(batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be positive or None.")
        validation_every = int(validation_every)
        if validation_every < 1:
            raise ValueError("validation_every must be positive.")
        losses = {name: self._loss_functions(loss.get(name, "mse") if isinstance(loss, dict) else loss)
                  for name in self.observations}
        variances, weights = self._prepare_weights(train, self.observations, normalize, weights or {})
        if any(w < 0 or not np.isfinite(w) for w in weights.values()):
            raise ValueError("Loss weights must be finite and non-negative.")
        self.normalization = variances.copy()
        if optimizer is None:
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        if not isinstance(optimizer, keras.optimizers.Optimizer):
            raise TypeError("optimizer must be a Keras optimizer.")
        rng = np.random.default_rng(seed)
        best_loss, bad_epochs, best_weights = float("inf"), 0, None
        self.history = []
        for epoch in range(int(epochs)):
            epoch_start = time.perf_counter()
            indices = rng.permutation(train.batch_size) if shuffle else np.arange(train.batch_size)
            epoch_losses = []
            n_batches = (train.batch_size + int(batch_size) - 1) // int(batch_size)
            for a in range(0, train.batch_size, int(batch_size)):
                batch_number = a // int(batch_size) + 1
                if verbose >= 2 and batch_number == 1:
                    print(f"Epoch {epoch + 1}/{int(epochs)}: starting {n_batches} batches", flush=True)
                selected = indices[a:a + int(batch_size)]
                chunk = train.subset(selected)
                initial = {name: np.asarray(value, np.float32)[selected]
                           for name, value in (train_initial_states or {}).items()}
                with tf.GradientTape() as tape:
                    physical = self._forward(chunk.water, chunk.drivers, **initial)
                    objective, _ = self._objective(chunk, physical, losses, weights, variances, penalties)
                variables = self._variables()
                if not variables:
                    raise ValueError("The model has no trainable parameters.")
                gradients = tape.gradient(objective, variables)
                pairs = [(g, v) for g, v in zip(gradients, variables) if g is not None]
                if not pairs:
                    raise ValueError("No training gradients were obtained.")
                tf.debugging.assert_all_finite(objective, "Training objective must be finite.")
                for grad, _ in pairs:
                    tf.debugging.assert_all_finite(grad, "Non-finite training gradient.")
                optimizer.apply_gradients(pairs)
                batch_loss = float(objective.numpy())
                epoch_losses.append(batch_loss)
                if verbose >= 2:
                    print(f"  Batch {batch_number}/{n_batches}: loss={batch_loss:.6g}", flush=True)
            record = {"epoch": epoch + 1, "train_loss": float(np.mean(epoch_losses))}
            should_validate = (
                validation is not None
                and ((epoch + 1) % validation_every == 0 or (epoch + 1) == int(epochs))
            )
            if should_validate:
                validation_start = time.perf_counter()
                record["validation_loss"] = self.evaluate(validation, loss=losses, weights=weights,
                                                           normalization=variances, penalties=penalties,
                                                           initial_states=validation_initial_states)["loss"]
                record["validation_seconds"] = round(time.perf_counter() - validation_start, 3)
                if record["validation_loss"] < best_loss - 1e-9:
                    best_loss, bad_epochs = record["validation_loss"], 0
                    best_weights = [np.array(v.numpy(), copy=True) for v in self._variables()]
                else:
                    bad_epochs += 1
            record["epoch_seconds"] = round(time.perf_counter() - epoch_start, 3)
            self.history.append(record)
            if verbose and (epoch == 0 or (epoch + 1) % max(1, int(epochs)//10) == 0):
                print(record, flush=True)
            if should_validate and patience is not None and bad_epochs >= patience:
                break
        if validation is not None and best_weights is not None:
            for variable, value in zip(self._variables(), best_weights):
                variable.assign(value)
        return list(self.history)

    def evaluate(self, batch: HydroBatch, *, loss="mse", weights=None,
                 normalization=None, penalties=(), initial_states=None):
        """Return the training objective and per-observation hydrological metrics."""
        if not isinstance(batch, HydroBatch):
            raise TypeError("evaluate expects a HydroBatch.")
        losses = {name: self._loss_functions(loss.get(name, "mse") if isinstance(loss, dict) else loss)
                  for name in self.observations}
        weights = {name: float((weights or {}).get(name, 1.0)) for name in self.observations}
        variances = normalization or self.normalization or {name: 1.0 for name in self.observations}
        physical = self._forward(batch.water, batch.drivers, **(initial_states or {}))
        total, _ = self._objective(batch, physical, losses, weights, variances, penalties)
        metrics = {}
        for name, operator in self.observations.items():
            sim = observe(physical, observation=None if isinstance(operator, str) else operator,
                          target=operator if isinstance(operator, str) else "outlet")
            metrics[name] = hydro_metrics(batch.targets[name], sim.numpy(), batch.masks[name])
        return {"loss": float(total.numpy()), "metrics": metrics,
                "max_abs_mass_residual": float(tf.reduce_max(tf.abs(physical["mass_residual"])).numpy())}

    def describe(self):
        """Return a concise description of the configured hydrological model."""
        return {
            "n_du": self.model.n_du,
            "process_destinations": self.model.destinations,
            "overflow_destinations": self.model.overflow_destinations,
            "route_lags": list(self.model._route_support_lengths),
            "routing_class": type(self.model).__name__,
            "units": [
                {
                    "n_processes": len(du.process_nets),
                    "n_sources": len(du.sources),
                    "n_sinks": len(du.sinks),
                    "bounded": du.bounded,
                    "capacity_mode": "learned" if du._smax_bounds is not None else
                                     "fixed" if du._fixed_smax is not None else "unbounded",
                    "memory": du.has_memory,
                    "deficit": du.include_deficit,
                }
                for du in self.model.discover_units
            ],
            "observations": sorted(self.observations),
            "snow": self.snow is not None,
            "driver_channels": (list(self.driver_channels)
                                if self.driver_channels is not None else None),
        }

    def _checkpoint_variables(self):
        variables = list(self.model.variables)
        if self.snow is not None:
            variables += list(self.snow.variables)
        for op in self.observations.values():
            if isinstance(op, keras.layers.Layer):
                variables += list(op.variables)
        unique = []
        for variable in variables:
            if not any(variable is item for item in unique):
                unique.append(variable)
        return unique

    def save_checkpoint(self, directory, metadata: dict | None = None):
        """Save weights, training statistics and metadata.

        To load, create the same architecture and observation operators, run a
        forward pass to build their weights, then call ``load_checkpoint``.
        Custom layers are recreated by their original Python definitions.
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        variables = self._checkpoint_variables()
        if not variables:
            raise ValueError("Run a forward pass before saving.")
        np.savez_compressed(path / "weights.npz", **{f"weight_{i}": v.numpy() for i, v in enumerate(variables)})
        content = {"version": "1.0.0", "n_du": self.model.n_du,
                   "n_routes": self.model.n_routes, "n_processes": self.model.n_processes,
                   "weights": len(variables),
                   "weight_roles": [v.path.split("/")[-1] for v in variables],
                   "structure": self.describe(),
                   "normalization": self.normalization, "metadata": metadata or {}}
        (path / "checkpoint.json").write_text(json.dumps(content, indent=2), encoding="utf-8")

    def load_checkpoint(self, directory):
        """Restore weights into an already-built model with matching architecture."""
        path = Path(directory)
        metadata = json.loads((path / "checkpoint.json").read_text(encoding="utf-8"))
        if metadata["n_du"] != self.model.n_du or metadata["n_routes"] != self.model.n_routes or metadata["n_processes"] != self.model.n_processes:
            raise ValueError("The checkpoint does not match the physical architecture.")
        variables = self._checkpoint_variables()
        if len(variables) != metadata["weights"]:
            raise ValueError("Build the same model and observation operators with simulate/predict before loading.")
        if metadata.get("weight_roles") != [v.path.split("/")[-1] for v in variables]:
            raise ValueError("Checkpoint parameter roles differ from the active model.")
        if metadata.get("structure") != self.describe():
            raise ValueError("Checkpoint topology differs from the active model.")
        with np.load(path / "weights.npz", allow_pickle=False) as values:
            for i, var in enumerate(variables):
                array = values[f"weight_{i}"]
                if var.shape != array.shape:
                    raise ValueError(f"Checkpoint weight {i} has a different shape.")
                var.assign(array)
        self.normalization = metadata.get("normalization")
        return metadata
