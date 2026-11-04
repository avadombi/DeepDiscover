from __future__ import annotations

import numpy as np
import keras
import tensorflow as tf


MONOTONE_ACTIVATIONS = {"linear", "relu", "elu", "selu", "softplus", "sigmoid", "tanh"}


def _select_columns(values, indices):
    """Select driver columns while handling an intentionally empty selection."""
    if not indices:
        return tf.zeros((tf.shape(values)[0], 0), dtype=values.dtype)

    indices = tf.constant(indices, dtype=tf.int32)
    return tf.gather(values, indices, axis=-1)


class PositiveDense(keras.layers.Layer):
    """Dense layer with a non-negative kernel for CRC monotonicity."""

    def __init__(self, units: int, activation="linear", bias_init=0.0, **kwargs):
        super().__init__(**kwargs)

        if isinstance(activation, str) and activation not in MONOTONE_ACTIVATIONS:
            raise ValueError(
                f"Activation '{activation}' is not in the approved monotone set: "
                f"{sorted(MONOTONE_ACTIVATIONS)}."
            )

        self.units = int(units)
        self.activation_name = activation
        self.activation = keras.activations.get(activation)
        self.bias_init = float(bias_init)

    def build(self, input_shape):
        input_dim = int(input_shape[-1])

        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_dim, self.units),
            initializer=keras.initializers.RandomUniform(minval=0.01, maxval=0.10),
            constraint=keras.constraints.NonNeg(),
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(self.units,),
            initializer=keras.initializers.Constant(self.bias_init),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        y = tf.linalg.matmul(x, self.kernel) + self.bias
        return self.activation(y)


def scale_nonnegative(context, x_ref):
    """Scale non-negative physical variables as x / (x + x_ref)."""
    context = tf.convert_to_tensor(context, dtype=tf.float32)
    x_ref = tf.convert_to_tensor(x_ref, dtype=tf.float32)

    tf.debugging.assert_greater_equal(
        context,
        0.0,
        message="CRC context must be non-negative before sign application.",
    )
    tf.debugging.assert_positive(x_ref, message="x_ref must be strictly positive.")
    return context / (context + x_ref)


class CRCNet(keras.Model):
    """
    Small DeepDiscover monotonic network for one process.

    gamma = +1 : non-decreasing response
    gamma = -1 : non-increasing response
    gamma =  0 : direct context channel removed
    """

    def __init__(
        self,
        gamma,
        x_ref,
        hidden_units=16,
        hidden_activation="tanh",
        output="softplus",
        name=None,
    ):
        super().__init__(name=name)

        gamma = np.asarray(gamma, dtype=np.float32)
        x_ref = np.asarray(x_ref, dtype=np.float32)

        if gamma.ndim != 1 or x_ref.ndim != 1:
            raise ValueError("gamma and x_ref must be 1-D vectors.")
        if gamma.shape != x_ref.shape:
            raise ValueError("gamma and x_ref must have the same shape.")
        if not np.all(np.isin(gamma, (-1.0, 0.0, 1.0))):
            raise ValueError("gamma values must be -1, 0, or +1.")
        if np.any(x_ref <= 0.0):
            raise ValueError("All x_ref values must be strictly positive.")
        if hidden_activation not in MONOTONE_ACTIVATIONS:
            raise ValueError(
                f"hidden_activation must be one of {sorted(MONOTONE_ACTIVATIONS)}."
            )
        if output not in {"softplus", "sigmoid", "relu"}:
            raise ValueError("output must be 'softplus', 'sigmoid', or 'relu'.")

        self.gamma = tf.constant(gamma, dtype=tf.float32)
        self.x_ref = tf.constant(x_ref, dtype=tf.float32)
        self.output_name = output

        self.hidden = PositiveDense(
            hidden_units,
            activation=hidden_activation,
            bias_init=0.0,
            name="hidden",
        )

        head_bias = -3.0 if output == "softplus" else (-1.0 if output == "sigmoid" else 0.0)
        self.head = PositiveDense(1, activation=output, bias_init=head_bias, name="head")

    def signed_context(self, context):
        """Scale physical variables, then apply their CRC signs."""
        context = tf.convert_to_tensor(context, dtype=tf.float32)

        if context.shape[-1] != self.gamma.shape[0]:
            raise ValueError(
                f"Expected {self.gamma.shape[0]} context variables, got {context.shape[-1]}."
            )

        scaled = scale_nonnegative(context, self.x_ref)
        return scaled * self.gamma

    def call(self, context):
        z = self.signed_context(context)
        return self.head(self.hidden(z))


class PrescribedSource(keras.layers.Layer):
    """
    Non-negative external water source read from one driver column.

    Parameters
    ----------
    driver_index
        Column in `drivers` containing the source flux, e.g. managed recharge.
    scale
        Optional multiplicative conversion factor.
    """

    kind = "source"

    def __init__(self, driver_index, scale=1.0, name=None):
        super().__init__(name=name)

        if int(driver_index) < 0:
            raise ValueError("driver_index must be non-negative.")
        if scale <= 0.0:
            raise ValueError("scale must be strictly positive.")

        self.driver_index = int(driver_index)
        self.scale = float(scale)

    def call(self, drivers):
        source = drivers[:, self.driver_index:self.driver_index + 1] * self.scale
        tf.debugging.assert_greater_equal(source, 0.0, message="boundary source must be non-negative.")
        return source


class DemandLimitedSink(keras.layers.Layer):
    """
    Non-negative boundary sink controlled by a CRC efficiency network.

    The CRC network returns an efficiency eta in [0,1]. The requested sink is

        sink_request = eta * min(demand, available)

    For evapotranspiration, `demand` can be PET.

    Parameters
    ----------
    efficiency_net
        CRCNet with sigmoid output.
    demand_index
        Column in `drivers` containing the non-negative external demand.
    context_indices
        Driver columns used by the efficiency network. Beginning-of-step
        storage S(t-1) is appended automatically.
    """

    kind = "sink"

    def __init__(self, efficiency_net, demand_index, context_indices=(), name=None):
        super().__init__(name=name)

        if not isinstance(efficiency_net, keras.layers.Layer):
            raise TypeError("efficiency_net must be a Keras layer.")
        if isinstance(efficiency_net, CRCNet) and efficiency_net.output_name != "sigmoid":
            raise ValueError("DemandLimitedSink requires sigmoid output from CRCNet.")
        if int(demand_index) < 0:
            raise ValueError("demand_index must be non-negative.")

        self.efficiency_net = efficiency_net
        self.demand_index = int(demand_index)
        self.context_indices = tuple(int(i) for i in context_indices)

        expected = len(self.context_indices) + 1
        if isinstance(self.efficiency_net, CRCNet) and int(self.efficiency_net.gamma.shape[0]) != expected:
            raise ValueError(
                f"efficiency_net expects {int(self.efficiency_net.gamma.shape[0])} variables, "
                f"but the sink context contains {expected}."
            )

    def call(self, drivers, storage, available):
        demand = drivers[:, self.demand_index:self.demand_index + 1]
        tf.debugging.assert_greater_equal(demand, 0.0, message="boundary demand must be non-negative.")

        selected = _select_columns(drivers, self.context_indices)
        context = tf.concat([selected, storage], axis=-1)
        efficiency = self.efficiency_net(context)
        tf.debugging.assert_greater_equal(efficiency, 0.0, message="Sink efficiency must be non-negative.")
        tf.debugging.assert_less_equal(efficiency, 1.0, message="Sink efficiency must not exceed one.")

        request = efficiency * tf.minimum(demand, available)
        return request


def conservative_allocation(resource, propensities, smax=None):
    """Conservative split between storage, process fluxes, and optional overflow."""
    resource = tf.convert_to_tensor(resource, dtype=tf.float32)
    propensities = tf.convert_to_tensor(propensities, dtype=tf.float32)

    tf.debugging.assert_greater_equal(resource, 0.0, message="resource must be non-negative.")
    tf.debugging.assert_greater_equal(
        propensities, 0.0, message="process propensities must be non-negative."
    )

    denominator = 1.0 + tf.reduce_sum(propensities, axis=-1, keepdims=True)
    storage_raw = resource / denominator
    fluxes = storage_raw * propensities

    if smax is None:
        storage = storage_raw
        overflow = tf.zeros_like(storage_raw)
    else:
        smax = tf.convert_to_tensor(smax, dtype=tf.float32)
        tf.debugging.assert_positive(smax, message="smax must be strictly positive.")
        overflow = tf.maximum(storage_raw - smax, 0.0)
        storage = tf.minimum(storage_raw, smax)

    return storage, fluxes, overflow


class InformationalMemoryCell(keras.layers.Layer):
    """Optional informational memory attached to one DiscoverUnit.

    This keeps the memory-cell idea used in eDD but deliberately separates it
    from eDD process routing. The state ``h`` is dimensionless and carries no
    water. Its update is

        h_next = (1 - lambda) * h + lambda * z

    where ``lambda`` is a sigmoid CRC output and ``z`` is a non-negative target
    generated from the selected drivers and beginning-of-step storage. ``h`` can
    enrich the DD constitutive context, but it never enters the water balance and
    never introduces a recurrence in the generated process. Temporal routing is
    handled independently by ``MultiDUModel`` route kernels.
    """

    def __init__(self, driver_indices, gamma, x_ref, hidden_units=8, name=None):
        super().__init__(name=name)
        self.driver_indices = tuple(int(i) for i in driver_indices)
        expected = len(self.driver_indices) + 1  # selected drivers + storage
        gamma = tuple(float(v) for v in gamma)
        x_ref = tuple(float(v) for v in x_ref)
        if len(gamma) != expected or len(x_ref) != expected:
            raise ValueError(
                "Memory-cell gamma/x_ref must contain selected drivers + storage."
            )
        self.decay_net = CRCNet(
            gamma=gamma,
            x_ref=x_ref,
            hidden_units=hidden_units,
            output="sigmoid",
            name=f"{name}_decay" if name else None,
        )
        self.target_net = CRCNet(
            gamma=gamma,
            x_ref=x_ref,
            hidden_units=hidden_units,
            output="relu",
            name=f"{name}_target" if name else None,
        )

    def call(self, memory, drivers, storage):
        memory = tf.convert_to_tensor(memory, dtype=tf.float32)
        drivers = tf.convert_to_tensor(drivers, dtype=tf.float32)
        storage = tf.convert_to_tensor(storage, dtype=tf.float32)
        selected = _select_columns(drivers, self.driver_indices)
        context = tf.concat([selected, storage], axis=-1)
        decay = self.decay_net(context)
        target = self.target_net(context)
        next_memory = (1.0 - decay) * memory + decay * target
        return next_memory, decay, target


class DiscoverUnit(keras.layers.Layer):
    """Conservative DiscoverUnit with optional capacity and memory state.

    The hydrological state is the conserved storage ``S``.  Optional finite
    capacity produces an explicit overflow flux.  Optional informational memory
    ``h`` follows the optional informational-memory update above but never contains
    water. It is independent of the route-kernel mechanism and does not create the
    eDD previous-process recurrence.

    Process contexts are assembled in this transparent order:

        selected drivers, S, [h], [deficit], routed upstream components

    where ``deficit = Smax - S``.  The deficit itself remains non-negative; a
    negative CRC orientation is used when the desired information channel is
    ``-deficit``.
    """

    def __init__(
        self,
        process_nets,
        smax=None,
        smax_bounds=None,
        context_indices=None,
        upstream_count=0,
        exchanges=None,
        memory_cell=None,
        include_deficit=False,
        name=None,
    ):
        super().__init__(name=name)

        if not process_nets:
            raise ValueError("DiscoverUnit requires at least one process network.")
        if not all(isinstance(net, keras.layers.Layer) for net in process_nets):
            raise TypeError("Process networks must be Keras layers.")
        if smax is not None and smax_bounds is not None:
            raise ValueError("Use either fixed smax or trainable smax_bounds, not both.")
        if smax is not None and float(smax) <= 0.0:
            raise ValueError("smax must be strictly positive or None.")
        if smax_bounds is not None:
            low, high = map(float, smax_bounds)
            if not (0.0 < low < high):
                raise ValueError("smax_bounds must satisfy 0 < low < high.")
            self._smax_bounds = (low, high)
            midpoint = 0.5 * (low + high)
            fraction = (midpoint - low) / (high - low)
            raw0 = np.log(fraction / (1.0 - fraction))
            self.raw_smax = self.add_weight(
                name="raw_smax",
                shape=(1,),
                initializer=keras.initializers.Constant(raw0),
                trainable=True,
            )
            self._fixed_smax = None
        else:
            self._smax_bounds = None
            self.raw_smax = None
            self._fixed_smax = None if smax is None else float(smax)

        self.process_nets = list(process_nets)
        self.memory_cell = memory_cell
        if memory_cell is not None and not isinstance(memory_cell, InformationalMemoryCell):
            raise TypeError("memory_cell must be InformationalMemoryCell or None.")
        self.include_deficit = bool(include_deficit)
        if self.include_deficit and self.smax is None:
            raise ValueError("Storage deficit can only be used when Smax exists.")

        if int(upstream_count) < 0:
            raise ValueError("upstream_count must be non-negative.")
        self.upstream_count = int(upstream_count)

        # Process-context structure is configuration, not build-time state.
        # Keep it available before the first traced call to `step`/`run_sequence`.
        if context_indices is None:
            self.context_indices = None
        else:
            if len(context_indices) != len(self.process_nets):
                raise ValueError("context_indices must contain one index list per process.")
            self.context_indices = [tuple(int(i) for i in indices) for indices in context_indices]
            extra = 1  # storage
            extra += 1 if self.memory_cell is not None else 0
            extra += 1 if self.include_deficit else 0
            extra += self.upstream_count
            for net, indices in zip(self.process_nets, self.context_indices):
                expected = len(indices) + extra
                if isinstance(net, CRCNet) and int(net.gamma.shape[0]) != expected:
                    raise ValueError(
                        f"CRCNet expects {int(net.gamma.shape[0])} variables, "
                        f"but its context contains {expected}."
                    )

        self.exchanges = list(exchanges or [])
        for exchange in self.exchanges:
            if not isinstance(exchange, keras.layers.Layer):
                raise TypeError("Boundary exchanges must be Keras layers.")
            if getattr(exchange, "kind", None) not in ("source", "sink"):
                raise ValueError("Each boundary exchange must declare kind='source' or kind='sink'.")
        self.sources = [exchange for exchange in self.exchanges if exchange.kind == "source"]
        self.sinks = [exchange for exchange in self.exchanges if exchange.kind == "sink"]

    def build(self, input_shape):
        """Mark the container layer built; child networks build lazily on first call.

        DiscoverUnit owns no direct kernel whose shape depends on ``input_shape``.
        Context configuration is validated in ``__init__`` so it is available before
        TensorFlow traces ``step`` or ``run_sequence``.
        """
        super().build(input_shape)

    @property
    def smax(self):
        if self._smax_bounds is not None:
            low, high = self._smax_bounds
            return low + tf.nn.sigmoid(self.raw_smax) * (high - low)
        return self._fixed_smax

    @property
    def has_memory(self):
        return self.memory_cell is not None

    @property
    def bounded(self):
        return self.smax is not None

    def _apply_exchanges(self, storage, inflow, drivers):
        batch_size = tf.shape(storage)[0]
        if self.sources:
            source_values = [source(drivers) for source in self.sources]
            for value in source_values:
                tf.debugging.assert_rank(value, 2, message="Source must return (B,1).")
                tf.debugging.assert_equal(tf.shape(value)[-1], 1, message="Source must return (B,1).")
            source_fluxes = tf.concat(source_values, axis=-1)
            tf.debugging.assert_all_finite(source_fluxes, "Source fluxes must be finite.")
            tf.debugging.assert_greater_equal(source_fluxes, 0.0, message="Sources cannot remove water.")
            source_total = tf.reduce_sum(source_fluxes, axis=-1, keepdims=True)
        else:
            source_fluxes = tf.zeros((batch_size, 0), dtype=tf.float32)
            source_total = tf.zeros_like(storage)
        available = storage + inflow + source_total

        if self.sinks:
            sink_values = [sink(drivers, storage, available) for sink in self.sinks]
            for value in sink_values:
                tf.debugging.assert_rank(value, 2, message="Sink must return (B,1).")
                tf.debugging.assert_equal(tf.shape(value)[-1], 1, message="Sink must return (B,1).")
            requests = tf.concat(sink_values, axis=-1)
            tf.debugging.assert_all_finite(requests, "Sink requests must be finite.")
            tf.debugging.assert_greater_equal(requests, 0.0, message="Sink requests cannot be negative.")
            request_total = tf.reduce_sum(requests, axis=-1, keepdims=True)
            scale = tf.minimum(1.0, tf.math.divide_no_nan(available, request_total))
            sink_fluxes = requests * scale
            sink_total = tf.reduce_sum(sink_fluxes, axis=-1, keepdims=True)
        else:
            sink_fluxes = tf.zeros((batch_size, 0), dtype=tf.float32)
            sink_total = tf.zeros_like(storage)
        resource = tf.maximum(available - sink_total, 0.0)
        return available, resource, source_fluxes, sink_fluxes, source_total, sink_total

    def call(self, storage, inflow, contexts, drivers=None):
        storage = tf.convert_to_tensor(storage, dtype=tf.float32)
        inflow = tf.convert_to_tensor(inflow, dtype=tf.float32)
        tf.debugging.assert_greater_equal(storage, 0.0, message="storage must be non-negative.")
        tf.debugging.assert_greater_equal(inflow, 0.0, message="inflow must be non-negative.")
        if self.smax is not None:
            tf.debugging.assert_less_equal(
                storage, self.smax + 1.0e-5,
                message="storage cannot exceed smax in a bounded DiscoverUnit.",
            )
        if len(contexts) != len(self.process_nets):
            raise ValueError(f"Expected {len(self.process_nets)} process contexts, got {len(contexts)}.")
        if self.exchanges and drivers is None:
            raise ValueError("drivers are required when boundary exchanges are configured.")

        if drivers is None:
            batch_size = tf.shape(storage)[0]
            available = storage + inflow
            resource = available
            source_fluxes = tf.zeros((batch_size, 0), dtype=tf.float32)
            sink_fluxes = tf.zeros((batch_size, 0), dtype=tf.float32)
            source_total = tf.zeros_like(storage)
            sink_total = tf.zeros_like(storage)
        else:
            drivers = tf.convert_to_tensor(drivers, dtype=tf.float32)
            available, resource, source_fluxes, sink_fluxes, source_total, sink_total = (
                self._apply_exchanges(storage, inflow, drivers)
            )

        process_values = [net(context) for net, context in zip(self.process_nets, contexts)]
        for value in process_values:
            tf.debugging.assert_rank(value, 2, message="Process network must return (B,1).")
            tf.debugging.assert_equal(tf.shape(value)[-1], 1, message="Process network must return (B,1).")
        propensities = tf.concat(process_values, axis=-1)
        storage_next, fluxes, overflow = conservative_allocation(
            resource, propensities, smax=self.smax
        )
        return {
            "available": available,
            "resource": resource,
            "storage": storage_next,
            "propensities": propensities,
            "fluxes": fluxes,
            "overflow": overflow,
            "source_fluxes": source_fluxes,
            "sink_fluxes": sink_fluxes,
            "source_total": source_total,
            "sink_total": sink_total,
        }

    def _build_contexts(self, drivers_t, storage, memory=None, upstream=None):
        if self.context_indices is None:
            raise ValueError("context_indices are required to build process contexts.")
        if upstream is None:
            upstream = tf.zeros((tf.shape(storage)[0], 0), dtype=tf.float32)
        tf.debugging.assert_equal(
            tf.shape(upstream)[-1], self.upstream_count,
            message="upstream component count does not match DiscoverUnit.upstream_count.",
        )
        if memory is None:
            memory = tf.zeros_like(storage)

        common = [storage]
        if self.memory_cell is not None:
            common.append(memory)
        if self.include_deficit:
            deficit = tf.maximum(tf.cast(self.smax, tf.float32) - storage, 0.0)
            common.append(deficit)
        common.append(upstream)

        contexts = []
        for indices in self.context_indices:
            selected = _select_columns(drivers_t, indices)
            contexts.append(tf.concat([selected, *common], axis=-1))
        return contexts

    def step(self, storage, inflow, drivers, upstream=None, memory=None):
        if memory is None:
            memory = tf.zeros_like(storage)
        contexts = self._build_contexts(
            drivers, storage, memory=memory, upstream=upstream
        )
        output = self(storage, inflow, contexts, drivers=drivers)
        if self.memory_cell is None:
            memory_next = tf.zeros_like(memory)
            memory_decay = tf.zeros_like(memory)
            memory_target = tf.zeros_like(memory)
        else:
            memory_next, memory_decay, memory_target = self.memory_cell(
                memory, drivers, storage
            )
        output["memory"] = memory_next
        output["memory_decay"] = memory_decay
        output["memory_target"] = memory_target
        return output

    @tf.function(reduce_retracing=True)
    def run_sequence(self, inflow, drivers, initial_storage, initial_memory=None):
        inflow = tf.convert_to_tensor(inflow, dtype=tf.float32)
        drivers = tf.convert_to_tensor(drivers, dtype=tf.float32)
        storage = tf.convert_to_tensor(initial_storage, dtype=tf.float32)
        if initial_memory is None:
            memory = tf.zeros_like(storage)
        else:
            memory = tf.convert_to_tensor(initial_memory, dtype=tf.float32)

        tf.debugging.assert_rank(inflow, 3)
        tf.debugging.assert_rank(drivers, 3)
        tf.debugging.assert_rank(storage, 2)
        tf.debugging.assert_equal(tf.shape(inflow)[:2], tf.shape(drivers)[:2])
        tf.debugging.assert_equal(tf.shape(inflow)[-1], 1)
        tf.debugging.assert_equal(tf.shape(storage)[-1], 1)
        tf.debugging.assert_equal(tf.shape(memory), tf.shape(storage))

        T = tf.shape(inflow)[1]
        n_processes = len(self.process_nets)
        n_sources = len(self.sources)
        n_sinks = len(self.sinks)
        arrays = {
            name: tf.TensorArray(tf.float32, size=T)
            for name in (
                "storage", "memory", "memory_decay", "memory_target", "fluxes",
                "overflow", "propensities", "available", "resource", "source_fluxes",
                "sink_fluxes", "source_total", "sink_total",
            )
        }

        for t in tf.range(T):
            output = self.step(storage, inflow[:, t, :], drivers[:, t, :], memory=memory)
            storage = output["storage"]
            memory = output["memory"]
            for name in arrays:
                arrays[name] = arrays[name].write(t, output[name])

        output = {name: tf.transpose(array.stack(), [1, 0, 2]) for name, array in arrays.items()}
        output["storage"].set_shape([None, None, 1])
        output["memory"].set_shape([None, None, 1])
        output["memory_decay"].set_shape([None, None, 1])
        output["memory_target"].set_shape([None, None, 1])
        output["fluxes"].set_shape([None, None, n_processes])
        output["overflow"].set_shape([None, None, 1])
        output["propensities"].set_shape([None, None, n_processes])
        output["available"].set_shape([None, None, 1])
        output["resource"].set_shape([None, None, 1])
        output["source_fluxes"].set_shape([None, None, n_sources])
        output["sink_fluxes"].set_shape([None, None, n_sinks])
        output["source_total"].set_shape([None, None, 1])
        output["sink_total"].set_shape([None, None, 1])
        return output


class StorageToHead(keras.layers.Layer):
    """
    Affine observation operator from one DU storage to piezometric head.

    Parameters
    ----------
    du_index
        DiscoverUnit whose storage is observed.
    scale
        Positive conversion factor from storage units to head units.
    offset
        Reference head added after the storage conversion.

    The physical model is not modified:

        head = offset + scale * storage
    """

    def __init__(self, du_index, scale=1.0, offset=0.0, name=None):
        super().__init__(name=name)

        if int(du_index) < 0:
            raise ValueError("du_index must be non-negative.")
        if scale <= 0.0:
            raise ValueError("scale must be strictly positive.")

        self.du_index = int(du_index)
        self.scale = float(scale)
        self.offset = float(offset)

    def call(self, physical_output):
        if "storage" not in physical_output:
            raise KeyError("StorageToHead requires physical_output['storage'].")

        storage = physical_output["storage"]
        tf.debugging.assert_rank(storage, 3)
        tf.debugging.assert_greater(
            tf.shape(storage)[-1],
            self.du_index,
            message="du_index is outside the available storage dimension.",
        )

        storage = storage[:, :, self.du_index:self.du_index + 1]
        return self.offset + self.scale * storage


class TrainableStorageToHead(keras.layers.Layer):
    """
    Affine storage-to-head operator with explicit parameter control.

    The physical model remains unchanged:

        head = offset + scale * storage

    `scale` stays strictly positive through a softplus transform. Scale and
    offset can be learned independently, which allows observation-operator
    identifiability to be tested directly.
    """

    def __init__(
        self,
        du_index,
        scale_init=1.0,
        offset_init=0.0,
        learn_scale=True,
        learn_offset=True,
        name=None,
    ):
        super().__init__(name=name)

        if int(du_index) < 0:
            raise ValueError("du_index must be non-negative.")
        if scale_init <= 0.0:
            raise ValueError("scale_init must be strictly positive.")

        self.du_index = int(du_index)
        self.scale_init = float(scale_init)
        self.offset_init = float(offset_init)
        self.learn_scale = bool(learn_scale)
        self.learn_offset = bool(learn_offset)

    def build(self, input_shape):
        raw_scale_init = np.log(np.expm1(self.scale_init))

        self.raw_scale = self.add_weight(
            name="raw_scale",
            shape=(),
            initializer=keras.initializers.Constant(raw_scale_init),
            trainable=self.learn_scale,
        )
        self.offset = self.add_weight(
            name="offset",
            shape=(),
            initializer=keras.initializers.Constant(self.offset_init),
            trainable=self.learn_offset,
        )

        super().build(input_shape)

    @property
    def scale(self):
        return tf.nn.softplus(self.raw_scale)

    def call(self, physical_output):
        if "storage" not in physical_output:
            raise KeyError(
                "TrainableStorageToHead requires physical_output['storage']."
            )

        storage = physical_output["storage"]
        tf.debugging.assert_rank(storage, 3)
        tf.debugging.assert_greater(
            tf.shape(storage)[-1],
            self.du_index,
            message="du_index is outside the available storage dimension.",
        )

        storage = storage[:, :, self.du_index:self.du_index + 1]
        return self.offset + self.scale * storage



def observe(physical_output, observation=None, target="outlet"):
    """
    Convert physical model output into the quantity compared with observations.

    Parameters
    ----------
    physical_output
        Dictionary returned by the physical model.
    observation
        Optional observation operator. None means use a physical output directly.
    target
        Physical-output key used when `observation` is None, e.g. "outlet".
    """
    if observation is None:
        if target not in physical_output:
            raise KeyError(f"physical_output does not contain '{target}'.")
        return physical_output[target]

    if not callable(observation):
        raise TypeError("observation must be None or a callable observation operator.")

    return observation(physical_output)


class RoutingKernel:
    """
    Fixed conservative discrete routing kernel.

    Parameters
    ----------
    weights
        Fractions arriving at lags 0, 1, ..., L. Values must be non-negative
        and sum to one. Example [0.2, 0.5, 0.3].
    """

    def __init__(self, weights):
        weights = np.asarray(weights, dtype=np.float32)

        if weights.ndim != 1 or weights.size == 0:
            raise ValueError("Routing weights must be a non-empty 1-D vector.")
        if np.any(weights < 0.0):
            raise ValueError("Routing weights must be non-negative.")
        if not np.isclose(weights.sum(), 1.0, atol=1e-6):
            raise ValueError("Routing weights must sum to one.")

        self.weights = weights

    @property
    def max_lag(self):
        return int(self.weights.size - 1)

    @property
    def mean_delay(self):
        lags = np.arange(self.weights.size, dtype=np.float32)
        return float(np.sum(lags * self.weights))


class MultiDUModel(keras.Model):
    is_dynamic_routing = False

    """
    Conservative acyclic network of DiscoverUnits with optional temporal routing.

    The DU list is already in topological order: an internal route can only
    point from DU i to a DU with a larger index.

    Parameters
    ----------
    discover_units
        Ordered DiscoverUnits forming the hydrological graph.
    destinations
        One destination per release process. Use an integer DU index for an
        internal transfer or "outlet" for catchment discharge.
    overflow_destinations
        One destination per DU for capacity overflow. Use None for unbounded
        DUs. A bounded DU must provide an integer downstream DU or "outlet".
    routing
        Optional nested list matching `destinations`. Each entry is None for
        instantaneous transfer or a RoutingKernel for temporal propagation.
    overflow_routing
        Optional routing specification for each DU overflow route.
    """

    def __init__(
        self,
        discover_units,
        destinations,
        overflow_destinations=None,
        routing=None,
        overflow_routing=None,
        name=None,
    ):
        super().__init__(name=name)

        if not discover_units:
            raise ValueError("MultiDUModel requires at least one DiscoverUnit.")
        if not all(isinstance(du, DiscoverUnit) for du in discover_units):
            raise TypeError("Every element of discover_units must be a DiscoverUnit.")

        self.discover_units = list(discover_units)
        self.n_du = len(self.discover_units)

        if len(destinations) != self.n_du:
            raise ValueError("destinations must contain one list per DiscoverUnit.")

        self.destinations = [list(row) for row in destinations]

        for i, (du, row) in enumerate(zip(self.discover_units, self.destinations)):
            if len(row) != len(du.process_nets):
                raise ValueError(
                    f"DU{i} has {len(du.process_nets)} processes but {len(row)} destinations."
                )
            for destination in row:
                self._validate_destination(i, destination)

        if overflow_destinations is None:
            overflow_destinations = [None] * self.n_du
        if len(overflow_destinations) != self.n_du:
            raise ValueError("overflow_destinations must contain one entry per DiscoverUnit.")

        self.overflow_destinations = list(overflow_destinations)

        for i, (du, destination) in enumerate(
            zip(self.discover_units, self.overflow_destinations)
        ):
            if du.smax is None:
                if destination is not None:
                    raise ValueError(f"DU{i} is unbounded, so its overflow destination must be None.")
            else:
                if destination is None:
                    raise ValueError(f"DU{i} is bounded and requires an overflow destination.")
                self._validate_destination(i, destination)

        if routing is None:
            routing = [[None] * len(row) for row in self.destinations]
        if len(routing) != self.n_du:
            raise ValueError("routing must contain one list per DiscoverUnit.")

        self.routing = [list(row) for row in routing]

        for i, (routes, destinations_i) in enumerate(zip(self.routing, self.destinations)):
            if len(routes) != len(destinations_i):
                raise ValueError(f"routing for DU{i} must contain one entry per process.")
            for kernel in routes:
                self._validate_routing(kernel)

        if overflow_routing is None:
            overflow_routing = [None] * self.n_du
        if len(overflow_routing) != self.n_du:
            raise ValueError("overflow_routing must contain one entry per DiscoverUnit.")

        self.overflow_routing = list(overflow_routing)

        for i, (destination, kernel) in enumerate(
            zip(self.overflow_destinations, self.overflow_routing)
        ):
            if destination is None and kernel is not None:
                raise ValueError(f"DU{i} has no overflow route, so overflow_routing must be None.")
            self._validate_routing(kernel)

        incoming_counts = [0] * self.n_du

        for row in self.destinations:
            for destination in row:
                if isinstance(destination, int):
                    incoming_counts[destination] += 1

        for destination in self.overflow_destinations:
            if isinstance(destination, int):
                incoming_counts[destination] += 1

        for i, (du, count) in enumerate(zip(self.discover_units, incoming_counts)):
            if du.upstream_count != count:
                raise ValueError(
                    f"DU{i} declares upstream_count={du.upstream_count}, "
                    f"but the topology provides {count} incoming routed components."
                )

        self.process_map = [
            (i, k)
            for i, du in enumerate(self.discover_units)
            for k in range(len(du.process_nets))
        ]
        self.n_processes = len(self.process_map)
        self.source_map = [(i, k, layer.name)
                           for i, du in enumerate(self.discover_units)
                           for k, layer in enumerate(du.sources)]
        self.sink_map = [(i, k, layer.name)
                         for i, du in enumerate(self.discover_units)
                         for k, layer in enumerate(du.sinks)]
        self.n_sources = len(self.source_map)
        self.n_sinks = len(self.sink_map)

        # ------------------------------------------------------------------
        # Canonical route registry
        # ------------------------------------------------------------------
        # Every *existing* hydrological route is represented by exactly one
        # RoutingKernel.  Instantaneous propagation is the explicit kernel
        # (1.0,), never a special numerical case hidden behind None.  None is
        # reserved only for an overflow route that does not exist because the
        # corresponding DU is unbounded.
        for i, row in enumerate(self.routing):
            self.routing[i] = [
                RoutingKernel((1.0,)) if kernel is None else kernel
                for kernel in row
            ]
        for i, destination in enumerate(self.overflow_destinations):
            if destination is not None and self.overflow_routing[i] is None:
                self.overflow_routing[i] = RoutingKernel((1.0,))

        self.route_map = []
        self.process_route_indices = []
        route_kernels = []

        for i, row in enumerate(self.destinations):
            indices = []
            for k, destination in enumerate(row):
                route_index = len(self.route_map)
                indices.append(route_index)
                self.route_map.append(("process", i, k, destination))
                route_kernels.append(self.routing[i][k])
            self.process_route_indices.append(indices)

        self.overflow_route_indices = [None] * self.n_du
        for i, destination in enumerate(self.overflow_destinations):
            if destination is None:
                continue
            route_index = len(self.route_map)
            self.overflow_route_indices[i] = route_index
            self.route_map.append(("overflow", i, None, destination))
            route_kernels.append(self.overflow_routing[i])

        self.n_routes = len(self.route_map)
        self._route_kernels = tuple(route_kernels)
        self._route_support_lengths = tuple(
            int(kernel.weights.size) for kernel in self._route_kernels
        )
        self.max_lag = max((length - 1 for length in self._route_support_lengths), default=0)

        fixed_weights = np.zeros(
            (self.n_routes, self.max_lag + 1), dtype=np.float32
        )
        for route, kernel in enumerate(self._route_kernels):
            fixed_weights[route, :kernel.weights.size] = kernel.weights
        self._fixed_route_weight_matrix = tf.constant(
            fixed_weights, dtype=tf.float32
        )

        self._validate_route_registry()

    def mark_built_for_checkpoint(self):
        """
        Mark the orchestration model as built after at least one sequence run.

        MultiDUModel is executed through run_sequence(); Keras call() is unused.
        Its child layers can therefore be fully built and trained while the
        parent Keras `built` flag remains False. This helper only updates that
        bookkeeping flag so Keras save_weights() can serialize existing weights.
        """
        if not self.weights:
            raise ValueError(
                "The model has no weights yet. Run at least one sequence before checkpointing."
            )
        self.built = True


    def _validate_destination(self, source, destination):
        if destination == "outlet":
            return
        if not isinstance(destination, int):
            raise TypeError("A destination must be an integer DU index or 'outlet'.")
        if destination < 0 or destination >= self.n_du:
            raise ValueError(f"Invalid destination DU index: {destination}.")
        if destination <= source:
            raise ValueError(
                "DUs must be supplied in topological order: internal routes must "
                "point to a larger DU index."
            )

    @staticmethod
    def _validate_routing(kernel):
        if kernel is not None and not isinstance(kernel, RoutingKernel):
            raise TypeError("Routing entries must be None or RoutingKernel objects.")

    def _validate_route_registry(self):
        """Validate the single canonical route representation used everywhere."""
        if len(self.route_map) != self.n_routes:
            raise AssertionError("route_map and n_routes disagree.")
        if len(self._route_kernels) != self.n_routes:
            raise AssertionError("Route-kernel registry and n_routes disagree.")
        if len(self._route_support_lengths) != self.n_routes:
            raise AssertionError("Route-support registry and n_routes disagree.")
        matrix = np.asarray(self._fixed_route_weight_matrix.numpy(), dtype=np.float64)
        if matrix.shape != (self.n_routes, self.max_lag + 1):
            raise AssertionError("Routing weight matrix has an inconsistent shape.")
        for route, length in enumerate(self._route_support_lengths):
            active = matrix[route, :length]
            padded = matrix[route, length:]
            if np.any(active < 0.0) or not np.isclose(active.sum(), 1.0, atol=1e-6):
                raise AssertionError(f"Route {route} is not a conservative kernel.")
            if padded.size and np.any(padded != 0.0):
                raise AssertionError(f"Route {route} has non-zero padded routing weights.")

    def route_weight_matrix(self):
        """Return the routing matrix actually used by the hydrological forward pass.

        Rows are routes and columns are lags 0..max_lag.  Subclasses may
        override this method (for example, to learn time-invariant kernels),
        but every caller uses this same interface.
        """
        return self._fixed_route_weight_matrix

    def route_weights_numpy(self):
        """Return unpadded route kernels in the exact route-map order."""
        matrix = np.asarray(self.route_weight_matrix().numpy(), dtype=np.float64)
        return tuple(
            tuple(float(v) for v in matrix[route, :length])
            for route, length in enumerate(self._route_support_lengths)
        )

    @property
    def route_weights(self):
        """Human/export view of one time-invariant fitted routing state."""
        return self.route_weights_numpy()

    def route_support_weights_numpy(self):
        """Return one kernel per route to define finite routing support.

        For fixed and learned time-invariant routing this is the fitted kernel.
        A dynamic-routing subclass can override this method with an initialization
        template while saving the actual per-time kernels separately.
        """
        return self.route_weights_numpy()

    def route_weights_for_step(
        self, route_index, source_storage, source_inflow, route_flow,
        source_memory, drivers=None,
    ):
        """Return one batch of kernels for a route at the current time step.

        The base implementation simply broadcasts the fixed/time-invariant
        kernel. Dynamic routing overrides this method without changing the water
        routing equation itself.
        """
        batch_size = tf.shape(route_flow)[0]
        del source_storage, source_inflow, source_memory, drivers
        weights = self.route_weight_matrix()[route_index]
        return tf.broadcast_to(weights[tf.newaxis, :], [batch_size, self.max_lag + 1])

    def _route_flow(self, flow, route_index, due, transit_state, weights):
        """Propagate one newly produced route flux with supplied batch kernels."""
        weights = tf.ensure_shape(
            tf.convert_to_tensor(weights, dtype=tf.float32),
            [None, self.max_lag + 1],
        )
        tf.debugging.assert_greater_equal(weights, 0.0)
        tf.debugging.assert_near(
            tf.reduce_sum(weights, axis=-1),
            tf.ones(tf.shape(weights)[0], dtype=weights.dtype),
            atol=2.0e-6,
            message="Every routing kernel must sum to one.",
        )
        arrival = due[:, route_index:route_index + 1] + flow * weights[:, 0:1]

        if self.max_lag > 0:
            delayed = flow * weights[:, 1:]
            route_mask = tf.one_hot(
                route_index, self.n_routes, dtype=tf.float32
            )[tf.newaxis, :, tf.newaxis]
            transit_state = transit_state + route_mask * delayed[:, tf.newaxis, :]

        return arrival, transit_state

    @tf.function(reduce_retracing=True)
    def run_sequence(
        self, external_inflow, drivers, initial_storage, initial_transit=None, initial_memory=None
    ):
        """
        Run the complete multi-DU graph over time.

        Parameters
        ----------
        external_inflow
            External conserved water entering each DU, shape (B,T,N_DU).
        drivers
            Shared exogenous predictors, shape (B,T,n).
        initial_storage
            Initial storage of all DUs, shape (B,N_DU).
        initial_transit
            Optional routing queue from a previous sequence, shape
            (B,N_route,max_lag). None starts with no water in transit.

        Returns
        -------
        storage
            DU storages, shape (B,T,N_DU).
        outlet
            Routed water reaching the outlet, shape (B,T,1).
        route_arrivals
            Arrival produced by each route after temporal propagation,
            shape (B,T,N_route).
        transit
            Water still in transit on each route after the current time step,
            shape (B,T,N_route).
        """
        external_inflow = tf.convert_to_tensor(external_inflow, dtype=tf.float32)
        drivers = tf.convert_to_tensor(drivers, dtype=tf.float32)
        storage = tf.convert_to_tensor(initial_storage, dtype=tf.float32)

        # Freeze the DU dimensions at tracing time. This gives an immediate,
        # readable error if external inflow or initial storage has the wrong
        # number of DU columns, instead of failing later inside the time loop.
        external_inflow = tf.ensure_shape(
            external_inflow,
            [None, None, self.n_du],
        )
        storage = tf.ensure_shape(
            storage,
            [None, self.n_du],
        )

        tf.debugging.assert_rank(external_inflow, 3)
        tf.debugging.assert_rank(drivers, 3)
        tf.debugging.assert_rank(storage, 2)
        tf.debugging.assert_equal(tf.shape(external_inflow)[:2], tf.shape(drivers)[:2])
        tf.debugging.assert_equal(tf.shape(external_inflow)[-1], self.n_du)
        tf.debugging.assert_equal(tf.shape(storage)[-1], self.n_du)
        tf.debugging.assert_equal(tf.shape(storage)[0], tf.shape(external_inflow)[0])
        tf.debugging.assert_greater_equal(
            external_inflow, 0.0, message="external_inflow must be non-negative."
        )

        batch_size = tf.shape(storage)[0]
        T = tf.shape(external_inflow)[1]

        if initial_memory is None:
            memory = tf.zeros((batch_size, self.n_du), dtype=tf.float32)
        else:
            memory = tf.convert_to_tensor(initial_memory, dtype=tf.float32)
            memory = tf.ensure_shape(memory, [None, self.n_du])
            tf.debugging.assert_equal(tf.shape(memory)[0], batch_size)

        if initial_transit is None:
            transit_state = tf.zeros(
                (batch_size, self.n_routes, self.max_lag), dtype=tf.float32
            )
        else:
            transit_state = tf.convert_to_tensor(initial_transit, dtype=tf.float32)
            tf.debugging.assert_rank(transit_state, 3)
            tf.debugging.assert_equal(tf.shape(transit_state)[0], batch_size)
            tf.debugging.assert_equal(tf.shape(transit_state)[1], self.n_routes)
            tf.debugging.assert_equal(tf.shape(transit_state)[2], self.max_lag)
            tf.debugging.assert_greater_equal(
                transit_state, 0.0, message="initial_transit must be non-negative."
            )

        storage_ta = tf.TensorArray(tf.float32, size=T)
        memory_ta = tf.TensorArray(tf.float32, size=T)
        memory_decay_ta = tf.TensorArray(tf.float32, size=T)
        memory_target_ta = tf.TensorArray(tf.float32, size=T)
        outlet_ta = tf.TensorArray(tf.float32, size=T)
        process_ta = tf.TensorArray(tf.float32, size=T)
        propensity_ta = tf.TensorArray(tf.float32, size=T)
        overflow_ta = tf.TensorArray(tf.float32, size=T)
        internal_ta = tf.TensorArray(tf.float32, size=T)
        source_ta = tf.TensorArray(tf.float32, size=T)
        sink_ta = tf.TensorArray(tf.float32, size=T)
        source_components_ta = tf.TensorArray(tf.float32, size=T)
        sink_components_ta = tf.TensorArray(tf.float32, size=T)
        arrival_ta = tf.TensorArray(tf.float32, size=T)
        transit_ta = tf.TensorArray(tf.float32, size=T)
        transit_total_ta = tf.TensorArray(tf.float32, size=T)
        routing_weights_ta = tf.TensorArray(tf.float32, size=T)

        for t in tf.range(T):
            drivers_t = drivers[:, t, :]
            external_t = external_inflow[:, t, :]
            previous_storage = tf.split(storage, self.n_du, axis=-1)
            previous_memory = tf.split(memory, self.n_du, axis=-1)

            if self.max_lag > 0:
                due = transit_state[:, :, 0]
                transit_state = tf.concat(
                    [
                        transit_state[:, :, 1:],
                        tf.zeros((batch_size, self.n_routes, 1), dtype=tf.float32),
                    ],
                    axis=-1,
                )
            else:
                due = tf.zeros((batch_size, self.n_routes), dtype=tf.float32)

            next_storage = list(previous_storage)
            next_memory = list(previous_memory)
            pending_total = [tf.zeros_like(previous_storage[i]) for i in range(self.n_du)]
            pending_components = [[] for _ in range(self.n_du)]

            outlet_t = tf.zeros_like(previous_storage[0])
            process_fluxes_t = []
            propensities_t = []
            overflow_t = []
            memory_decay_t = []
            memory_target_t = []
            source_total_t = []
            sink_total_t = []
            source_components_t = []
            sink_components_t = []
            internal_total_t = []
            route_arrivals_t = [tf.zeros_like(previous_storage[0]) for _ in range(self.n_routes)]
            route_weights_t = [None] * self.n_routes

            for i, du in enumerate(self.discover_units):
                internal_i = pending_total[i]

                if pending_components[i]:
                    upstream_i = tf.concat(pending_components[i], axis=-1)
                else:
                    upstream_i = tf.zeros((batch_size, 0), dtype=tf.float32)

                inflow_i = external_t[:, i:i + 1] + internal_i
                output = du.step(
                    previous_storage[i],
                    inflow_i,
                    drivers_t,
                    upstream=upstream_i,
                    memory=previous_memory[i],
                )

                next_storage[i] = output["storage"]
                next_memory[i] = output["memory"]
                internal_total_t.append(internal_i)
                process_fluxes_t.append(output["fluxes"])
                propensities_t.append(output["propensities"])
                overflow_t.append(output["overflow"])
                memory_decay_t.append(output["memory_decay"])
                memory_target_t.append(output["memory_target"])
                source_total_t.append(output["source_total"])
                sink_total_t.append(output["sink_total"])
                source_components_t.append(output["source_fluxes"])
                sink_components_t.append(output["sink_fluxes"])

                for k, destination in enumerate(self.destinations[i]):
                    flow = output["fluxes"][:, k:k + 1]
                    route_index = self.process_route_indices[i][k]

                    weights = self.route_weights_for_step(
                        route_index=route_index,
                        source_storage=previous_storage[i],
                        source_inflow=inflow_i,
                        route_flow=flow,
                        source_memory=previous_memory[i],
                        drivers=drivers_t,
                    )
                    arrival, transit_state = self._route_flow(
                        flow, route_index, due, transit_state, weights
                    )
                    route_arrivals_t[route_index] = arrival
                    route_weights_t[route_index] = weights

                    if destination == "outlet":
                        outlet_t = outlet_t + arrival
                    else:
                        pending_total[destination] = pending_total[destination] + arrival
                        pending_components[destination].append(arrival)

                overflow_destination = self.overflow_destinations[i]
                if overflow_destination is not None:
                    route_index = self.overflow_route_indices[i]
                    weights = self.route_weights_for_step(
                        route_index=route_index,
                        source_storage=previous_storage[i],
                        source_inflow=inflow_i,
                        route_flow=output["overflow"],
                        source_memory=previous_memory[i],
                        drivers=drivers_t,
                    )
                    arrival, transit_state = self._route_flow(
                        output["overflow"], route_index, due, transit_state, weights
                    )
                    route_arrivals_t[route_index] = arrival
                    route_weights_t[route_index] = weights

                    if overflow_destination == "outlet":
                        outlet_t = outlet_t + arrival
                    else:
                        pending_total[overflow_destination] = (
                            pending_total[overflow_destination] + arrival
                        )
                        pending_components[overflow_destination].append(arrival)

            storage = tf.concat(next_storage, axis=-1)
            memory = tf.concat(next_memory, axis=-1)
            route_transit = tf.reduce_sum(transit_state, axis=-1)
            transit_total = tf.reduce_sum(route_transit, axis=-1, keepdims=True)

            storage_ta = storage_ta.write(t, storage)
            memory_ta = memory_ta.write(t, memory)
            memory_decay_ta = memory_decay_ta.write(t, tf.concat(memory_decay_t, axis=-1))
            memory_target_ta = memory_target_ta.write(t, tf.concat(memory_target_t, axis=-1))
            outlet_ta = outlet_ta.write(t, outlet_t)
            process_ta = process_ta.write(t, tf.concat(process_fluxes_t, axis=-1))
            propensity_ta = propensity_ta.write(t, tf.concat(propensities_t, axis=-1))
            overflow_ta = overflow_ta.write(t, tf.concat(overflow_t, axis=-1))
            internal_ta = internal_ta.write(t, tf.concat(internal_total_t, axis=-1))
            source_ta = source_ta.write(t, tf.concat(source_total_t, axis=-1))
            sink_ta = sink_ta.write(t, tf.concat(sink_total_t, axis=-1))
            source_components_ta = source_components_ta.write(t, tf.concat(source_components_t, axis=-1))
            sink_components_ta = sink_components_ta.write(t, tf.concat(sink_components_t, axis=-1))
            arrival_ta = arrival_ta.write(t, tf.concat(route_arrivals_t, axis=-1))
            transit_ta = transit_ta.write(t, route_transit)
            transit_total_ta = transit_total_ta.write(t, transit_total)
            routing_weights_ta = routing_weights_ta.write(
                t, tf.stack(route_weights_t, axis=1)
            )

        output = {
            "storage": tf.transpose(storage_ta.stack(), [1, 0, 2]),
            "memory": tf.transpose(memory_ta.stack(), [1, 0, 2]),
            "memory_decay": tf.transpose(memory_decay_ta.stack(), [1, 0, 2]),
            "memory_target": tf.transpose(memory_target_ta.stack(), [1, 0, 2]),
            "outlet": tf.transpose(outlet_ta.stack(), [1, 0, 2]),
            "process_fluxes": tf.transpose(process_ta.stack(), [1, 0, 2]),
            "propensities": tf.transpose(propensity_ta.stack(), [1, 0, 2]),
            "overflow": tf.transpose(overflow_ta.stack(), [1, 0, 2]),
            "internal_inflow": tf.transpose(internal_ta.stack(), [1, 0, 2]),
            "source_total": tf.transpose(source_ta.stack(), [1, 0, 2]),
            "sink_total": tf.transpose(sink_ta.stack(), [1, 0, 2]),
            "source_fluxes": tf.transpose(source_components_ta.stack(), [1, 0, 2]),
            "sink_fluxes": tf.transpose(sink_components_ta.stack(), [1, 0, 2]),
            "route_arrivals": tf.transpose(arrival_ta.stack(), [1, 0, 2]),
            "transit": tf.transpose(transit_ta.stack(), [1, 0, 2]),
            "transit_total": tf.transpose(transit_total_ta.stack(), [1, 0, 2]),
            "routing_weights": tf.transpose(routing_weights_ta.stack(), [1, 0, 2, 3]),
            "final_transit_state": transit_state,
            "final_storage": storage,
            "final_memory": memory,
        }

        # All water in the graph is either stored, travelling, or leaving.
        current_total = tf.reduce_sum(output["storage"], axis=-1, keepdims=True) + output["transit_total"]
        first_total = tf.reduce_sum(initial_storage, axis=-1, keepdims=True) + tf.reduce_sum(
            tf.reshape(initial_transit, (batch_size, -1)), axis=-1, keepdims=True
        ) if initial_transit is not None else tf.reduce_sum(initial_storage, axis=-1, keepdims=True)
        previous_total = tf.concat([first_total[:, None, :], current_total[:, :-1, :]], axis=1)
        water_in = tf.reduce_sum(external_inflow, axis=-1, keepdims=True)
        source_in = tf.reduce_sum(output["source_total"], axis=-1, keepdims=True)
        sink_out = tf.reduce_sum(output["sink_total"], axis=-1, keepdims=True)
        output["mass_residual"] = previous_total + water_in + source_in - sink_out - output["outlet"] - current_total

        output["storage"].set_shape([None, None, self.n_du])
        output["memory"].set_shape([None, None, self.n_du])
        output["memory_decay"].set_shape([None, None, self.n_du])
        output["memory_target"].set_shape([None, None, self.n_du])
        output["outlet"].set_shape([None, None, 1])
        output["process_fluxes"].set_shape([None, None, self.n_processes])
        output["propensities"].set_shape([None, None, self.n_processes])
        output["overflow"].set_shape([None, None, self.n_du])
        output["internal_inflow"].set_shape([None, None, self.n_du])
        output["source_total"].set_shape([None, None, self.n_du])
        output["sink_total"].set_shape([None, None, self.n_du])
        output["source_fluxes"].set_shape([None, None, self.n_sources])
        output["sink_fluxes"].set_shape([None, None, self.n_sinks])
        output["route_arrivals"].set_shape([None, None, self.n_routes])
        output["transit"].set_shape([None, None, self.n_routes])
        output["transit_total"].set_shape([None, None, 1])
        output["routing_weights"].set_shape(
            [None, None, self.n_routes, self.max_lag + 1]
        )
        output["final_transit_state"].set_shape([None, self.n_routes, self.max_lag])
        output["final_storage"].set_shape([None, self.n_du])
        output["final_memory"].set_shape([None, self.n_du])
        output["mass_residual"].set_shape([None, None, 1])

        return output
