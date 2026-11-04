"""Conservative learned routing options for Multiple-DD.

Two learned routing modes are provided and kept separate:

``TrainableRoutingMultiDUModel``
    Learns one conservative kernel per route. The fitted kernel is constant in
    time.

``DynamicRoutingMultiDUModel``
    Learns a conservative kernel that can change at every time step as a
    function of local hydrological context. Water routed at time ``t`` keeps
    the kernel assigned to it at that time, so delayed water and tracer mass can
    later use exactly the same lag fractions.

Neither class introduces the eDD previous-discharge recurrence. The routing
state is explicit water in transit.
"""
from __future__ import annotations

import numpy as np
import keras
import tensorflow as tf

from .model import MultiDUModel


class TrainableRoutingMultiDUModel(MultiDUModel):
    """Multiple-DD with learned conservative route kernels that are fixed in time."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        initial = np.asarray(
            self._fixed_route_weight_matrix.numpy(), dtype=np.float32
        )
        support = np.zeros_like(initial, dtype=bool)
        for route, length in enumerate(self._route_support_lengths):
            support[route, :length] = True

        logits = np.zeros_like(initial, dtype=np.float32)
        logits[support] = np.log(np.maximum(initial[support], 1.0e-8))

        self._routing_support = tf.constant(support, dtype=tf.bool)
        self._initial_route_weight_matrix = initial.copy()
        self.routing_logits = self.add_weight(
            name="routing_logits",
            shape=logits.shape,
            initializer=keras.initializers.Constant(logits),
            trainable=True,
        )

    def route_weight_matrix(self):
        """Return learned route kernels in the canonical padded matrix shape."""
        minus_large = tf.cast(-1.0e9, self.routing_logits.dtype)
        masked_logits = tf.where(
            self._routing_support,
            self.routing_logits,
            minus_large,
        )
        weights = tf.nn.softmax(masked_logits, axis=-1)
        return tf.where(self._routing_support, weights, tf.zeros_like(weights))

    def initial_route_weights_numpy(self):
        """Return the unpadded kernels used to initialize optimization."""
        matrix = np.asarray(self._initial_route_weight_matrix, dtype=np.float64)
        return tuple(
            tuple(float(v) for v in matrix[route, :length])
            for route, length in enumerate(self._route_support_lengths)
        )


class _DynamicRoutingHead(keras.layers.Layer):
    """Small route-specific network that returns one conservative kernel."""

    def __init__(self, initial_kernel, hidden_units=8, seed=42, **kwargs):
        super().__init__(**kwargs)
        initial = np.asarray(initial_kernel, dtype=np.float32)
        if initial.ndim != 1 or initial.size < 1:
            raise ValueError("A routing kernel must contain at least one lag.")
        if np.any(initial < 0.0) or not np.isclose(initial.sum(), 1.0, atol=1.0e-6):
            raise ValueError(
                "Dynamic routing kernels must be non-negative and sum to one."
            )
        if int(hidden_units) < 1:
            raise ValueError("DYNAMIC_ROUTING_HIDDEN_UNITS must be at least 1.")

        self.initial_kernel = initial
        self.hidden_units = int(hidden_units)
        self.instantaneous = initial.size == 1
        if self.instantaneous:
            return
        # A small positive floor permits a zero initial tap while preserving
        # the conservative softmax parameterization.
        initial = np.maximum(initial, 1.0e-8)
        initial = initial / initial.sum()
        self.hidden = keras.layers.Dense(
            self.hidden_units,
            activation="tanh",
            kernel_initializer=keras.initializers.GlorotUniform(seed=seed),
            bias_initializer="zeros",
            name=f"{self.name}_hidden",
        )
        # A zero output kernel makes the initial routing exactly equal to the
        # user-specified constant kernel. Training can then introduce temporal
        # variation only when it improves the quantity objective.
        self.logits = keras.layers.Dense(
            initial.size,
            activation=None,
            kernel_initializer="zeros",
            bias_initializer=keras.initializers.Constant(np.log(initial)),
            name=f"{self.name}_logits",
        )

    def call(self, context):
        if self.instantaneous:
            return tf.ones((tf.shape(context)[0], 1), dtype=tf.float32)
        return tf.nn.softmax(self.logits(self.hidden(context)), axis=-1)


class DynamicRoutingMultiDUModel(MultiDUModel):
    """Multiple-DD with state-dependent conservative routing kernels.

    At each time step, every route receives a kernel over lags 0..L. The
    kernel is produced from local source-DU context and sums to one through a
    softmax. The resulting fractions are applied to the water produced during
    that same time step and are stored in the explicit routing queue.

    This is deliberately different from the eDD recurrence. With ``L=1`` the
    kernel has two taps, ``[g0(t), g1(t)]``. The lag-1 fraction can be compared
    qualitatively with a time-varying routing coefficient, but it routes newly
    produced water without a recurrence in previous routed discharge.
    """

    is_dynamic_routing = True
    _ALLOWED_CONTEXT = ("storage", "inflow", "flow", "memory")

    def __init__(
        self,
        *args,
        routing_context=("storage", "inflow", "flow"),
        routing_hidden_units=8,
        routing_storage_ref=100.0,
        routing_flow_ref=1.0,
        routing_memory_ref=1.0,
        routing_seed=42,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        context = tuple(str(v).lower() for v in routing_context)
        if not context:
            raise ValueError("DYNAMIC_ROUTING_CONTEXT must contain at least one variable.")
        unknown = [name for name in context if name not in self._ALLOWED_CONTEXT]
        if unknown:
            raise ValueError(
                "Unknown dynamic routing context variable(s): " + ", ".join(unknown)
            )
        if routing_storage_ref <= 0.0 or routing_flow_ref <= 0.0 or routing_memory_ref <= 0.0:
            raise ValueError("Dynamic routing reference scales must be strictly positive.")

        self.routing_context = context
        self.routing_hidden_units = int(routing_hidden_units)
        self.routing_storage_ref = float(routing_storage_ref)
        self.routing_flow_ref = float(routing_flow_ref)
        self.routing_memory_ref = float(routing_memory_ref)
        self.routing_seed = int(routing_seed)

        initial = np.asarray(self._fixed_route_weight_matrix.numpy(), dtype=np.float32)
        self._dynamic_initial_route_weight_matrix = initial.copy()
        heads = []
        for route, length in enumerate(self._route_support_lengths):
            kernel = initial[route, :length]
            heads.append(
                _DynamicRoutingHead(
                    initial_kernel=kernel,
                    hidden_units=self.routing_hidden_units,
                    seed=self.routing_seed + 97 * route,
                    name=f"dynamic_route_{route}",
                )
            )
        self.routing_heads = heads

    @staticmethod
    def _scale_nonnegative(value, reference):
        value = tf.maximum(tf.convert_to_tensor(value, tf.float32), 0.0)
        ref = tf.cast(reference, value.dtype)
        return value / (value + ref)

    def _routing_context_tensor(self, storage, inflow, flow, memory):
        pieces = []
        for name in self.routing_context:
            if name == "storage":
                pieces.append(self._scale_nonnegative(storage, self.routing_storage_ref))
            elif name == "inflow":
                pieces.append(self._scale_nonnegative(inflow, self.routing_flow_ref))
            elif name == "flow":
                pieces.append(self._scale_nonnegative(flow, self.routing_flow_ref))
            elif name == "memory":
                pieces.append(self._scale_nonnegative(memory, self.routing_memory_ref))
        return tf.concat(pieces, axis=-1)

    def route_weights_for_step(
        self,
        route_index,
        source_storage,
        source_inflow,
        route_flow,
        source_memory,
        drivers=None,
    ):
        """Return the time-varying kernel for one route and one batch step."""
        del drivers
        context = self._routing_context_tensor(
            source_storage, source_inflow, route_flow, source_memory
        )
        active = self.routing_heads[route_index](context)
        pad = self.max_lag + 1 - self._route_support_lengths[route_index]
        if pad > 0:
            active = tf.pad(active, [[0, 0], [0, pad]])
        return active

    def route_weights_numpy(self):
        raise RuntimeError(
            "Dynamic routing has no single fitted kernel. Use the saved per-time "
            "routing_weights trajectory or route_support_weights_numpy()."
        )

    def route_support_weights_numpy(self):
        """Return initial kernels only to define route support in transport metadata."""
        matrix = np.asarray(self._dynamic_initial_route_weight_matrix, dtype=np.float64)
        return tuple(
            tuple(float(v) for v in matrix[route, :length])
            for route, length in enumerate(self._route_support_lengths)
        )
