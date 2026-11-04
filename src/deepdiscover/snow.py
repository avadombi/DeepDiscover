import math

import keras
import tensorflow as tf


def _logit(value):
    value = min(max(float(value), 1.0e-6), 1.0 - 1.0e-6)
    return math.log(value / (1.0 - value))


class DifferentiableDegreeDaySnow(keras.layers.Layer):
    """
    Conservative smooth degree-day snow module.

    This is the current EXP-HYDRO-style option. Other snow modules can replace
    it as long as they return the same physical output keys.

    Parameters
    ----------
    snowfall_temperature
        Temperature around which precipitation changes from snow to rain.
    melt_temperature
        Temperature above which degree-day melt becomes active.
    degree_day_factor
        Potential melt per positive degree-day, in mm / (degC day).
    transition_sharpness
        Sharpness of the differentiable temperature thresholds.
    """

    def __init__(
        self,
        snowfall_temperature=-1.5,
        melt_temperature=1.5,
        degree_day_factor=2.5,
        snowfall_temperature_bounds=(-3.0, 1.0),
        melt_temperature_bounds=(-1.0, 4.0),
        degree_day_factor_bounds=(0.05, 8.0),
        transition_sharpness=5.0,
        trainable_snowfall_temperature=True,
        trainable_melt_temperature=True,
        trainable_degree_day_factor=True,
        name=None,
    ):
        super().__init__(name=name)

        self.t_snow_bounds = tuple(float(v) for v in snowfall_temperature_bounds)
        self.t_melt_bounds = tuple(float(v) for v in melt_temperature_bounds)
        self.ddf_bounds = tuple(float(v) for v in degree_day_factor_bounds)

        self.t_snow_initial = float(snowfall_temperature)
        self.t_melt_initial = float(melt_temperature)
        self.ddf_initial = float(degree_day_factor)

        self.transition_sharpness = float(transition_sharpness)

        self.trainable_t_snow = bool(trainable_snowfall_temperature)
        self.trainable_t_melt = bool(trainable_melt_temperature)
        self.trainable_ddf = bool(trainable_degree_day_factor)

        self._validate_parameter(
            self.t_snow_initial, self.t_snow_bounds, "snowfall_temperature"
        )
        self._validate_parameter(
            self.t_melt_initial, self.t_melt_bounds, "melt_temperature"
        )
        self._validate_parameter(
            self.ddf_initial, self.ddf_bounds, "degree_day_factor"
        )

        if self.transition_sharpness <= 0.0:
            raise ValueError("transition_sharpness must be strictly positive.")

    @staticmethod
    def _validate_parameter(value, bounds, name):
        lower, upper = bounds
        if upper <= lower:
            raise ValueError(f"{name} upper bound must exceed lower bound.")
        if not lower <= value <= upper:
            raise ValueError(f"{name} initial value must lie inside its bounds.")

    @staticmethod
    def _initial_raw(value, bounds):
        lower, upper = bounds
        fraction = (value - lower) / (upper - lower)
        return _logit(fraction)

    def build(self, input_shape):
        self.raw_t_snow = self.add_weight(
            name="raw_snowfall_temperature",
            shape=(),
            initializer=keras.initializers.Constant(
                self._initial_raw(self.t_snow_initial, self.t_snow_bounds)
            ),
            trainable=self.trainable_t_snow,
        )

        self.raw_t_melt = self.add_weight(
            name="raw_melt_temperature",
            shape=(),
            initializer=keras.initializers.Constant(
                self._initial_raw(self.t_melt_initial, self.t_melt_bounds)
            ),
            trainable=self.trainable_t_melt,
        )

        self.raw_ddf = self.add_weight(
            name="raw_degree_day_factor",
            shape=(),
            initializer=keras.initializers.Constant(
                self._initial_raw(self.ddf_initial, self.ddf_bounds)
            ),
            trainable=self.trainable_ddf,
        )

        super().build(input_shape)

    @staticmethod
    def _bounded(raw, bounds):
        lower, upper = bounds
        return lower + (upper - lower) * tf.sigmoid(raw)

    @property
    def snowfall_temperature(self):
        return self._bounded(self.raw_t_snow, self.t_snow_bounds)

    @property
    def melt_temperature(self):
        return self._bounded(self.raw_t_melt, self.t_melt_bounds)

    @property
    def degree_day_factor(self):
        return self._bounded(self.raw_ddf, self.ddf_bounds)

    def parameter_values(self):
        return {
            "snowfall_temperature": self.snowfall_temperature,
            "melt_temperature": self.melt_temperature,
            "degree_day_factor": self.degree_day_factor,
        }

    def _heaviside(self, value):
        return 0.5 * (tf.tanh(self.transition_sharpness * value) + 1.0)

    def _step(self, precipitation, temperature, previous_storage):
        precipitation = tf.maximum(precipitation, 0.0)

        snow_fraction = self._heaviside(
            self.snowfall_temperature - temperature
        )

        snowfall = precipitation * snow_fraction
        rain = precipitation - snowfall

        available_snow = previous_storage + snowfall

        thermal_excess = tf.nn.relu(
            temperature - self.melt_temperature
        )

        potential_melt = (
            self._heaviside(temperature - self.melt_temperature)
            * self.degree_day_factor
            * thermal_excess
        )

        snowmelt = tf.minimum(available_snow, potential_melt)
        storage = available_snow - snowmelt
        liquid_water = rain + snowmelt

        residual = (
            storage + liquid_water
            - previous_storage - precipitation
        )

        return storage, rain, snowfall, snowmelt, liquid_water, residual

    def call(self, inputs, initial_storage=None):
        precipitation = tf.cast(tf.convert_to_tensor(inputs["precipitation"]), tf.float32)
        temperature = tf.cast(tf.convert_to_tensor(inputs["temperature"]), tf.float32)

        tf.debugging.assert_rank(precipitation, 3)
        tf.debugging.assert_rank(temperature, 3)
        tf.debugging.assert_equal(
            tf.shape(precipitation), tf.shape(temperature)
        )
        tf.debugging.assert_equal(tf.shape(precipitation)[-1], 1)
        tf.debugging.assert_greater_equal(
            precipitation, 0.0, message="precipitation must be non-negative."
        )

        batch_size = tf.shape(precipitation)[0]
        T = tf.shape(precipitation)[1]

        if initial_storage is None:
            storage = tf.zeros((batch_size,), dtype=tf.float32)
        else:
            storage = tf.cast(tf.convert_to_tensor(initial_storage), tf.float32)
            storage = tf.reshape(storage, (batch_size,))
            tf.debugging.assert_greater_equal(
                storage, 0.0, message="initial snow storage must be non-negative."
            )

        P = precipitation[:, :, 0]
        temperature = temperature[:, :, 0]

        rain_ta = tf.TensorArray(tf.float32, size=T)
        snowfall_ta = tf.TensorArray(tf.float32, size=T)
        snowmelt_ta = tf.TensorArray(tf.float32, size=T)
        liquid_ta = tf.TensorArray(tf.float32, size=T)
        storage_ta = tf.TensorArray(tf.float32, size=T)
        residual_ta = tf.TensorArray(tf.float32, size=T)

        # An explicit TensorFlow loop works inside Keras calls in graph mode.
        def update(t, storage, rain_ta, snowfall_ta, snowmelt_ta,
                   liquid_ta, storage_ta, residual_ta):
            storage, rain, snowfall, snowmelt, liquid, residual = self._step(
                P[:, t], temperature[:, t], storage
            )
            return (
                t + 1, storage,
                rain_ta.write(t, rain),
                snowfall_ta.write(t, snowfall),
                snowmelt_ta.write(t, snowmelt),
                liquid_ta.write(t, liquid),
                storage_ta.write(t, storage),
                residual_ta.write(t, residual),
            )

        (_, storage, rain_ta, snowfall_ta, snowmelt_ta,
         liquid_ta, storage_ta, residual_ta) = tf.while_loop(
            lambda t, *_: t < T,
            update,
            (tf.constant(0, tf.int32), storage, rain_ta, snowfall_ta,
             snowmelt_ta, liquid_ta, storage_ta, residual_ta),
            parallel_iterations=1,
        )

        def btc(array):
            return tf.transpose(array.stack(), [1, 0])[:, :, None]

        return {
            "liquid_water": btc(liquid_ta),
            "rain": btc(rain_ta),
            "snowfall": btc(snowfall_ta),
            "snowmelt": btc(snowmelt_ta),
            "snow_storage": btc(storage_ta),
            "mass_residual": btc(residual_ta),
            "final_snow_storage": storage[:, None],
            **self.parameter_values(),
        }


def apply_snow(snow, precipitation, temperature=None, initial_storage=None):
    """
    Apply an optional snow module.

    `snow=None` means all precipitation is immediately liquid.

    Any alternative snow object is accepted if it is callable and returns:
    liquid_water, rain, snowfall, snowmelt, snow_storage,
    mass_residual, and final_snow_storage.
    """
    precipitation = tf.cast(tf.convert_to_tensor(precipitation), tf.float32)

    if snow is None:
        zero = tf.zeros_like(precipitation)
        batch_size = tf.shape(precipitation)[0]

        return {
            "liquid_water": precipitation,
            "rain": precipitation,
            "snowfall": zero,
            "snowmelt": zero,
            "snow_storage": zero,
            "mass_residual": zero,
            "final_snow_storage": tf.zeros((batch_size, 1), dtype=tf.float32),
        }

    if temperature is None:
        raise ValueError("temperature is required when a snow module is active.")
    if not callable(snow):
        raise TypeError("snow must be None or a callable snow module.")

    output = snow(
        {
            "precipitation": precipitation,
            "temperature": temperature,
        },
        initial_storage=initial_storage,
    )

    required = (
        "liquid_water",
        "rain",
        "snowfall",
        "snowmelt",
        "snow_storage",
        "mass_residual",
        "final_snow_storage",
    )

    missing = [name for name in required if name not in output]
    if missing:
        raise KeyError(
            "snow module is missing required output(s): "
            + ", ".join(missing)
        )

    return output
