"""Objective functions and hydrological evaluation metrics."""
from __future__ import annotations

import numpy as np
import tensorflow as tf


def _values(y_true, y_pred, mask):
    obs = tf.convert_to_tensor(y_true, tf.float32)
    sim = tf.convert_to_tensor(y_pred, tf.float32)
    mask = tf.convert_to_tensor(mask, tf.float32)
    tf.debugging.assert_equal(tf.shape(obs), tf.shape(sim))
    tf.debugging.assert_equal(tf.shape(obs), tf.shape(mask))
    valid = tf.greater(mask, 0)
    tf.debugging.assert_positive(tf.reduce_sum(tf.cast(valid, tf.int32)),
                                 message="No scored observations are available.")
    obs = tf.boolean_mask(obs, valid)
    sim = tf.boolean_mask(sim, valid)
    tf.debugging.assert_all_finite(obs, "Scored observations must be finite.")
    tf.debugging.assert_all_finite(sim, "Predictions must be finite.")
    return obs, sim


def masked_mse(y_true, y_pred, mask):
    obs, sim = _values(y_true, y_pred, mask)
    return tf.reduce_mean(tf.square(obs - sim))


def masked_mae(y_true, y_pred, mask):
    obs, sim = _values(y_true, y_pred, mask)
    return tf.reduce_mean(tf.abs(obs - sim))


def masked_rmse(y_true, y_pred, mask):
    return tf.sqrt(masked_mse(y_true, y_pred, mask))


def masked_huber(y_true, y_pred, mask, delta=1.0):
    obs, sim = _values(y_true, y_pred, mask)
    delta = tf.cast(delta, tf.float32)
    err = tf.abs(obs - sim)
    return tf.reduce_mean(tf.where(err <= delta, 0.5 * tf.square(err), delta * (err - 0.5 * delta)))


def masked_nse(y_true, y_pred, mask):
    obs, sim = _values(y_true, y_pred, mask)
    denom = tf.reduce_sum(tf.square(obs - tf.reduce_mean(obs)))
    tf.debugging.assert_positive(denom, message="NSE undefined for constant observations.")
    return 1.0 - tf.reduce_sum(tf.square(obs - sim)) / denom


def hydro_metrics(y_true, y_pred, mask):
    """Hydrological scores and optional flow-duration-curve diagnostics.

    FDC diagnostics are meaningful for non-negative streamflow. PBIAS uses
    the sign convention 100*(observed-simulated)/observed.
    """
    y_true = np.asarray(y_true, np.float64)
    y_pred = np.asarray(y_pred, np.float64)
    mask = np.asarray(mask, np.float32)
    if y_true.shape != y_pred.shape or mask.shape != y_true.shape:
        raise ValueError("Observations, predictions and masks must have matching shapes.")
    good = mask > 0
    if not good.any():
        raise ValueError("No scored observations.")
    obs, sim = y_true[good], y_pred[good]
    if not np.isfinite(obs).all() or not np.isfinite(sim).all():
        raise ValueError("Scored observations and predictions must be finite.")
    error = sim - obs
    mse = float(np.mean(error**2))
    obs_mean, sim_mean = float(obs.mean()), float(sim.mean())
    obs_std, sim_std = float(obs.std()), float(sim.std())
    corr = float(np.corrcoef(obs, sim)[0, 1]) if len(obs) > 1 and obs_std > 0 and sim_std > 0 else np.nan
    nse = 1.0 - float(np.sum(error**2) / np.sum((obs - obs_mean)**2)) if obs_std > 0 else np.nan
    kge = (1.0 - float(np.sqrt((corr-1)**2+(sim_std/obs_std-1)**2+(sim_mean/obs_mean-1)**2))
           if np.isfinite(corr) and obs_std > 0 and obs_mean != 0 else np.nan)
    pbias = float(100 * np.sum(obs-sim) / np.sum(obs)) if obs.sum() != 0 else np.nan
    # Flow-duration-curve diagnostics; meaningful for non-negative flows.
    obs_fdc = np.sort(obs)[::-1]
    sim_fdc = np.sort(sim)[::-1]
    n = obs_fdc.size
    n_high = max(1, round(0.02*n))
    high_sum = obs_fdc[:n_high].sum()
    fhv = float(100*(sim_fdc[:n_high]-obs_fdc[:n_high]).sum()/high_sum) if high_sum != 0 else np.nan
    i20 = min(round(0.20*n), n-1)
    i70 = min(round(0.70*n), n-1)
    obs_log = np.log(np.maximum(obs_fdc, 1e-6))
    sim_log = np.log(np.maximum(sim_fdc, 1e-6))
    obs_slope = obs_log[i20] - obs_log[i70]
    sim_slope = sim_log[i20] - sim_log[i70]
    fms = float(100*(sim_slope-obs_slope)/(obs_slope+1e-6)) if abs(obs_slope) > 1e-12 else np.nan
    n_low = max(1, round(0.30*n))
    obs_low = obs_log[-n_low:]
    sim_low = sim_log[-n_low:]
    obs_low_vol = np.sum(obs_low-obs_low.min())
    sim_low_vol = np.sum(sim_low-sim_low.min())
    flv = (float(-100*(sim_low_vol-obs_low_vol)/(obs_low_vol+1e-6))
           if abs(obs_low_vol) > 1e-12 else np.nan)
    return {
        "nse": nse, "kge": kge, "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(error))), "pbias_percent": pbias,
        "pearson_r": corr, "pearson_r2": corr**2 if np.isfinite(corr) else np.nan,
        "nrmse_std": float(np.sqrt(mse)/obs_std) if obs_std > 0 else np.nan,
        "fhv_percent": fhv, "fms_percent": fms, "flv_percent": flv,
    }
