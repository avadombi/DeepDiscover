"""Explicit comparison of alternative conservative model structures."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from math import prod
import math
import json
from pathlib import Path

from .api import DeepDiscover
from .dataset import HydroBatch


@dataclass
class Candidate:
    """A named model family with a function that creates a fresh model."""
    name: str
    build: Callable[[], DeepDiscover]
    properties: dict[str, bool] | None = None


def compare_structures(candidates: list[Candidate], train: HydroBatch,
                       validation: HydroBatch, *, seeds=(42, 200, 22),
                       epochs=100, batch_size=8, learning_rate=1e-3,
                       loss="mse", weights=None, patience=15,
                       normalize=True, mass_tolerance=1e-3):
    """Fit each candidate on training windows and report validation scores.

    Each candidate is reinitialized for every random seed. The validation
    target is used to select training checkpoints. No independent test period
    is used by this function. Callers must interpret structural equivalence
    using a separately specified criterion and suitable optimization checks.

    Returns a list of dictionaries, one for each candidate and seed.
    """
    import tensorflow as tf

    if not candidates or not seeds:
        raise ValueError("At least one candidate and one seed are required.")
    names = [candidate.name for candidate in candidates]
    if len(names) != len(set(names)):
        raise ValueError("Candidate names must be unique.")
    results = []
    for candidate in candidates:
        for seed in seeds:
            tf.keras.utils.set_random_seed(int(seed))
            model = candidate.build()
            if not isinstance(model, DeepDiscover):
                raise TypeError(f"Builder for {candidate.name} must return DeepDiscover.")
            history = model.fit(
                train, validation, epochs=epochs, batch_size=batch_size,
                learning_rate=learning_rate, loss=loss, weights=weights,
                normalize=normalize, patience=patience, verbose=0, seed=int(seed),
            )
            score = model.evaluate(validation, loss=loss, weights=weights)
            physical = model.model
            result = {
                "candidate": candidate.name,
                "seed": int(seed),
                "properties": dict(candidate.properties or {}),
                "epochs": len(history),
                "n_du": physical.n_du,
                "n_processes": physical.n_processes,
                "n_routes": physical.n_routes,
                "n_parameters": sum(prod(v.shape) for v in model._variables()),
                "validation_loss": score["loss"],
                "validation_metrics": score["metrics"],
                "max_abs_mass_residual": score["max_abs_mass_residual"],
                "conservative": score["max_abs_mass_residual"] <= mass_tolerance,
            }
            results.append(result)
    return results


def equivalent_candidates(results, reference, loss_tolerance):
    """Identify validation-equivalent runs under a declared absolute loss margin.

    ``reference`` is the name of a reference candidate. Lower loss is better.
    Equivalence here is numerical and experiment-relative; it does not imply
    uniqueness of a physical topology or proof of optimization adequacy.
    """
    if loss_tolerance < 0:
        raise ValueError("loss_tolerance must be non-negative.")
    reference_scores = [r["validation_loss"] for r in results
                        if r["candidate"] == reference and r["conservative"]]
    if not reference_scores:
        raise ValueError("No conservative reference results were found.")
    ceiling = min(reference_scores) + loss_tolerance
    return [{**row, "validation_equivalent": bool(row["conservative"] and
            row["validation_loss"] <= ceiling)} for row in results]


def save_comparison(results, path):
    """Write a JSON record of candidate definitions and validation scores."""
    def finite(value):
        if isinstance(value, dict):
            return {key: finite(item) for key, item in value.items()}
        if isinstance(value, list):
            return [finite(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    Path(path).write_text(json.dumps(finite(results), indent=2, allow_nan=False), encoding="utf-8")
