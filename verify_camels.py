"""Check a saved CAMELS-US example checkpoint without fitting the model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import keras
import numpy as np

from deepdiscover import (
    CRCNet, DeepDiscover, DemandLimitedSink, DifferentiableDegreeDaySnow,
    DiscoverUnit, MultiDUModel, load_camels_dataset,
)


def build_example_model():
    """Rebuild the two-DU configuration used in examples/04_camels_us.ipynb."""
    keras.utils.set_random_seed(42)
    efficiency = CRCNet([1], [100.0], hidden_units=16, output="sigmoid")
    first_du = DiscoverUnit(
        process_nets=[
            CRCNet([1, 1, 1], [10.0, 10.0, 100.0], hidden_units=16),
            CRCNet([1, 1, 1], [10.0, 10.0, 100.0], hidden_units=16),
        ],
        context_indices=[(0, 1), (0, 1)],
        exchanges=[DemandLimitedSink(efficiency, demand_index=2)],
    )
    second_du = DiscoverUnit(
        process_nets=[CRCNet([1, 1], [100.0, 10.0], hidden_units=16)],
        context_indices=[()],
        upstream_count=1,
    )
    return DeepDiscover(
        MultiDUModel(
            [first_du, second_du],
            destinations=[["outlet", 1], ["outlet"]],
        ),
        observations={"streamflow": "outlet"},
        snow=DifferentiableDegreeDaySnow(),
        snow_unit=0,
        temperature_index=2,
        driver_channels=("rain", "snowmelt", 1),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."),
                        help="Project directory containing data/ and results/")
    parser.add_argument("--basin", default="01013500")
    args = parser.parse_args()

    root = args.root
    record = root / "results" / "camels" / args.basin / "two_du_serial_seed42"
    checkpoint = record / "checkpoint"
    if not (checkpoint / "checkpoint.json").exists():
        parser.error(f"No saved checkpoint at {checkpoint}")
    if not (record / "history.json").exists():
        parser.error(f"No training history at {record / 'history.json'}")

    data = load_camels_dataset(root, args.basin, n_du=2,
                               start="1980-10-01", end="2010-09-30")
    train_end = 365 + 6940
    validation_end = train_end + 1826
    if len(data.water) != validation_end + 1826:
        raise ValueError("CAMELS period does not match the original notebook.")
    validation = data.continuous_batch(train_end, validation_end, warmup=365)

    model = build_example_model()
    model.predict(validation.water, validation.drivers)  # create all variables
    metadata = model.load_checkpoint(checkpoint)
    result = model.evaluate(validation)
    physical = model.simulate(validation.water, validation.drivers)
    history = json.loads((record / "history.json").read_text(encoding="utf-8"))
    selection = min((r for r in history if "validation_loss" in r),
                    key=lambda r: r["validation_loss"])

    saved_epoch = metadata["metadata"].get("selected_epoch")
    if saved_epoch != selection["epoch"]:
        raise AssertionError(f"Saved epoch {saved_epoch} differs from history {selection['epoch']}.")
    expected_loss = float(selection["validation_loss"])
    actual_loss = float(result["loss"])
    if not np.isclose(expected_loss, actual_loss, rtol=2e-4, atol=2e-4):
        raise AssertionError(
            f"Checkpoint validation loss {actual_loss:.8f} differs from "
            f"saved best {expected_loss:.8f}."
        )
    maximum_residual = float(np.max(np.abs(physical["mass_residual"].numpy())))
    if not np.isfinite(maximum_residual) or maximum_residual > 1e-3:
        raise AssertionError(f"Water-balance residual is too large: {maximum_residual:.3g} mm")

    print(f"Checkpoint: {checkpoint}")
    print(f"Selected epoch: {saved_epoch}")
    print(f"Validation MSE: {actual_loss:.7f} (recorded: {expected_loss:.7f})")
    print(f"Validation NSE: {result['metrics']['streamflow']['nse']:.6f}")
    print(f"Maximum water-balance residual: {maximum_residual:.4g} mm")
    print("Checkpoint verification passed.")


if __name__ == "__main__":
    main()
