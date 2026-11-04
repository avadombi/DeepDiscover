from __future__ import annotations

from pathlib import Path

import numpy as np


CFS_TO_MM3 = 28_316_846.592
SECONDS_PER_DAY = 86_400.0


def hamon_pet(daylength_seconds: np.ndarray, temperature_c: np.ndarray) -> np.ndarray:
    """Hamon potential evapotranspiration [mm/day]."""
    daylength_hours = np.asarray(daylength_seconds, dtype=np.float64) / 3600.0
    temperature_c = np.asarray(temperature_c, dtype=np.float64)

    pet = (
        29.8 * daylength_hours * 0.611
        * np.exp(17.3 * temperature_c / (temperature_c + 237.3))
        / (temperature_c + 273.2)
    )
    return pet.astype(np.float32)


def _validate_basin_id(basin_id: str) -> str:
    basin_id = str(basin_id)
    if len(basin_id) != 8 or not basin_id.isdigit():
        raise ValueError("basin_id must contain exactly 8 digits, e.g. '01013500'.")
    return basin_id


def _camels_paths(project_root: str | Path, basin_id: str) -> tuple[Path, Path]:
    basin_dir = Path(project_root) / "data" / "camels" / basin_id
    forcing = basin_dir / f"{basin_id}_lump_cida_forcing_leap.txt"
    flow = basin_dir / f"{basin_id}_streamflow_qc.txt"

    missing = [path for path in (forcing, flow) if not path.exists()]
    if missing:
        message = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError("Missing CAMELS-US file(s):\n" + message)

    return forcing, flow


def _check_tensor_pair(X: np.ndarray, q: np.ndarray) -> None:
    if X.ndim != 3:
        raise ValueError(f"X must have shape (B,T,n), got {X.shape}.")
    if q.ndim != 3:
        raise ValueError(f"q must have shape (B,T,p), got {q.shape}.")
    if X.shape[:2] != q.shape[:2]:
        raise ValueError(f"X and q must share B and T, got {X.shape} and {q.shape}.")
    if not np.isfinite(X).all():
        raise ValueError("X contains non-finite values.")
    if np.isinf(q).any():
        raise ValueError("q contains infinite values.")


def load_camels(
    project_root: str | Path,
    basin_id: str,
    features: tuple[str, ...] = ("P", "T", "PET"),
    start: str | None = None,
    end: str | None = None,
) -> dict:
    """Load one continuous CAMELS-US series as X(1,T,n) and q(1,T,1)."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("CAMELS loading requires pandas; install deepdiscover-hydro[examples].") from exc
    basin_id = _validate_basin_id(basin_id)
    forcing_path, flow_path = _camels_paths(project_root, basin_id)

    forcing = pd.read_csv(forcing_path, sep=r"\s+", header=0, skiprows=3, engine="python")
    forcing = forcing.rename(columns={"Mnth": "Month"})

    required = {"Year", "Month", "Day", "dayl(s)", "prcp(mm/day)", "tmax(C)", "tmin(C)"}
    missing = required.difference(forcing.columns)
    if missing:
        raise ValueError(f"Missing forcing columns: {sorted(missing)}")

    forcing["date"] = pd.to_datetime(
        {"year": forcing["Year"], "month": forcing["Month"], "day": forcing["Day"]}
    )
    forcing["P"] = forcing["prcp(mm/day)"].astype(np.float32)
    forcing["T"] = (forcing["tmax(C)"] + forcing["tmin(C)"]) / 2.0
    forcing["PET"] = hamon_pet(forcing["dayl(s)"].to_numpy(), forcing["T"].to_numpy())

    with forcing_path.open("r", encoding="utf-8") as handle:
        metadata = [handle.readline().strip() for _ in range(3)]

    area_m2 = float(metadata[2])
    if not np.isfinite(area_m2) or area_m2 <= 0.0:
        raise ValueError("Basin area must be positive.")

    flow = pd.read_csv(
        flow_path,
        sep=r"\s+",
        names=["Id", "Year", "Month", "Day", "Q_cfs", "QC"],
        header=None,
        engine="python",
    )
    flow["date"] = pd.to_datetime(
        {"year": flow["Year"], "month": flow["Month"], "day": flow["Day"]}
    )
    # CAMELS uses -999 and quality flag M for missing streamflow.
    missing_discharge = flow["Q_cfs"].eq(-999.0) | flow["QC"].astype(str).eq("M")
    invalid_discharge = flow["Q_cfs"].lt(0.0) & ~missing_discharge
    if invalid_discharge.any():
        raise ValueError("Streamflow contains unexpected negative values.")
    flow["Q"] = (
        CFS_TO_MM3 * SECONDS_PER_DAY * flow["Q_cfs"].astype(np.float64) / (area_m2 * 1.0e6)
    ).astype(np.float32)
    flow.loc[missing_discharge, "Q"] = np.nan

    frame = pd.merge(
        forcing[["date", "P", "T", "PET"]],
        flow[["date", "Q", "QC"]],
        on="date",
        how="inner",
    )

    if start is not None:
        frame = frame.loc[frame["date"] >= pd.Timestamp(start)]
    if end is not None:
        frame = frame.loc[frame["date"] <= pd.Timestamp(end)]

    frame = frame.sort_values("date").reset_index(drop=True)
    if frame.empty:
        raise ValueError("Requested CAMELS period is empty.")

    dates = frame["date"].to_numpy(dtype="datetime64[D]")
    if dates.size > 1:
        dt = np.diff(dates).astype("timedelta64[D]").astype(int)
        if np.any(dt != 1):
            raise ValueError("CAMELS series is not continuous at daily resolution.")

    allowed = {"P", "T", "PET"}
    unknown = set(features) - allowed
    if unknown:
        raise ValueError(f"Unknown feature(s): {sorted(unknown)}. Allowed: P, T, PET.")
    if not features:
        raise ValueError("At least one feature is required.")

    X = np.stack([frame[name].to_numpy(dtype=np.float32) for name in features], axis=-1)[None, ...]
    q = frame["Q"].to_numpy(dtype=np.float32)[None, :, None]
    _check_tensor_pair(X, q)

    if "P" in features and np.any(frame["P"].to_numpy() < 0.0):
        raise ValueError("Precipitation contains negative values.")
    if np.any(q[np.isfinite(q)] < 0.0):
        raise ValueError("Discharge contains negative values.")

    qc_names, qc_counts = np.unique(frame["QC"].astype(str), return_counts=True)

    return {
        "X": X,
        "q": q,
        "dates": dates,
        "feature_names": tuple(features),
        "target_names": ("Q",),
        "basin_id": basin_id,
        "area_m2": area_m2,
        "qc_counts": {str(name): int(count) for name, count in zip(qc_names, qc_counts)},
        "missing_discharge_count": int(np.isnan(q).sum()),
    }




def load_camels_dataset(project_root, basin_id, n_du=1, start=None, end=None):
    """Load CAMELS forcing and observed streamflow into a HydroDataset.

    Driver order is precipitation, PET, temperature. Temperature is supplied
    for the snow module and should not be passed directly to a standard CRCNet
    without a documented non-negative transformation.
    """
    from .dataset import HydroDataset
    raw = load_camels(project_root, basin_id, features=("P", "PET", "T"), start=start, end=end)
    x = raw["X"][0]
    water = np.zeros((len(x), int(n_du)), dtype=np.float32)
    water[:, 0] = x[:, 0]
    return HydroDataset(water, x, {"streamflow": raw["q"][0]}, time=raw["dates"])
