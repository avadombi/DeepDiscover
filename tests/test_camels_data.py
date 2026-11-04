"""Checks for the CAMELS-US text reader and missing discharge values."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


DATA_MODULE = Path(__file__).resolve().parents[1] / "src" / "deepdiscover" / "data.py"
spec = importlib.util.spec_from_file_location("camels_reader_test", DATA_MODULE)
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


def _write_camels_files(root: Path) -> None:
    basin = root / "data" / "camels" / "01013500"
    basin.mkdir(parents=True)
    (basin / "01013500_lump_cida_forcing_leap.txt").write_text(
        "46.84\n353.00\n2260093113\n"
        "Year Mnth Day Hr dayl(s) prcp(mm/day) srad(W/m2) swe(mm) tmax(C) tmin(C) vp(Pa)\n"
        "2014 09 30 12 40000 3.5 150 0 15 5 300\n"
        "2014 10 01 12 40000 1.2 150 0 12 2 300\n"
        "2014 10 02 12 40000 0.0 150 0 10 0 300\n",
        encoding="utf-8",
    )
    (basin / "01013500_streamflow_qc.txt").write_text(
        "01013500 2014 09 30 655 A\n"
        "01013500 2014 10 01 -999 M\n"
        "01013500 2014 10 02 640 A:e\n",
        encoding="utf-8",
    )


def test_hamon_pet_is_finite_for_normal_temperatures():
    pet = reader.hamon_pet(np.array([36000., 43200., 50000.]),
                           np.array([0., 10., 20.]))
    assert np.isfinite(pet).all()
    assert (pet > 0).all()


def test_missing_streamflow_is_maskable(tmp_path):
    _write_camels_files(tmp_path)
    raw = reader.load_camels(tmp_path, "01013500")
    assert raw["X"].shape == (1, 3, 3)
    assert raw["q"].shape == (1, 3, 1)
    assert raw["missing_discharge_count"] == 1
    assert np.isnan(raw["q"][0, 1, 0])
    assert raw["qc_counts"] == {"A": 1, "A:e": 1, "M": 1}
    assert np.all(np.isfinite(raw["q"][0, [0, 2], 0]))
    subset = reader.load_camels(tmp_path, "01013500", start="2014-09-30", end="2014-09-30")
    assert subset["q"].shape == (1, 1, 1)
    assert subset["missing_discharge_count"] == 0


def test_unexpected_negative_flow_is_rejected(tmp_path):
    _write_camels_files(tmp_path)
    p = tmp_path / "data" / "camels" / "01013500" / "01013500_streamflow_qc.txt"
    p.write_text(p.read_text().replace("655 A", "-50 A"), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected negative"):
        reader.load_camels(tmp_path, "01013500")
