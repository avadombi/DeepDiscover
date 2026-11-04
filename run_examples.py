"""Execute the example notebooks from the project root and save their outputs.

The three synthetic examples run by default. The CAMELS-US example is optional
because the original data files must be supplied separately.

Examples
--------
python run_examples.py
python run_examples.py --only 01_streamflow
python run_examples.py --include-camels
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXAMPLES = ROOT / "examples"
DEFAULT_NOTEBOOKS = (
    "01_streamflow",
    "02_groundwater_level",
    "03_sources_losses_routing",
)
CAMELS_NOTEBOOK = "04_camels_us"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", choices=(*DEFAULT_NOTEBOOKS, CAMELS_NOTEBOOK),
        help="Run only the selected notebook.",
    )
    parser.add_argument(
        "--include-camels", action="store_true",
        help="Also run the CAMELS-US notebook (original CAMELS-US files required).",
    )
    parser.add_argument(
        "--camels-mode", choices=("check", "benchmark", "full"), default="check",
        help="CAMELS example: short check, timing of full windows, or full calibration (default: check).",
    )
    parser.add_argument(
        "--timeout", type=int, default=3600,
        help="Maximum seconds per notebook cell (default: 3600).",
    )
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive.")

    try:
        import nbformat
        from nbclient import NotebookClient

        class ConsoleNotebookClient(NotebookClient):
            """Display notebook text as it is produced by the Python kernel."""

            def process_message(self, msg, cell, cell_index):
                if msg.get("msg_type") == "stream":
                    message = msg.get("content", {}).get("text", "")
                    if message:
                        print(message, end="" if message.endswith("\n") else "\n", flush=True)
                return super().process_message(msg, cell, cell_index)
    except ImportError as exc:
        parser.error(
            "Notebook execution requires the example dependencies. "
            "Install with: python -m pip install -e '.[examples]'"
        )

    notebooks = [args.only] if args.only else list(DEFAULT_NOTEBOOKS)
    if args.include_camels and CAMELS_NOTEBOOK not in notebooks:
        notebooks.append(CAMELS_NOTEBOOK)

    if CAMELS_NOTEBOOK in notebooks:
        basin = "01013500"
        data = ROOT / "data" / "camels" / basin
        required = [
            data / f"{basin}_lump_cida_forcing_leap.txt",
            data / f"{basin}_streamflow_qc.txt",
        ]
        missing = [p for p in required if not p.is_file()]
        if missing:
            print("CAMELS-US input files are missing:", file=sys.stderr)
            for path in missing:
                print(f"  {path}", file=sys.stderr)
            print("See docs/camels.md for the required folder layout.", file=sys.stderr)
            return 2

    target = ROOT / "results" / "notebooks"
    target.mkdir(parents=True, exist_ok=True)
    report = {
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "notebooks": [],
    }
    has_error = False

    for name in notebooks:
        source = EXAMPLES / f"{name}.ipynb"
        notebook = nbformat.read(source, as_version=4)
        start = time.perf_counter()
        result = {"name": name, "status": "passed", "seconds": None}
        print(f"Running {name} ...", flush=True)
        previous_mode = os.environ.get("DEEPDISCOVER_CAMELS_MODE")
        if name == CAMELS_NOTEBOOK:
            os.environ["DEEPDISCOVER_CAMELS_MODE"] = args.camels_mode
            print(f"  CAMELS mode: {args.camels_mode}", flush=True)

        def announce_cell(cell, cell_index):
            if cell.cell_type == "code":
                preview = next((line.strip() for line in cell.source.splitlines()
                                if line.strip() and not line.lstrip().startswith("#")), "Python code")
                print(f"  Cell {cell_index + 1}/{len(notebook.cells)}: {preview[:85]}", flush=True)

        try:
            client = ConsoleNotebookClient(
                notebook,
                timeout=args.timeout,
                kernel_name="python3",
                allow_errors=False,
                on_cell_execute=announce_cell,
                resources={"metadata": {"path": str(ROOT)}},
            )
            client.execute()
        except Exception as exc:
            result["status"] = "failed"
            result["error"] = f"{type(exc).__name__}: {exc}"
            has_error = True
        finally:
            if name == CAMELS_NOTEBOOK:
                if previous_mode is None:
                    os.environ.pop("DEEPDISCOVER_CAMELS_MODE", None)
                else:
                    os.environ["DEEPDISCOVER_CAMELS_MODE"] = previous_mode
            result["seconds"] = round(time.perf_counter() - start, 2)
            nbformat.write(notebook, target / f"{name}_executed.ipynb")
            report["notebooks"].append(result)
            print(f"  {result['status']} ({result['seconds']} s)", flush=True)
            if "error" in result:
                print(result["error"][-3500:], file=sys.stderr)
            (target / "report.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )

    print(f"Results: {target}")
    return 1 if has_error else 0


if __name__ == "__main__":
    sys.exit(main())
