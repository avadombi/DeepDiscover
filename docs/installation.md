# Installation

DeepDiscover requires Python, TensorFlow and Keras. Python 3.11 or 3.12 is recommended.

Create a virtual environment and install DeepDiscover from PyPI.

Windows:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install deepdiscover-hydro
```

Linux and macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install deepdiscover-hydro
```

To run the notebooks, build the documentation, or execute the tests, clone the repository and install the optional dependencies from its root directory:

```bash
python -m pip install -e ".[examples,docs,test]"
python -m pytest -q
```

TensorFlow support depends on the Python version and operating system. Consult the [TensorFlow installation guide](https://www.tensorflow.org/install/pip) if installation fails.

The PyPI distribution is named `deepdiscover-hydro`, while the Python import is `deepdiscover`.
