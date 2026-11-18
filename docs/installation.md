# Installation

DeepDiscover is distributed as `deepdiscover-hydro` through the Python Package Index. The examples below use Windows commands, but the same `pip` package is available on other supported operating systems.

## Python environment

Python **3.11 or 3.12** is recommended. A separate environment keeps TensorFlow and its dependencies independent from other projects.

Using Conda:

```console
conda create -n dd-check python=3.12
conda activate dd-check
python -m pip install deepdiscover-hydro
```

Using an existing virtual environment:

```console
python -m pip install deepdiscover-hydro
```

## Confirm the installation

```python
import deepdiscover
import tensorflow as tf

print('TensorFlow:', tf.__version__)
print('DeepDiscover:', getattr(deepdiscover, '__version__', 'installed'))
```

For precise package metadata, run:

```console
python -m pip show deepdiscover-hydro
```

## Optional scientific tools

The core package installs the dependencies declared in its package metadata. Application notebooks may require additional packages, including `pandas`, `matplotlib`, and `jupyter`. Install the notebook extras with:

```console
python -m pip install "deepdiscover-hydro[examples]"
```

## Source code and notebooks

The public repository includes four example notebooks and a collection of automated tests. To work directly from the source repository, follow its [installation instructions](https://github.com/avadombi/DeepDiscover#installation). The published version and a development checkout should not be confused when reproducing a numerical experiment.

The **documentation preview** uses its own `.venv` folder. Removing the extracted documentation folder deletes this local environment. It does not uninstall DeepDiscover from an unrelated Conda environment.
