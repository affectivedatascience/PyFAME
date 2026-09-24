# Getting Started

## Installing PyFAME

PyFAME requires Python >= 3.9 to be installed on your system. You can find information on installing and setting up Python [here](https://www.python.org/downloads/). Once you have Python installed, PyFAME can be installed with pip via PyPi:

```sh
pip install pyfame
```

## Quick Start

PyFAME provides a complete image and video manipulation pipeline, acheivable with only a few lines of code:
```python
import pyfame as pf

# Get dataframe of file paths, pre-loaded with sample data
files = pf.make_paths(load_sample_data = True)

# Create your manipulation layer
layer_recolour = pf.layer_colour_recolour()

# Apply your layer to your file paths
pf.apply_layers(file_paths = files, layers = layer_recolour)
```

---

## General Workflow

The first step in most PyFAME scripts is to call the `make_paths()` function, which will return all of the file paths within the projects data folder (named 'data/' by default; but can be user-specified). Additionally, `make_paths()` will create a subdirectory structure internal to the project folder; namely the `raw/` and `processed/` subdirectories will contain all of the pre, and post-manipulation files respectively. To automatically populate the `raw/` subdirectory with several sample videos, set `load_sample_data` = `True`.

After retrieving the file paths, the next step is to define your manipulation layers. An optional TimingConfiguration object can be defined and passed to each layer for precise temporal control, but by default all layers will perform their manipulation at 100% strength for the duration of the video.

Lastly, to apply your manipulation layers to your files, pass both the file_paths object and one or more layers to the `apply_layers()` function. Additionally, users may pass the file_paths directly to an analysis function (i.e. `analyse_optical_flow()`).

!!! info
    If you would like to see more specific examples or tutorials, check out the [Examples](../examples/index.md)