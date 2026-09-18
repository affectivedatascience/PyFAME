# Examples
 
Step-by-step walkthroughs of some of PyFAME's more complex use cases.

## Example 1: Simple timing configuration

The following example highlights how to define a `TimingConfiguration` object, and how to use it alongside manipulation `Layers`. `TimingConfiguration` serves as the base of all temporal controls in PyFAME, and it allows the user to define when a manipulation onsets and offsets, its rise and fall curves, as well as its rise and fall durations.

```python
import pyfame as pf

# Get your file paths object
file_paths = pf.make_paths()

# Define a TimingConfiguration Object
config = pf.TimingConfiguration(
    rise_curve=pf.timing_linear,    # Linear shaped onset curve
    rise_time_msec=1000             # The duration of the off->on transition
)

# Define your layer and pass in time configuration
brighten = pf.layer_colour_brightness(
    timing_configuration=config,
    magnitude=12.0
)

# Apply your layer to your files
pf.apply_layer(
    file_paths = file_paths,
    layers = brighten
)
```

---

## Example 2: Stacking multiple layers

The `Layer` class and its children got their names based on their common ability to be layered or stacked together. The following is a simple example highlighting how to apply multiple layers to the same set of files, within a single `apply_layers()` call.

```python
import pyfame as pf
from pyfame import facial_landmarks as lms

# Get your file paths object
file_paths = pf.make_paths()

# Define a TimingConfiguration object
config = pf.TimingConfiguration(
    rise_curve=pf.timing_linear,
    rise_time_msec=750,  
)

# Define your Layers
mask = layer_mask(
    timing_configuration=config,
    landmark_paths=lms.LANDMARK_FACE_OVAL
)

saturate = layer_colour_saturation(
    timing_configuration=config,
    landmark_paths=lms.LANDMARK_BOTH_CHEEKS,
    magnitude=15.0
)

# Combining specific layers may cause unintended behaviour,
# play around with different combinations to see what works best!
pf.apply_layers(
    file_paths = file_paths,
    layers = [mask, saturate]   # Pass multiple layers as a list
)
```

---

## Example 3: Multiple layers with multiple timing configurations

```python
import pyfame as pf
from pyfame import facial_landmarks as lms

# Get your file paths object
file_paths = pf.make_paths()

# Define your timing configurations
config_colour = pf.TimingConfiguration(
    rise_curve=pf.timing_sigmoid, 
    rise_time_msec=750,               
)

config_sat = pf.TimingConfiguration(
    fall_curve=pf.timing_gaussian,
    fall_time_msec=1000,          
)

# Define your Layers, each with their own timing configuration
colour = layer_colour_recolour(
    timing_configuration=config_colour,
    landmark_paths=lms.LANDMARK_FACE_SKIN,
    focus_colour="green",
    magnitude=15.0
)

desaturate = layer_colour_saturation(
    timing_configuration=config_sat,
    landmark_paths=lms.LANDMARK_FACE_SKIN,
    magnitude=-12.0
)

# Apply your layers to your file paths
pf.apply_layers(
    file_paths = file_paths,
    layers = [desaturate, colour]
)
```

---

## Advanced Example 1: File path filtering

