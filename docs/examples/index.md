# Examples
 
Step-by-step walkthroughs of some of PyFAME's more complex use cases.

## Example 1: Simple timing configuration

The following example highlights how to define a `TimingConfiguration` object, and how to use it alongside manipulation `Layers`. `TimingConfiguration` serves as the base of all temporal controls in PyFAME, and it allows the user to define when a manipulation onsets and offsets, its rise and fall curves, as well as its rise and fall durations.

```python
import pyfame as pf

# Get your file paths object
file_paths = pf.load_sample_data()

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
file_paths = pf.load_sample_data()

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
file_paths = pf.load_sample_data()

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

## Advanced Example 1: Custom landmark region

Though PyFAME comes prepackaged with a variety of defined landmark paths, users may want to define their own custom paths. That is where the tastefully named `create_landmark_path()` comes in handy.

```python
import cv2
import pyfame as pf
from pyfame.utils import display_landmarks_face_overlay

# Load in file paths
file_paths = pf.load_sample_data()

# Optionally, visualize the MediaPipe facial landmarks first

# Read in image
my_img_path = "C:/my/path/to/img.png"
img = cv2.imread(my_img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Get FaceLandmarker and landmark coordinates
faceLM = pf.get_face_landmarker()
coords = pf.get_landmarker_coordinates(
    frame_rgb = img_rgb, 
    face_landmarker = faceLM, 
    static_image_mode = True
)
# Display the coords over the face in the provided image
display_landmarks_face_overlay(
    frame = img, 
    landmarker_coordinates = coords
)

# Define a set of landmark points, in order to ensure the resulting path 
# is circular you must duplicate the first landmark to the end of the list
landmark_points = [112, 109, 104, 103, 100, 78, 82, 207, 69, 110, 112]

# Create a custom path
custom_path = pf.create_landmark_path(
    landmark_set = landmark_points
)

# Define a layer with default constant timing
mask = pf.layer_mask(
    landmark_paths = custom_path
)

# Apply the mask manipulation to your file paths
pf.apply_layer(
    file_paths = file_paths,
    layers = mask
)

```

---

## Advanced Example 2: Landmark relocation with custom LandmarkRelocateSpec

```python
import pyfame as pf

# load in file paths
file_paths = pf.load_sample_data()

# The layer_spatial_landmark_relocate() takes in a parameter
# landmark_relocate_specs, which is a dict of (int, FaceAnchor) pairs

# Keys range [0-3] corresponding to 0-left eye, 1-right eye,
# 2-nose, and 3-mouth. Each index is paired with a LandmarkRelocateSpec
lm_reloc_specs = {
    # Each spec must define a FaceAnchor, and optionally may define 
    # a rotation and displacement
    0 : pf.LandmarkRelocateSpec(                
        anchor = pf.FaceAnchor.SUBJECT_LOWER_CENTER,  
        rotation_deg = 35.0                    
    ),
    1 : pf.LandmarkRelocateSpec(
        anchor = pf.FaceAnchor.SUBJECT_UPPER_LEFT,
        rotation_deg = -45.0,
        offsets = (15, 9)
    ),
    2 : pf.LandmarkRelocateSpec(
        anchor = pf.FaceAnchor.SUBJECT_UPPER_RIGHT
    ),
    3 : pf.LandmarkRelocateSpec(
        anchor = pf.FaceAnchor.SUBJECT_CENTER
        rotation_deg = 90.0
    )
}

# Define the layer with above relocation specs
lm_relocate = pf.layer_spatial_landmark_relocate(
    landmark_relocate_specs = lm_reloc_specs,
    out_greyscale = True
)

# Apply the layer to our files
pf.apply_layer(
    file_paths = file_paths,
    layers = lm_relocate
)

```
