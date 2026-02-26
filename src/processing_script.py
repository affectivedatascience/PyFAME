### TODO
# If I havent already, add the option to weigh pupil size by timestamp in pupil overlay.
# MAYBE: Look for more overlay objects, consider creating mappings for tracking midpoints of overlays

# Rename TEMPORAL_SHUFFLE constants to FRAME_SHUFFLE_...
# Update pytest suite for basic i/o and error raise checks
# Continue to update docstrings for autodoc generation (next is layer_..._saturation)
# Simplify mask from landmark functionality to better leverage get_pixel_coordinates 


import pyfame as pf
file_paths = pf.make_paths()
pencil = pf.layer_stylise_pencil_sketch(detail_level=0.0)
pf.apply_layers(file_paths.iloc[[1]], pencil)
