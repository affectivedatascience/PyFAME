### TODO

# Colour, Mask, Occlusion, spatial, overlay, layers now work as expected (+ Analysis methods)
# Don't forget to go through stylise and temporal layers, and conversion methods

# Consider looking into passing both the unaltered and current (altered) frame to each layer in LayerPipeline
# If I havent already, weigh pupil size by timestamp in pupil overlay.
# Also adding a greyscale conversion colouring layer 

# Look for more overlay objects, consider creating mappings for tracking midpoints of overlays
# Talk to stephen about timing function wrapper, would allow for asymetric timing curves

import pyfame as pf

file_paths = pf.make_paths()
timing_config = pf.TimingConfiguration(onset_time=250, rise_time=750, rise_curve=pf.timing_gaussian, fall_time=750, fall_curve=pf.timing_sigmoid)

col = pf.layer_color_recolor(timing_config, pf.LANDMARK_FACE_OVAL, magnitude=20.0)

pf.apply_layers(file_paths.iloc[[0]], col)