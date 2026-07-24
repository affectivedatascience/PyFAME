# Layer/manipulations init file
# Explicit submodule imports for docsite
import pyfame.layer.manipulations.colour as colour
import pyfame.layer.manipulations.mask as mask
import pyfame.layer.manipulations.occlusion as occlusion
import pyfame.layer.manipulations.overlay as overlay
import pyfame.layer.manipulations.spatial as spatial
import pyfame.layer.manipulations.stylise as stylise
import pyfame.layer.manipulations.temporal as temporal

# Wildcard re-exports for runtime API convenience
from .colour import *
from .mask import *
from .occlusion import *
from .overlay import *
from .spatial import *
from .stylise import *
from .temporal import *

_submodules = ["colour", "mask", "occlusion", "overlay", "spatial", "stylise", "temporal"]
_primary_api = [
    "layer_colour_recolour", 
    "layer_colour_brightness", 
    "layer_colour_saturation",
    "layer_color_recolor",
    "layer_color_brightness",
    "layer_color_saturation",
    "layer_mask", 
    "mask_from_landmarks",
    "layer_occlusion_landmark", 
    "layer_occlusion_bar", 
    "layer_occlusion_blur", 
    "layer_occlusion_noise",
    "layer_overlay",
    "layer_spatial_grid_shuffle", 
    "layer_spatial_landmark_relocate", 
    "FaceAnchor", 
    "LandmarkRelocateSpec",
    "layer_stylise_point_light",
    "layer_stylise_pencil_sketch",
    "generate_shuffled_block_array", 
    "apply_temporal_shuffle"
]

__all__ = _submodules + _primary_api