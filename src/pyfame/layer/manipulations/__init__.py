# Layer/manipulations init file
# Explicit submodule import for docsite
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

__all__ = [
    "colour", 
    "mask", 
    "occlusion", 
    "overlay", 
    "spatial", 
    "stylise", 
    "temporal"
]
# list(colour.__all__) + list(mask.__all__) + list(occlusion.__all__) + list(overlay.__all__) + list(temporal.__all__) + list(spatial.__all__) + list(stylise.__all__)