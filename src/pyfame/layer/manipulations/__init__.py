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
_modules = [colour, mask, occlusion, overlay, spatial, stylise, temporal]

_flattened = []
for _m in _modules:
    _flattened.extend(getattr(_m, "__all__", []))

# sanity check for silent collisions between manipulation categories
_dupes = {n for n in _flattened if _flattened.count(n) > 1}
if _dupes:
    raise ImportError(f"Name collisions across manipulation submodules: {_dupes}")

__all__ = _submodules + _flattened

# list(colour.__all__) + list(mask.__all__) + list(occlusion.__all__) + list(overlay.__all__) + list(temporal.__all__) + list(spatial.__all__) + list(stylise.__all__)