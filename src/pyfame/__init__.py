# Package root init file
__version__ = "1.0.5"
__author__ = "Gavin Bosman"

# Direct submodule import
import pyfame.analyse as analyse
import pyfame.file_access as file_access
import pyfame.landmark as landmark
import pyfame.layer as layer
import pyfame.utils as utils
import pyfame.logging as logging

# Re-export to allow import pyfame; pyfame.someClass usage
# Primary API
from pyfame.analyse import *
from pyfame.landmark import *
from pyfame.layer import *
from pyfame.file_access import make_paths

from pyfame.analyse import __all__ as _analyse_all
from pyfame.landmark import __all__ as _landmark_all
from pyfame.layer import __all__ as _layer_all

__all__ = (
    ["analyse", "file_access", "landmark", "layer", "utils", "logging", "make_paths"]
    + _analyse_all + _landmark_all + _layer_all
)