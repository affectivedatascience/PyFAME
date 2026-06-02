# Package root init file
__version__ = "1.0.0"
__author__ = "Gavin Bosman"

# Direct submodule import
import pyfame.analyse as analyse
import pyfame.file_access as file_access
import pyfame.landmark as landmark
import pyfame.layer as layer
import pyfame.utils as utils
import pyfame.logging as logging

# Re-export to allow import pyfame; pyfame.someClass usage
from pyfame.analyse import *
from pyfame.file_access import *
from pyfame.landmark import *
from pyfame.layer import *
from pyfame.utils import *
from pyfame.logging import *

__all__ = [
    "analyse", 
    "file_access", 
    "landmark", 
    "layer", 
    "utils", 
    "logging"
]
#list(analyse.__all__) + list(file_access.__all__) + list(landmark.__all__) + list(layer.__all__) + list(utils.__all__) + list(logging.__all__)