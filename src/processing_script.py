### TODO
# Convert all ValueError calls in pydantic models to ValidationError
# Update pytest suite for basic i/o and error raise checks

# Update log file ingest to take in an optional list of timing configurations, 
# otherwise output same layers with default timing configuration

#####################################################################
# Go through each file and add __all__, use claude to generate filters to 
# exclude exceptions and such on a per module basis (cur at occlusion)

# Update the home page
####################################################################

import pyfame as pf

paths = pf.make_paths()
tc = pf.TimingConfiguration()
col = pf.manipulations.layer_colour_recolour(tc, pf.LANDMARK_CHEEKS_AND_NOSE, focus_colour="blue", magnitude=15.0)

pf.apply_layers(paths.iloc[[0]], col)