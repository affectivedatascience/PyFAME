### TODO
# Update temporal shuffle and img_to_video functions to work with file_paths dataframe
# If I havent already, weigh pupil size by timestamp in pupil overlay.
# Update pytest suite for basic i/o and error raise checks
# MAYBE: Look for more overlay objects, consider creating mappings for tracking midpoints of overlays
# Split create_paths and get_paths into two distinct functionalities

# Expand analysis colour channel means to take a list of landmarks to sample
# Rename TEMPORAL_SHUFFLE constants to FRAME_SHUFFLE_...


import pyfame as pf

file_paths = pf.make_paths()
pf.apply_frame_shuffle(file_paths.iloc[[1]], pf.TEMPORAL_SHUFFLE_PALINDROME)