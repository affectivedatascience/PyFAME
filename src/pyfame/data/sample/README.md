# PyFAME Sample Data

This directory contains sample video files distributed with PyFAME for use in package examples, documentation, testing, and demonstrations.

## RAVDESS Sample Videos

The included video files are unmodified samples from the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS):

Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. PLOS ONE, 13(5), e0196391.

RAVDESS is available from Zenodo under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License.

## Licensing

The RAVDESS sample videos contained in this directory are not covered by the PyFAME software license. They remain subject to the terms of the RAVDESS CC BY-NC-SA 4.0 license.

The sample videos are redistributed in their original, unmodified form solely to provide readily available input data for demonstrating and testing PyFAME.

Users who redistribute, modify, or otherwise use these sample videos are responsible for complying with the terms of the RAVDESS license, including its attribution, non-commercial, and share-alike requirements.

For the complete license terms, see the accompanying LICENSE-RAVDESS file or the Creative Commons CC BY-NC-SA 4.0 license.

## Using the Sample Data

PyFAME provides load_sample_data() to copy the packaged sample videos into the standard PyFAME data directory:

from pyfame.file_access import make_paths, load_sample_data paths = make_paths() sample_paths = load_sample_data(paths)

The sample videos are copied to data/raw/samples, and load_sample_data() returns the paths to the copied files for use with PyFAME.