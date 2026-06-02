import json
import os
import jsonschema
from jsonschema import ValidationError
from importlib.resources import files
from pyfame.layer.layer import Layer, TimingConfiguration
from pyfame.layer.manipulations.colour import (layer_colour_recolour, layer_colour_brightness, layer_colour_saturation)
from pyfame.layer.manipulations.mask import layer_mask
from pyfame.layer.manipulations.occlusion import (layer_occlusion_bar, layer_occlusion_landmark, layer_occlusion_blur, layer_occlusion_noise)
from pyfame.layer.manipulations.overlay import layer_overlay
from pyfame.layer.manipulations.spatial import (layer_spatial_grid_shuffle, layer_spatial_landmark_relocate)
from pyfame.layer.manipulations.stylise import (layer_stylise_point_light, layer_stylise_pencil_sketch)
import pyfame.layer.timing_curves as t_curves
import pyfame.utils.constants as const
import pyfame.landmark.facial_landmarks as landmark

def read_experiment_log(log_file_path:str) -> list[Layer]:
    """
    Given a log file path, attempt to extract and recreate the manipulation layers
    applied in the original manipulation run.

    Parameters
    ----------
    log_file_path: str
        A path string to the log file containing manipulation info 
        to be replicated.

    Raises
    ------
    ValueError
        When provided an invalid file path, or the path points
        to a non json logfile.
    IsADirectoryError
        When the provided path exists, but points to a dir
        rather than a file. 

    Notes
    -----
    - This function may not perform as expected if the new input video is
    a different length or frame rate than the files used in the original run. 
    This function is intended to speed up the reproduction of manipulations 
    over large (typically standardised datasets). 
    - As the timing configuration is irrelevant to static images, this 
    function will always work as expected for static image manipulation 
    reproduction.
    """
    if not os.path.exists(log_file_path):
        raise ValueError("Invalid log file path provided, log file Cannot be read.")
    if not os.path.isfile(log_file_path):
        raise IsADirectoryError("Log file path exists, but does not point to a file.")
    if not log_file_path.lower().endswith(".json"):
        raise ValueError(f"Expected a .json log file, got: {log_file_path}")
    
    const_dict = vars(const)
    lm_dict = vars(landmark)
    timing_dict = vars(t_curves)

    # Layer mappings
    layer_name_map = {
        "LayerColourBrightness":layer_colour_brightness,
        "LayerColourRecolour":layer_colour_recolour,
        "LayerColourSaturation":layer_colour_saturation,
        "LayerMask":layer_mask,
        "LayerOcclusionBar":layer_occlusion_bar,
        "LayerOcclusionLandmark":layer_occlusion_landmark,
        "LayerOcclusionBlur":layer_occlusion_blur,
        "LayerOcclusionNoise":layer_occlusion_noise,
        "LayerSpatialGridShuffle":layer_spatial_grid_shuffle,
        "LayerSpatialLandmarkRelocate":layer_spatial_landmark_relocate,
        "LayerStylisePointLight":layer_stylise_point_light,
        "LayerStylisePencilSketch":layer_stylise_pencil_sketch,
        "LayerOverlay":layer_overlay
    }
    
    # Read in the json
    with open(log_file_path, "r") as fp:
        experiment_log = json.load(fp)
    
    # Validate input log against schema
    try:
        schema_path = files("pyfame.schema").joinpath("manipulation_log.v1.schema.json")
        schema = json.load(open(schema_path))
        jsonschema.validate(instance=experiment_log, schema=schema)
    except ValidationError as e:
        raise ValueError(f"Experiment log failed to validate: {e.message}.")
    
    layers_return = []
    layers = experiment_log.get("layers")
    layer_names = list(layers.keys())

    for name in layer_names:
        fn = layer_name_map.get(name)
        params = layers.get(name)

        onset = params.pop("onset_time")
        offset = params.pop("offset_time")
        rise = params.pop("rise_time")
        fall = params.pop("fall_time")
        rise_fn = params.pop("rise_curve")
        fall_fn = params.pop("fall_curve")
        rise_fn_kwargs = params.pop("rise_curve_kwargs")
        fall_fn_kwargs = params.pop("fall_curve_kwargs")

        timeconfig = TimingConfiguration(
            onset_time_msec=onset,
            offset_time_msec=offset,
            rise_time_msec=rise,
            fall_time_msec=fall,
            rise_curve=timing_dict.get(rise_fn),
            fall_curve=timing_dict.get(fall_fn),
            rise_curve_kwargs=rise_fn_kwargs,
            fall_curve_kwargs=fall_fn_kwargs
        )

        # Resolve remaining params
        for k,v in params.items():
            if isinstance(v, str):
                if v in const_dict:
                    params.update({k:getattr(const, v)})
                elif v in lm_dict:
                    params.update({k:getattr(landmark, v)})

        layer_instance = fn(timeconfig, **params)
        layers_return.append(layer_instance)
    
    return layers_return

__all__ = ["read_experiment_log"]