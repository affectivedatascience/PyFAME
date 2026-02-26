from pydantic import BaseModel, ValidationError, NonNegativeFloat, field_validator, ValidationInfo
from typing import Union, List, Tuple, Optional
from pyfame.layer.layer import Layer, TimingConfiguration
from pyfame.layer.manipulations.mask.mask_from_landmarks import mask_from_landmarks
from pyfame.landmark.facial_landmarks import *
from pyfame.landmark.get_landmark_coordinates import get_pixel_coordinates_from_landmark
import cv2 as cv
import numpy as np
from operator import itemgetter

class PencilSketchParameters(BaseModel):
    landmark_paths:Optional[Union[List[List[Tuple[int,...]]], List[Tuple[int,...]]]]
    detail_level:NonNegativeFloat
    threshold_bias:NonNegativeFloat

    @field_validator("detail_level")
    @classmethod
    def check_normal_range(cls, val, info:ValidationInfo):
        field_name = info.field_name

        if not (0.0 <= val <= 1.0):
            raise ValueError(f"Parameter {field_name} must be a normalised float in the range [0,1].")
        
        return val

class LayerStylisePencilSketch(Layer):
    def __init__(self, timing_configuration:TimingConfiguration, ps_parameters:PencilSketchParameters):
        self.time_config = timing_configuration
        self.ps_params = ps_parameters

        super().__init__(self.time_config)

        # Define instance parameters
        self.landmark_paths = self.ps_params.landmark_paths
        self.detail_level = self.ps_params.detail_level
        self.illum_scale, self.filter_scale, self.thresh_scale = self.map_detail_to_spatial_scales(d=self.detail_level)
        self.thresh_const = self.ps_params.threshold_bias

        self._snapshot_state()
    
    def supports_weight(self):
        return False
    
    def get_layer_parameters(self) -> dict:
        # Dump the pydantic models to get dict of full parameter list
        self._layer_parameters = self.time_config.model_dump()
        self._layer_parameters.update(self.ps_params.model_dump())
        self._layer_parameters["onset_time_msec"] = self.onset_t
        self._layer_parameters["offset_time_msec"] = self.offset_t
        return dict(self._layer_parameters)
    
    def map_detail_to_spatial_scales(self, d, gamma=1.5) -> Tuple[float,float,float]:
        p = d**gamma

        # Illumination scale
        min_i, max_i = 0.02, 0.15
        scale_i = min_i * (max_i/min_i)**p

        # Bilateral filter scale
        min_b, max_b = 0.01, 0.05
        scale_b = max_b * (min_b/max_b)**p

        # Local contrast (thresholding) scale 
        min_t, max_t = 0.02, 0.1
        scale_t = max_t * (min_t/max_t)**p

        return (scale_i, scale_b, scale_t)
    
    def apply_layer(self, landmarker_coordinates, frame, dt) -> cv.typing.MatLike:
        
        if dt is None:
            weight = 1.0
        else:
            weight = super().compute_weight(dt, self.supports_weight())

        if weight == 0.0:
            return frame
        
        # Compute facial width for filter scaling downstream
        fo_coords = get_pixel_coordinates_from_landmark(landmarker_coordinates, LANDMARK_FACE_OVAL)

        min_x = min(fo_coords, key=itemgetter(0))[0]
        max_x = max(fo_coords, key=itemgetter(0))[0]

        face_width = max_x - min_x

        # Convert frame to greyscale (2D)
        frame_grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Perform a low-pass filter to estimate illumination gradients
        illum = cv.GaussianBlur(frame_grey, (0,0), sigmaX=face_width * self.illum_scale)
        norm = cv.divide(frame_grey, illum, scale=255)

        # Blur the image to smoothen, using bilateral filter to preserve facial landmark edges
        # --- Scale the filter diameter by facial width ---
        filter_d = int(face_width * self.filter_scale)
        filter_d = max(3, filter_d)     # Set a hard minimum to smoothing still occurs in small faces
        frame_grey_smoothed = cv.bilateralFilter(norm, filter_d, 50, filter_d)

        # Threshold the image to extract the edges
        # --- Scale the block size by facial width ---
        k_size = int(face_width * self.thresh_scale)     # Should be slightly larger than filtering diameter
        k_size = max(3, k_size)     # Set a hard minimum
        k_size |= 1     # Bit flip to odd number
        frame_contours = cv.adaptiveThreshold(frame_grey_smoothed, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, k_size, self.thresh_const)
        frame_contours = cv.cvtColor(frame_contours, cv.COLOR_GRAY2BGR)

        # Get landmark mask 
        if self.landmark_paths:
            mask = mask_from_landmarks(frame, self.landmark_paths, landmarker_coordinates)
            output_frame = np.where(mask[:,:,np.newaxis]  == 255, frame_contours, frame)
        else:
            output_frame = frame_contours

        return output_frame
    
def layer_stylise_pencil_sketch(timing_configuration:TimingConfiguration | None = None, landmark_paths:list[list[tuple[int,...]]] | list[tuple[int,...]] = None, 
                                detail_level:float = 0.35, threshold_bias:float = 7.0) -> LayerStylisePencilSketch:
    # Populate with defaults if not passed
    time_config = timing_configuration or TimingConfiguration()

    try:
        params = PencilSketchParameters(
            landmark_paths=landmark_paths,
            detail_level=detail_level,
            threshold_bias=threshold_bias
        )
    except ValidationError as e:
        raise ValueError(f"Invalid parameters for {LayerStylisePencilSketch.__name__}: {e}")
    
    return LayerStylisePencilSketch(time_config, params)