from pydantic import BaseModel, field_validator, ValidationError, ValidationInfo, PositiveInt
from typing import Union, List, Tuple
from pyfame.landmark.facial_landmarks import *
from pyfame.utilities.constants import *
from pyfame.layer.layer import Layer, TimingConfiguration
from pyfame.layer.manipulations.mask import mask_from_landmarks
import cv2 as cv
import numpy as np

# Investigate weight scaling the blurring kernel's dimensions

class BlurringParameters(BaseModel):
    blur_method:Union[str, int]
    max_kernel_size:PositiveInt
    landmark_paths:Union[List[List[Tuple[int,...]]], List[Tuple[int,...]]]

    @field_validator("blur_method", mode="before")
    @classmethod
    def check_compatible_value(cls, value, info:ValidationInfo):
        field_name = info.field_name
        blur_method_mapping = {11:"average", 12:"gaussian", 13:"median"}

        if isinstance(value, str):
            value = str.lower(value)
            if value not in {"average", "gaussian", "median"}:
                raise ValueError(f"Unrecognized value for parameter {field_name}.")
            return value
        
        elif isinstance(value, int):
            if value not in {11, 12, 13}:
                raise ValueError(f"Unrecognized value for parameter {field_name}.")
            return blur_method_mapping.get(value)
        
        raise TypeError(f"{field_name} provided an invalid type. Must be one of int, str.")

    @field_validator("max_kernel_size")
    @classmethod
    def check_odd_dims(cls, value, info:ValidationInfo):
        field_name = info.field_name

        # Ensures kernel size provided is odd and greater or equal to (3,3)
        if not (value % 2 == 1 and value >= 3 and value <= 31):
            raise ValueError(f"{field_name} must be odd and >= 3.")
        
        return value

class LayerOcclusionBlur(Layer):
    def __init__(self, timing_configuration:TimingConfiguration, blurring_parameters:BlurringParameters):

        self.time_config = timing_configuration
        self.blur_params = blurring_parameters

        # Initialise superclass
        super().__init__(self.time_config)
        
        # Define class parameters
        self.blur_method = self.blur_params.blur_method
        self.max_kernel_size = self.blur_params.max_kernel_size
        self.landmark_paths = self.blur_params.landmark_paths

        # Snapshot of initial state
        self._snapshot_state()
    
    def supports_weight(self):
        return True
    
    def get_layer_parameters(self) -> dict:
        # Dump the pydantic models to get dict of full parameter list
        self._layer_parameters = self.time_config.model_dump()
        self._layer_parameters.update(self.blur_params.model_dump())
        self._layer_parameters["onset_time_msec"] = self.onset_t
        self._layer_parameters["offset_time_msec"] = self.offset_t
        return dict(self._layer_parameters)
    
    def temporal_weight_to_kernel(self, weight:float) -> int:
        k = 0

        if self.blur_method in {"average", "gaussian"}:
            # linear progression
            k = 3 + weight * (self.max_kernel_size - 3)
        elif self.blur_method == "median":
            # Median blur becomes to strong at higher kernel sizes
            # so restrict to k//2
            k_med = max(3, (self.max_kernel_size//2) | 1)
            k = 3 + (weight**2) * (k_med - 3)
        
        k = int(round(k))
        if k % 2 == 0:
            k += 1
        
        return k

    def apply_layer(self, landmarker_coordinates:list[tuple[int,int]], frame:cv.typing.MatLike, dt:float = None):

        # Blurring does not support weight, so weight will always be 0.0 or 1.0
        if dt is None:
            weight = 1.0
        else:
            weight = super().compute_weight(dt, self.supports_weight())

        if weight == 0.0:
            return frame

        # Mask out region of interest
        mask = mask_from_landmarks(frame, self.landmark_paths, landmarker_coordinates)
        mask = mask[:,:,np.newaxis]     #reshape to 3-channel
        output_frame = np.zeros_like(frame, dtype=np.uint8)

        k_size = self.temporal_weight_to_kernel(weight=weight)

        # Blur the input frame depending on user-specified blur method
        match self.blur_method:
            case "average":
                frame_blurred = cv.blur(frame, (k_size, k_size))
                output_frame = np.where(mask == 255, frame_blurred, frame)
            
            case "gaussian":
                frame_blurred = cv.GaussianBlur(frame, (k_size, k_size), 0)
                output_frame = np.where(mask == 255, frame_blurred, frame)
            
            case "median":
                frame_blurred = cv.medianBlur(frame, k_size)
                output_frame = np.where(mask == 255, frame_blurred, frame)
        
        return output_frame

def layer_occlusion_blur(timing_configuration:TimingConfiguration | None = None, blur_method:str|int = "gaussian", landmark_paths:list[list[tuple[int,...]]] | list[tuple[int,...]] = LANDMARK_FACE_OVAL, max_kernel_size:int = 28) -> LayerOcclusionBlur:
    # Populate with defaults if None
    time_config = timing_configuration or TimingConfiguration()

    try:
        params = BlurringParameters(
            blur_method=blur_method, 
            max_kernel_size=max_kernel_size, 
            landmark_paths=landmark_paths
        )
    except ValidationError as e:
        raise ValueError(f"Invalid parameters for {LayerOcclusionBlur.__name__}: {e}")

    return LayerOcclusionBlur(time_config, params) 