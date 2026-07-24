from pydantic import BaseModel, field_validator, ValidationError, ValidationInfo, PositiveInt
from typing import Union, List, Tuple
from pyfame.landmark.facial_landmarks import *
from pyfame.utils.constants import *
from pyfame.layer._layer import Layer, TimingConfiguration
from pyfame.layer.manipulations.mask import mask_from_landmarks
import cv2 as cv
import numpy as np

class BlurringParameters(BaseModel):
    """
    Configuration model defining the control parameters for applying
    a blur occlusion to a frame or image.

    This class inherits from pydantic's `BaseModel` to provide validation
    and default handling of blurring parameters.

    Attributes
    ----------
    blur_method : str or int
        The blurring algorithm to apply. Accepted string values are
        ``"average"``, ``"gaussian"``, and ``"median"``. Accepted integer
        values are ``11`` (average), ``12`` (gaussian), and ``13`` (median).
        Integer inputs are normalized to their string equivalents on
        validation.
    max_kernel_size : int
        The maximum blur kernel size, reached at full temporal weight.
        Must be a positive odd integer in the range [3, 31].
    landmark_paths : list of list of tuple of int or list of tuple of int
        A list of one or more closed landmark paths representing the
        region in which the blur will be applied.
    """

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
    """
    Manipulation layer that applies a spatially localised blur within
    landmark-defined regions over time.

    This layer occludes a facial region of interest by blurring it using
    one of three supported kernel-based methods: average, Gaussian, or
    median. The effective blur intensity is modulated by the temporal
    weight computed from the timing configuration, with kernel size
    scaling linearly from 3 up to ``max_kernel_size`` as weight increases.

    Parameters
    ----------
    timing_configuration : TimingConfiguration
        Timing configuration controlling onset, offset, rise/fall durations,
        and temporal weighting behavior.
    blurring_parameters : BlurringParameters
        Configuration model specifying the blur method, maximum kernel size,
        and landmark region(s).

    Attributes
    ----------
    time_config : TimingConfiguration
        Timing configuration used by the layer.
    blur_params : BlurringParameters
        Blur-specific configuration parameters.
    blur_method : str
        The blurring algorithm to apply (one of ``"average"``, ``"gaussian"``,
        or ``"median"``).
    max_kernel_size : int
        The maximum blur kernel size reached at full temporal weight.
    landmark_paths : list of list of tuple of int or list of tuple of int
        Landmark paths defining the region(s) in which blur is applied.

    Notes
    -----
    - Blur intensity is coupled to temporal weight via ``temporal_weight_to_kernel``,
      which maps the continuous weight value to a valid odd kernel size.
    - Median blur kernel scaling is intentionally more conservative than
      average and Gaussian, using a quadratic curve capped at
      ``max(3, (max_kernel_size // 2) | 1)`` to avoid overly aggressive
      occlusion at moderate weights.
    - This layer supports continuous temporal weighting.
    """

    def __init__(self, timing_configuration:TimingConfiguration, blurring_parameters:BlurringParameters):
        """
        Initialize a blur occlusion manipulation layer.

        Parameters
        ----------
        timing_configuration : TimingConfiguration
            Timing configuration controlling when the blur effect is applied
            and how its intensity transitions over time.
        blurring_parameters : BlurringParameters
            Parameters defining the blur method, maximum kernel size, and
            target landmark region(s).

        Notes
        -----
        - The timing configuration is passed to the superclass ``Layer``.
        - A snapshot of the initial state is taken after initialization to
          allow safe resetting between independent applications.
        """

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
        """
        Indicate whether the layer supports temporal weighting.

        Returns
        -------
        bool
            ``True``, as blur occlusion supports continuous intensity
            modulation through kernel size scaling during rise/fall
            transitions.
        """
        return True
    
    def get_layer_parameters(self) -> dict:
        """
        Return the parameters defining this layer.

        This method exposes all configurable parameters required to reproduce
        the layer's behavior.

        Returns
        -------
        dict
            Dictionary mapping parameter names to their current values,
            combining both timing and blurring configuration fields.
        """
        # Dump the pydantic models to get dict of full parameter list
        self._layer_parameters = self.time_config.model_dump()
        self._layer_parameters.update(self.blur_params.model_dump())
        self._layer_parameters["onset_time_msec"] = self.onset_t
        self._layer_parameters["offset_time_msec"] = self.offset_t
        return dict(self._layer_parameters)
    
    def temporal_weight_to_kernel(self, weight:float) -> int:
        """
        Map a continuous temporal weight value to a valid odd kernel size.

        For average and Gaussian blur, kernel size scales linearly from 3
        to ``max_kernel_size`` as weight increases from 0.0 to 1.0. For
        median blur, a quadratic curve is used with the effective maximum
        capped at ``max(3, (max_kernel_size // 2) | 1)`` to prevent
        perceptually excessive occlusion at moderate weights.

        The resulting kernel size is always rounded to the nearest odd
        integer, as required by OpenCV's blurring functions.

        Parameters
        ----------
        weight : float
            The current temporal weight in the range [0.0, 1.0], as
            computed from the timing configuration.

        Returns
        -------
        int
            A valid odd kernel size in the range [3, ``max_kernel_size``],
            scaled according to the current weight and blur method.
        """
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
        """
        Apply the blur occlusion manipulation to a single frame.

        The landmark-defined region of interest is isolated using a binary
        mask. The full frame is blurred using the configured method and
        kernel size derived from the current temporal weight, then the
        blurred result is composited back over the original frame within
        the masked region.

        Parameters
        ----------
        landmarker_coordinates : list of tuple of int
            Facial landmark coordinates associated with the current frame.
        frame : MatLike
            Input image frame to which the blur occlusion is applied.
        dt : float, optional
            Current time (in milliseconds). If ``None``, a weight of 1.0
            is used, applying the maximum blur intensity.

        Returns
        -------
        MatLike
            The frame with blur applied within the landmark-defined region,
            and original pixel values preserved outside it.
        """
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

def layer_occlusion_blur(timing_configuration:TimingConfiguration | None = None, blur_method:str|int = "gaussian", landmark_paths:list[list[tuple[int,...]]] | list[tuple[int,...]] = LANDMARK_FACE_OVAL, 
                         max_kernel_size:int = 29) -> LayerOcclusionBlur:
    """
    Factory function for the blur occlusion manipulation layer.
    `LayerOcclusionBlur` applies a spatially localised blur within one or
    more landmark-defined facial regions. Blur intensity is modulated over
    time by scaling the kernel size in proportion to the temporal weight
    derived from the timing configuration, enabling smooth onset and offset
    transitions.

    Parameters
    ----------
    timing_configuration : TimingConfiguration or None, optional
        A pydantic model containing timing configurations controlling onset,
        offset, rise/fall durations, and weighting curves. If ``None``, a
        default ``TimingConfiguration`` is instantiated. The default
        instantiation assumes a linear rise and fall transition, onset at
        0.0 and offset at the video's duration.
    blur_method : str or int, default="gaussian"
        The blurring algorithm to apply. Accepted string values are
        ``"average"``, ``"gaussian"``, and ``"median"``. Accepted integer
        values are ``11`` (average), ``12`` (gaussian), and ``13`` (median).
    landmark_paths : list of list of tuple of int or list of tuple of int, default=LANDMARK_FACE_OVAL
        A list of one or more closed landmark paths representing the
        region in which the blur will be applied.
    max_kernel_size : int, default=28
        The maximum blur kernel size, reached at full temporal weight.
        Must be a positive odd integer in the range [3, 31].

    Returns
    -------
    LayerOcclusionBlur
        An instance of the blur occlusion manipulation layer.

    Raises
    ------
    ValueError
        When provided invalid, out-of-range, or unrecognized parameter values.
    """
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

__all__ = ["layer_occlusion_blur", "BlurringParameters"]