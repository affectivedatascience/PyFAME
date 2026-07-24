from pydantic import BaseModel, field_validator, ValidationInfo, ValidationError
from typing import Union, List, Tuple
from pyfame.layer._layer import Layer, TimingConfiguration
from pyfame.layer.manipulations.mask import mask_from_landmarks
from pyfame.landmark.facial_landmarks import LANDMARK_FACE_OVAL
from pyfame.utils.constants import *
import cv2 as cv
import numpy as np

class SaturationParameters(BaseModel):
    """
    Configuration model defining the control parameters for 
    manipulating the colour saturation of a frame or image.

    This class inherits from pydantic's `BaseModel` to provide validation 
    and default handling of saturation parameters.

    Attributes
    ----------
    landmark_paths : list of list of tuple of int or list of tuple of int
        A list of one or more closed landmark paths representing the 
        region in which the manipulation will be applied.
    magnitude : float
        The degree by which to increase or decrease the saturation.
        Must lie in the range [-25.0, 25.0]. Positive values increase
        saturation; negative values decrease it toward greyscale.
    saturation_mode : str
        The method by which the saturation is scaled. `relative` indicates
        the scaled saturation range begins relative to the provided image or 
        video frame, where `absolute` indicates the saturation is scaled to
        the full 0-255 range.
    """

    landmark_paths:Union[List[List[Tuple[int,...]]], List[Tuple[int,...]]]
    magnitude:float
    saturation_mode:str

    @field_validator("magnitude")
    @classmethod
    def check_value_range(cls, value, info:ValidationInfo):
        field_name = info.field_name
        if not (-25.0 <= value <= 25.0):
            raise ValueError(f"{field_name} must lie between -25.0 and 25.0.")
        
        return value

    @field_validator("saturation_mode")
    @classmethod
    def check_accepted_values(cls, value, info:ValidationInfo):
        field_name = info.field_name
        if str.lower(value) not in {"relative", "absolute"}:
            raise ValueError(f"{field_name} must be one of `relative` or `absolute`.")

        return str.lower(value)

class LayerColourSaturation(Layer):
    """
    Manipulation layer that selectively shifts colour saturation within
    landmark-defined regions over time.

    This layer performs targeted saturation adjustment by modifying the
    saturation channel in the HSV colour space. A foreground mask derived
    from Otsu thresholding and flood-filling is used to constrain the
    effect to the subject, preventing saturation shifts from bleeding into
    the image background.

    Parameters
    ----------
    timing_configuration : TimingConfiguration
        Timing configuration controlling onset, offset, rise/fall durations,
        and temporal weighting behavior.
    saturation_parameters : SaturationParameters
        Configuration model specifying landmark regions and saturation magnitude.

    Attributes
    ----------
    time_config : TimingConfiguration
        Timing configuration used by the layer.
    sat_params : SaturationParameters
        Saturation-specific configuration parameters.
    landmark_paths : list of list of tuple of int or list of tuple of int
        Landmark paths defining the region(s) in which saturation adjustment
        is applied.
    magnitude : float
        The degree by which to increase or decrease the saturation.
        Must lie in the range [-25.0, 25.0].

    Notes
    -----
    - Saturation manipulation is performed in HSV space for direct and
      intuitive control over colour vividness.
    - A foreground mask is computed per-frame using Otsu thresholding and
      flood-filling, ensuring that saturation changes are not applied to
      background regions outside the subject.
    - The effective saturation shift at a given time is scaled by the
      temporal weight computed from the timing configuration.
    - This layer supports continuous temporal weighting.
    """

    def __init__(self, timing_configuration:TimingConfiguration, saturation_parameters:SaturationParameters):
        """
        Initialize a saturation manipulation layer.

        Parameters
        ----------
        timing_configuration : TimingConfiguration
            Timing configuration controlling when the saturation effect
            is applied and how it transitions over time.
        saturation_parameters : SaturationParameters
            Parameters defining the target landmark region(s) and saturation
            magnitude.

        Notes
        -----
        - The timing configuration is passed to the superclass ``Layer``.
        - A snapshot of the initial state is taken after initialization to
          allow safe resetting between independent applications.
        """
        self.time_config = timing_configuration
        self.sat_params = saturation_parameters

        # Initialise the superclass
        super().__init__(self.time_config)
        
        # Define class parameters
        self.landmark_paths = self.sat_params.landmark_paths
        self.magnitude = self.sat_params.magnitude
        self.saturation_mode = self.sat_params.saturation_mode
        self.has_saturation_been_sampled = False
        self.relative_shift_amount = 0

        # Snapshot of initial state
        self._snapshot_state()
    
    def supports_weight(self) -> bool:
        """
        Indicate whether the layer supports temporal weighting.

        Returns
        -------
        bool
            ``True`` if the layer supports continuous rise/fall weighting,
            ``False`` if it operates as a binary on/off manipulation.
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
            combining both timing and saturation configuration fields.
        """
        # Dump the pydantic models to get dict of full parameter list
        self._layer_parameters = self.time_config.model_dump()
        self._layer_parameters.update(self.sat_params.model_dump())
        self._layer_parameters["onset_time_msec"] = self.onset_t
        self._layer_parameters["offset_time_msec"] = self.offset_t
        return dict(self._layer_parameters)
    
    def apply_layer(self, landmarker_coordinates:list[tuple[int,int]], frame:cv.typing.MatLike, dt:float):
        """
        Apply the layer's saturation manipulation to a single frame.

        The saturation channel of the HSV-converted frame is shifted by
        the configured magnitude within the landmark-defined region. A
        foreground mask computed from Otsu thresholding and flood-filling
        is used to preserve the original pixel values in background regions
        unrelated to the subject.

        Parameters
        ----------
        landmarker_coordinates : list of tuple of int
            Facial landmark coordinates associated with the current frame.
        frame : MatLike
            Input image frame to which the manipulation is applied.
        dt : float
            Current time (in milliseconds).

        Returns
        -------
        MatLike
            The manipulated frame with saturation adjusted within the
            specified landmark region, with background pixels restored
            to their original values.
        """
        if dt is None:
            weight = 1.0
        else:
            weight = super().compute_weight(dt, self.supports_weight())
        
        # Occurs when the dt < onset_time, or > offset_time
        if weight == 0.0:
            return frame
        
        # Mask out our region of interest
        mask = mask_from_landmarks(frame, self.landmark_paths, landmarker_coordinates)

        # Otsu thresholding to seperate foreground and background
        grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        grey_blurred = cv.GaussianBlur(grey_frame, (7,7), 0)
        thresh_val, thresholded = cv.threshold(grey_blurred, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

        # Adding a temporary image border to allow for correct floodfill behaviour
        bordered_thresholded = cv.copyMakeBorder(thresholded, 10, 10, 10, 10, cv.BORDER_CONSTANT)
        floodfilled = bordered_thresholded.copy()
        cv.floodFill(floodfilled, None, (0,0), 255)

        # Removing temporary border and creating foreground mask
        floodfilled = floodfilled[10:-10, 10:-10]
        floodfilled = cv.bitwise_not(floodfilled)
        foreground = cv.bitwise_or(thresholded, floodfilled)

        # Convert the image into the HSV space so we can manipulate the saturation
        img_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV).astype(np.float32)
        # Split the image channels so only the saturation can be shifted
        h,s,v = cv.split(img_hsv)

        if not self.has_saturation_been_sampled:
            s_mu = round(np.mean(s, where=mask.astype(bool)))
            relative_range = 255-s_mu
            self.relative_shift_amount = round((self.magnitude * relative_range) / 100)
            self.has_saturation_been_sampled = True

        if self.saturation_mode == "relative":
            s = np.where(mask == 255, s + (weight * self.relative_shift_amount), s)
            s = np.clip(s,0,255)
        else:
            shift_amount = (self.magnitude * 255) / 100
            s = np.where(mask == 255, s + (weight * shift_amount), s)
            s = np.clip(s,0,255)
        
        img_hsv = cv.merge([h,s,v])
        
        # Convert the HSV image back to BGR before returning the processed image
        img_bgr = cv.cvtColor(img_hsv.astype(np.uint8), cv.COLOR_HSV2BGR)
        img_bgr[foreground == 0] = frame[foreground == 0]
        return img_bgr
        
def layer_colour_saturation(timing_configuration:TimingConfiguration|None = None, landmark_paths:list[list[tuple[int,...]]] | list[tuple[int,...]] = LANDMARK_FACE_OVAL, 
                            magnitude:float = -12.0, saturation_mode:str = "relative") -> LayerColourSaturation:
    """
    Factory function for the saturation manipulation layer. `LayerColourSaturation`
    leverages the HSV colour space to perform intuitive saturation shifts within a
    specified region of the face. A foreground mask derived from Otsu thresholding
    ensures that saturation changes are constrained to the subject and do not affect
    the image background.

    Parameters
    ----------
    timing_configuration : TimingConfiguration or None, optional
        A pydantic model containing timing configurations controlling onset,
        offset, rise/fall durations, and weighting curves. If ``None``, a
        default ``TimingConfiguration`` is instantiated. The default
        instantiation assumes a constant rise and fall transition, onset at
        0.0 and offset at the video's duration.
    landmark_paths : list of list of tuple of int or list of tuple of int, default=LANDMARK_FACE_OVAL
        A list of one or more closed landmark paths representing the
        region in which the manipulation will be applied.
    magnitude : float, default=-12.0
        The degree by which to increase or decrease the saturation.
        Must lie in the range [-25.0, 25.0]. Positive values increase
        colour vividness; negative values shift the region toward greyscale.
    saturation_mode : str
            The method by which the saturation is scaled. `relative` indicates
            the scaled saturation range begins relative to the provided image or 
            video frame, where `absolute` indicates the saturation is scaled to
            the full 0-255 range.

    Returns
    -------
    LayerColourSaturation
        An instance of the saturation manipulation layer.

    Raises
    ------
    ValueError
        When provided invalid or out-of-range parameter values.
    """
    # Populate with defaults if None
    time_config = timing_configuration or TimingConfiguration()

    # Validate input parameters
    try:
        params = SaturationParameters(
            landmark_paths=landmark_paths, 
            magnitude=magnitude, 
            saturation_mode=saturation_mode
        )
    except ValidationError as e:
        raise ValueError(f"Invalid parameters for {LayerColourSaturation.__name__}: {e}")

    return LayerColourSaturation(time_config, params)

__all__ = ["SaturationParameters", "layer_colour_saturation"]