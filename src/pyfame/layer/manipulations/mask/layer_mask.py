from pydantic import BaseModel, field_validator, ValidationError, ValidationInfo
from typing import Union, List, Tuple
from pyfame.utils.constants import *
from pyfame.layer.manipulations.mask.mask_from_landmarks import mask_from_landmarks
from pyfame.layer.layer import Layer, TimingConfiguration
from pyfame.landmark.facial_landmarks import LANDMARK_FACE_OVAL
import numpy as np
import cv2 as cv

class MaskingParameters(BaseModel):
    """
    Configuration model defining the control parameters for 
    masking a frame or image to a landmark-defined region.

    This class inherits from pydantic's `BaseModel` to provide validation 
    and default handling of masking parameters.

    Attributes
    ----------
    landmark_paths : list of list of tuple of int or list of tuple of int
        A list of one or more closed landmark paths representing the 
        region to be retained in the output. Pixels outside this region
        are replaced with the specified background colour.
    background_colour : tuple of int
        A BGR colour tuple in the range [0, 255] per channel, used to 
        fill pixels outside the masked region.
    """

    landmark_paths:Union[List[List[Tuple[int,...]]], List[Tuple[int,...]]]
    background_colour:Tuple[int,int,int]

    @field_validator("background_colour")
    @classmethod
    def check_in_range(cls, value, info:ValidationInfo):
        field_name = info.field_name
        for elem in value:
            if not (0 <= elem <= 255):
                raise ValueError(f"{field_name} values must lie between 0 and 255.")
        
        return value

class LayerMask(Layer):
    """
    Manipulation layer that masks a frame to one or more landmark-defined
    regions, replacing pixels outside those regions with a solid background colour.

    This layer isolates a facial region of interest by combining a landmark-derived
    mask with a foreground segmentation mask computed via Otsu thresholding and
    flood-filling. The intersection of these two masks ensures that only pixels
    belonging to both the specified landmark region and the detected foreground
    subject are retained; all other pixels are replaced with the configured
    background colour.

    Parameters
    ----------
    timing_configuration : TimingConfiguration
        Timing configuration controlling onset, offset, and rise/fall durations.
    masking_parameters : MaskingParameters
        Configuration model specifying the landmark region(s) and background colour.

    Attributes
    ----------
    time_config : TimingConfiguration
        Timing configuration used by the layer.
    mask_params : MaskingParameters
        Masking-specific configuration parameters.
    landmark_paths : list of list of tuple of int or list of tuple of int
        Landmark paths defining the region(s) to be retained in the output.
    background_colour : tuple of int
        BGR colour tuple used to fill pixels outside the masked region.

    Notes
    -----
    - This layer does not support temporal weighting; the mask is applied
      as a binary on/off effect governed solely by onset and offset times.
    - Foreground segmentation via Otsu thresholding prevents background
      pixels from being incorrectly included within the landmark mask region,
      which can occur when landmark coordinates extend beyond the subject boundary.
    """

    def __init__(self, timing_configuration:TimingConfiguration, masking_parameters:MaskingParameters):
        """
        Initialize a masking manipulation layer.

        Parameters
        ----------
        timing_configuration : TimingConfiguration
            Timing configuration controlling when the masking effect
            is applied.
        masking_parameters : MaskingParameters
            Parameters defining the target landmark region(s) and the
            background fill colour.

        Notes
        -----
        - The timing configuration is passed to the superclass ``Layer``.
        - A snapshot of the initial state is taken after initialization to
          allow safe resetting between independent applications.
        """
        self.time_config = timing_configuration
        self.mask_params = masking_parameters

        # Initialise the superclass
        super().__init__(self.time_config)

        # Define class parameters
        self.landmark_paths = self.mask_params.landmark_paths
        self.background_colour = self.mask_params.background_colour

        # Snapshot of initial state
        self._snapshot_state()
    
    def supports_weight(self):
        """
        Indicate whether the layer supports temporal weighting.

        Returns
        -------
        bool
            ``False``, as masking operates as a binary on/off effect and
            does not support continuous rise/fall weighting.
        """
        return False
    
    def get_layer_parameters(self) -> dict:
        """
        Return the parameters defining this layer.

        This method exposes all configurable parameters required to reproduce
        the layer's behavior.

        Returns
        -------
        dict
            Dictionary mapping parameter names to their current values,
            combining both timing and masking configuration fields.
        """
        # Dump the pydantic models to get dict of full parameter list
        self._layer_parameters = self.time_config.model_dump()
        self._layer_parameters.update(self.mask_params.model_dump())
        self._layer_parameters["onset_time_msec"] = self.onset_t
        self._layer_parameters["offset_time_msec"] = self.offset_t
        return dict(self._layer_parameters)
    
    def apply_layer(self, landmarker_coordinates:list[tuple[int,int]], frame:cv.typing.MatLike, dt:float):
        """
        Apply the layer's masking manipulation to a single frame.

        The landmark-defined region of interest is intersected with a
        foreground segmentation mask to isolate the subject within that
        region. Pixels outside this intersection are replaced with the
        configured background colour.

        Parameters
        ----------
        landmarker_coordinates : list of tuple of int
            Facial landmark coordinates associated with the current frame.
        frame : MatLike
            Input image frame to which the masking is applied.
        dt : float
            Current time (in milliseconds).

        Returns
        -------
        MatLike
            The masked frame, with pixels outside the landmark region and
            foreground boundary replaced by the configured background colour.
        """
        # Masking does not support weight, so weight will always be 0.0 or 1.0
        if dt is None:
            weight = 1.0
        else:
            weight = super().compute_weight(dt, self.supports_weight())

        # Occurs when the dt is less than the onset_time, or greater than the offset_time
        if weight == 0.0:
            return frame
        
        # Mask out the region of interest
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

        # Remove unwanted background inclusions in the masked area
        masked_frame = cv.bitwise_and(mask, foreground)
        masked_frame = np.reshape(masked_frame, (masked_frame.shape[0], masked_frame.shape[1], 1))
        masked_frame = np.where(masked_frame == 255, frame, self.background_colour)
        masked_frame = masked_frame.astype(np.uint8)
        return masked_frame

def layer_mask(timing_configuration:TimingConfiguration | None = None, landmark_paths:list[list[tuple[int,...]]] | list[tuple[int,...]] = LANDMARK_FACE_OVAL, background_colour:tuple[int,int,int] = (0,0,0)) -> LayerMask:
    """
    Factory function for the masking manipulation layer. `LayerMask` isolates
    one or more landmark-defined facial regions by replacing all pixels outside
    those regions with a configurable background colour. A foreground segmentation
    mask derived from Otsu thresholding and flood-filling is intersected with the
    landmark mask to prevent spurious background inclusions within the region of
    interest.

    Parameters
    ----------
    timing_configuration : TimingConfiguration or None, optional
        A pydantic model containing timing configurations controlling onset
        and offset. If ``None``, a default ``TimingConfiguration`` is
        instantiated. The default instantiation assumes onset at 0.0 and
        offset at the video's duration.
    landmark_paths : list of list of tuple of int or list of tuple of int, default=LANDMARK_FACE_OVAL
        A list of one or more closed landmark paths representing the region
        to be retained in the output.
    background_colour : tuple of int, default=(0, 0, 0)
        A BGR colour tuple in the range [0, 255] per channel, used to fill
        all pixels outside the masked region. Defaults to black.

    Returns
    -------
    LayerMask
        An instance of the masking manipulation layer.

    Raises
    ------
    ValueError
        When provided invalid or out-of-range parameter values.

    Notes
    -----
    - An invert mask option is planned for a future release, which would
      retain the background and replace the landmark region instead.
    """
    # Populate with defaults if None
    time_config = timing_configuration or TimingConfiguration()

    # Validate input parameters
    try:
        params = MaskingParameters(
            landmark_paths=landmark_paths, 
            background_colour=background_colour
        )
    except ValidationError as e:
        raise ValueError(f"Invalid parameters for {LayerMask.__name__}: {e}")

    return LayerMask(time_config, params)

__all__ = ["MaskingParameters", "layer_mask"]