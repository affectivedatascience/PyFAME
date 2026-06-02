from pydantic import BaseModel, NonNegativeFloat, field_validator, ValidationInfo, ValidationError
from typing import Union, List, Tuple, Any
from pyfame.layer.manipulations.mask import mask_from_landmarks
from pyfame.layer.layer import Layer, TimingConfiguration
from pyfame.landmark.facial_landmarks import *
from pyfame.utils.constants import *
from pyfame.utils.general_utilities import get_landmark_names
from pyfame.landmark.blendshape_smoother import EyeBlendshapeSmoother
import cv2 as cv
import numpy as np

class RecolourParameters(BaseModel):
    """
    Configuration model defining the control parameters for 
    manipulating the colour of a frame or image.

    This class inherits from pydantic's `BaseModel` to provide validation 
    and default handling of colouring parameters.

    Attributes
    ----------
    landmark_paths : list of list of tuple of int or list of tuple of int
        A list of one or more closed landmark paths representing the 
        region in which the manipulation will be applied.
    focus_colour : string or int
        A string or integer specifier of the focus colour for the 
        current manipulation. `LayerColourRecolour` can shift and 
        manipulate the degree of red, green, blue or yellow in an
        image or frame.
    magnitude : float
        The degree by which to increase or decrease the colour. 
        Taken as a percentage of the total value range of each colour,
        i.e. a magnitude of 10 would shift the colour (128 * 10) = 12.8 units
        on the a* or b* axis.
    """
    landmark_paths:Union[List[List[Tuple[int,...]]], List[Tuple[int,...]]]
    focus_colour:Union[str, int]
    magnitude:NonNegativeFloat

    @field_validator('focus_colour')
    @classmethod
    def check_valid_focus_colour(cls, value, info:ValidationInfo):
        field_name = info.field_name
        if isinstance(value, int):
            if value not in SHIFT_COLOUR_OPTIONS:
                raise ValidationError(f"{field_name} has been provided an unrecognized value.")
        elif isinstance(value, str):
            if value.lower() not in {"red", "green", "blue", "yellow"}:
                raise ValidationError(f"{field_name} has been provided an unrecognized value.")
        
        return value

class LayerColourRecolour(Layer):
    """
    Manipulation layer that selectively shifts colour components within
    landmark-defined regions over time.

    This layer performs targeted colour recolouring by modifying channels
    in the perceptually uniform CIE L*a*b* colour space. A single focus
    colour (red, green, blue, or yellow) is adjusted by a specified
    magnitude within one or more facial landmark regions.

    Special handling is included for eye regions to ensure robust behaviour
    during blinking, using blendshape-based smoothing to conditionally
    apply scleral recolouring.

    Parameters
    ----------
    timing_configuration : TimingConfiguration
        Timing configuration controlling onset, offset, rise/fall durations,
        and temporal weighting behavior.
    recolour_parameters : RecolourParameters
        Configuration model specifying landmark regions, focus colour, and
        recolouring magnitude.

    Attributes
    ----------
    time_config : TimingConfiguration
        Timing configuration used by the layer.
    colour_params : RecolourParameters
        Colour-specific configuration parameters.
    landmark_paths : list of list of tuple of int or list of tuple of int
        Landmark paths defining the region(s) in which recolouring is applied.
    focus_colour : str or int
        Colour channel to be emphasized or attenuated.
    magnitude : float
        The degree by which to increase or decrease the colour. 
        Taken as a percentage of the total value range of each colour,
        i.e. a magnitude of 10 would shift the colour (128 * 10) = 12.8 units
        on the a* or b* axis.
    eye_blendshape_smoother : EyeBlendshapeSmoother
        Temporal smoother used to detect eye openness for scleral recolouring.
    colour_left_sclera : bool or None
        Whether recolouring is applied to the left eye sclera.
    colour_right_sclera : bool or None
        Whether recolouring is applied to the right eye sclera.
    adjusted_landmark_paths : list
        Landmark paths with eye regions removed when scleral recolouring
        is handled separately.

    Notes
    -----
    - Colour manipulation is performed in CIE L*a*b* space to improve
      perceptual consistency across lighting conditions.
    - The effective recolouring magnitude at a given time is scaled by
      the temporal weight computed from the timing configuration.
    - This layer supports continuous temporal weighting.
    """

    def __init__(self, timing_configuration:TimingConfiguration, recolour_parameters:RecolourParameters):
        """
        Initialize a recolouring manipulation layer.

        Parameters
        ----------
        timing_configuration : TimingConfiguration
            Timing configuration controlling when the recolouring effect
            is applied and how it transitions over time.
        recolour_parameters : RecolourParameters
            Parameters defining the target landmark region(s), focus colour,
            and recolouring magnitude.

        Notes
        -----
        - The timing configuration is passed to the superclass ``Layer``.
        - Internal state is initialized to support blink-aware eye colouring.
        - A snapshot of the initial state is taken after initialization to
          allow safe resetting between independent applications.
        """
        self.time_config = timing_configuration
        self.colour_params = recolour_parameters

        # Initialise the superclass
        super().__init__(configuration=self.time_config)
        
        # Define class parameters
        self.landmark_paths = self.colour_params.landmark_paths
        self.focus_colour = self.colour_params.focus_colour
        self.magnitude = self.colour_params.magnitude
        self.eye_blendshape_smoother = EyeBlendshapeSmoother(frame_window_size=1) 
        self.colour_left_sclera = None
        self.colour_right_sclera = None
        self.adjusted_landmark_paths = self.colour_params.landmark_paths.copy()

        # Snapshot of initial state
        self._snapshot_state()
    
    def supports_weight(self) -> bool:
        """
        Indicate whether the layer supports temporal weighting.

        Returns
        -------
        bool
            `True` if the layer supports continuous rise/fall weighting,
            `False` if it operates as a binary on/off manipulation.
        """
        return True
    
    def get_layer_parameters(self) -> dict:
        """
        Return the parameters defining this layer.

        This method should expose all configurable parameters required
        to reproduce the layer's behavior.

        Returns
        -------
        dict
            Dictionary mapping parameter names to their current values.
        """
        # Dump the pydantic models to get dict of full parameter list
        self._layer_parameters = self.time_config.model_dump()
        self._layer_parameters.update(self.colour_params.model_dump())
        self._layer_parameters["onset_time_msec"] = self.onset_t
        self._layer_parameters["offset_time_msec"] = self.offset_t
        return dict(self._layer_parameters)
    
    def _adjust_landmark_paths_eye_landmarks(self):
        """
        Sets internal state parameters required for colouring the irises.

        Colouring the irises requires special behaviour to handle blinking.
        This method is called to update state parameters indicating if iris 
        colouring is required, and it will remove the iris landmark paths 
        from the `landmark_paths` list to prevent duplicate colouring.
        """
        if self.colour_left_sclera is None or self.colour_right_sclera is None:
            # Check if scleral colouring is required
            landmark_names = get_landmark_names(self.landmark_paths)

            if isinstance(landmark_names, list):
                for name in landmark_names:
                    if name == 'LANDMARK_BOTH_EYES':
                        self.colour_left_sclera = True
                        self.colour_right_sclera = True

                        if isinstance(self.adjusted_landmark_paths[0], list):
                            self.adjusted_landmark_paths.pop(self.adjusted_landmark_paths.index(LANDMARK_BOTH_EYES))
                        else:
                            self.adjusted_landmark_paths = []
                    
                    elif name == 'LANDMARK_LEFT_EYE':
                        self.colour_left_sclera = True
                        self.colour_right_sclera = False if self.colour_right_sclera is None else self.colour_right_sclera

                        if isinstance(self.adjusted_landmark_paths[0], list):
                            self.adjusted_landmark_paths.pop(self.adjusted_landmark_paths.index(LANDMARK_LEFT_EYE))
                        else:
                            self.adjusted_landmark_paths = []
                    
                    elif name == 'LANDMARK_RIGHT_EYE':
                        self.colour_left_sclera = False if self.colour_left_sclera is None else self.colour_left_sclera
                        self.colour_right_sclera = True

                        if isinstance(self.adjusted_landmark_paths[0], list):
                            self.adjusted_landmark_paths.pop(self.adjusted_landmark_paths.index(LANDMARK_RIGHT_EYE))
                        else:
                            self.adjusted_landmark_paths = []

            else:
                if landmark_names == 'LANDMARK_BOTH_EYES':
                        self.colour_left_sclera = True
                        self.colour_right_sclera = True

                        if isinstance(self.adjusted_landmark_paths[0], list):
                            self.adjusted_landmark_paths.pop(self.adjusted_landmark_paths.index(LANDMARK_BOTH_EYES))
                        else:
                            self.adjusted_landmark_paths = []
                    
                elif landmark_names == 'LANDMARK_LEFT_EYE':
                    self.colour_left_sclera = True
                    self.colour_right_sclera = False if self.colour_right_sclera is None else self.colour_right_sclera

                    if isinstance(self.adjusted_landmark_paths[0], list):
                        self.adjusted_landmark_paths.pop(self.adjusted_landmark_paths.index(LANDMARK_LEFT_EYE))
                    else:
                        self.adjusted_landmark_paths = []
                
                elif landmark_names == 'LANDMARK_RIGHT_EYE':
                    self.colour_left_sclera = False if self.colour_left_sclera is None else self.colour_left_sclera
                    self.colour_right_sclera = True

                    if isinstance(self.adjusted_landmark_paths[0], list):
                        self.adjusted_landmark_paths.pop(self.adjusted_landmark_paths.index(LANDMARK_RIGHT_EYE))
                    else:
                        self.adjusted_landmark_paths = []
    
    def apply_layer(self, landmarker_coordinates:list[tuple[int, int]], frame:cv.typing.MatLike, dt:float, blendshapes:Any):
        """
        Apply the layer's manipulation to a single frame.

        Parameters
        ----------
        landmarker_coordinates : list of tuple of int
            Facial landmark coordinates associated with the current frame.
        frame : MatLike
            Input image frame to which the manipulation is applied.
        dt : float
            Current time (in milliseconds).
        blendshapes : Any
            A dictionary of blendshape scores returned by the mediapipe `FaceLandmarker`.

        Returns
        -------
        MatLike
            The manipulated frame.
        
        Raises
        ------
        ValueError
            Given invalid or unrecognized parameter values.
        """
        if dt is None:
            weight = 1.0
        else:
            weight = super().compute_weight(dt, self.supports_weight())
        
        # Occurs when the current dt is less than the onset_time, or greater than the offset_time
        if weight == 0.0:
            return frame

        self._adjust_landmark_paths_eye_landmarks()

        # Get a mask of our region of interest
        if len(self.adjusted_landmark_paths) > 0:
            mask = mask_from_landmarks(frame, self.adjusted_landmark_paths, landmarker_coordinates)
        else:
            mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        left_eye_open, right_eye_open = self.eye_blendshape_smoother.update(blendshapes)
        le_mask = mask_from_landmarks(frame, LANDMARK_LEFT_EYE, landmarker_coordinates)
        re_mask = mask_from_landmarks(frame, LANDMARK_RIGHT_EYE, landmarker_coordinates)
        li_mask = mask_from_landmarks(frame, LANDMARK_LEFT_IRIS, landmarker_coordinates)
        ri_mask = mask_from_landmarks(frame, LANDMARK_RIGHT_IRIS, landmarker_coordinates)

        # Extract just the scleral mask by taking the xor of the iris and the whole eye
        left_sclera_mask = cv.bitwise_xor(le_mask, li_mask)
        right_sclera_mask = cv.bitwise_xor(re_mask, ri_mask)
        
        # Convert input image to CIE La*b* color space (perceptually uniform space)
        img_LAB = cv.cvtColor(frame.astype(np.float32) / 255.0, cv.COLOR_BGR2LAB)
        # Split the image into individual channels for precise colour manipulation
        l,a,b = cv.split(img_LAB)

        # Shift the various colour channels according to the user-specified focus_colour
        match self.focus_colour:
            case "red" | 4:
                delta = 127.0 * ((weight * self.magnitude)/100.0)
                a = np.where(mask==255, a + delta, a)

                if self.colour_left_sclera and left_eye_open:
                    a = np.where(left_sclera_mask==255, a + (weight * self.magnitude), a)
                if self.colour_right_sclera and right_eye_open:
                    a = np.where(right_sclera_mask==255, a + (weight * self.magnitude), a)

                a = np.clip(a, -128, 127)

            case "blue" | 5:
                delta = 128.0 * ((weight * self.magnitude)/100.0)
                b = np.where(mask==255, b - delta, b)

                if self.colour_left_sclera and left_eye_open:
                    b = np.where(left_sclera_mask==255, b - (weight * self.magnitude), b)
                if self.colour_right_sclera and right_eye_open:
                    b = np.where(right_sclera_mask==255, b - (weight * self.magnitude), b)

                b = np.clip(b, -128, 127)

            case "green" | 6:
                delta = 128.0 * ((weight * self.magnitude)/100.0)
                a = np.where(mask==255, a - delta, a)

                if self.colour_left_sclera and left_eye_open:
                    a = np.where(left_sclera_mask==255, a - (weight * self.magnitude), a)
                if self.colour_right_sclera and right_eye_open:
                    a = np.where(right_sclera_mask==255, a - (weight * self.magnitude), a)

                a = np.clip(a, -128, 127)

            case "yellow" | 7:
                delta = 127.0 * ((weight * self.magnitude)/100.0)
                b = np.where(mask==255, b + delta, b)

                if self.colour_left_sclera and left_eye_open:
                    b = np.where(left_sclera_mask==255, b + (weight * self.magnitude), b)
                if self.colour_right_sclera and right_eye_open:
                    b = np.where(right_sclera_mask==255, b + (weight * self.magnitude), b)

                b = np.clip(b, -128, 127)

            case _:
                raise ValueError("Unidentified or incompatible focus colour passed to LayerColourRecolour.")
        
        # After shifting the colour channels, merge the individual channels back into one image
        img_LAB = cv.merge([l,a,b])

        # Convert CIE La*b* back to BGR
        result = cv.cvtColor(img_LAB, cv.COLOR_LAB2BGR)
        result = (result * 255.0).astype(np.uint8)

        return result

def layer_colour_recolour(timing_configuration:TimingConfiguration | None = None, landmark_paths:list[list[tuple[int,...]]] | list[tuple[int,...]] = LANDMARK_FACE_OVAL, focus_colour:str|int = "red", magnitude:float = 10.0) -> LayerColourRecolour:
    """
    Factory function for the colour manipulation layer. `LayerColourRecolour` leverages the 
    La*b* colour space to perform perceptually uniform colour shifts in a specified region of the
    face. Special behaviour is defined for colouring of the sclera, where the eye blendshapes are
    used to ensure colouring is only applied when the eyes are open.

    Parameters
    ----------
    timing_configuration : TimingConfiguration or None, optional
        A pydantic model containing timing configurations controlling onset, 
        offset, rise/fall durations, and weighting curves. If `None`, a 
        default `TimingConfiguration` is instantiated. The default 
        instantiation assumes a linear rise and fall transition, onset at 
        0.0 and offset at the video's duration.
    landmark_paths: list of list of tuple of int or list of tuple of int, default=LANDMARK_FACE_OVAL
        A list of one or more closed landmark paths representing the 
        region in which the manipulation will be applied.
    focus_colour : string or int
        A string or integer specifier of the focus colour for the 
        current manipulation. `LayerColourRecolour` can shift and 
        manipulate the degree of red, green, blue or yellow in an
        image or frame.
    magnitude: float, default=10.0
        The degree by which to increase or decrease the colour. 
        Taken as a percentage of the total value range of each colour,
        i.e. a magnitude of 10 would shift the colour (128 * 10) = 12.8 units
        on the a* or b* axis.

    Returns
    -------
    LayerColourRecolour
        an instance of the colour manipulation layer.

    Raises
    ------
    ValueError
        When provided invalid or unrecognized parameter values.

    """
    # Populate with defaults if None
    config = timing_configuration or TimingConfiguration()

    # Validate input parameters
    try:
        params = RecolourParameters(
            landmark_paths=landmark_paths, 
            focus_colour=focus_colour, 
            magnitude=magnitude
        )
    except ValidationError as e:
        raise ValueError(f"Invalid parameters for {LayerColourRecolour.__name__}: {e}")
    
    return LayerColourRecolour(config, params)

__all__ = ["RecolourParameters", "layer_colour_recolour"]