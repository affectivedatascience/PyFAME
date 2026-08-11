from pydantic import BaseModel, field_validator, ValidationInfo, ValidationError, PositiveFloat, NonNegativeInt
from typing import Union, List, Tuple, Any
from pyfame.landmark.facial_landmarks import *
from pyfame.landmark.get_landmark_coordinates import get_relative_landmark_coordinates
from pyfame.layer._layer import Layer, TimingConfiguration
from pyfame.layer.manipulations.mask import mask_from_landmarks
from pyfame.utils.constants import *
import cv2 as cv
import numpy as np
from skimage.util import *
# Reimplement passing of idx_to_track list for precise landmark inclusion

class PointLightParameters(BaseModel):
    """
    Configuration model defining the control parameters for applying
    a point-light display stylisation to a frame or image.

    This class inherits from pydantic's `BaseModel` to provide validation
    and default handling of point-light display parameters.

    Attributes
    ----------
    landmark_paths : list of list of tuple of int or list of tuple of int
        Landmark index paths defining facial regions to include. Each
        tuple corresponds to a landmark index (or index group depending
        on upstream representation). Nested lists allow multiple regions.
    point_density : float
        Proportion of valid landmarks to render, in the range (0.0, 1.0].
        Values less than 1.0 randomly subsample landmarks using a
        pseudo-normal distribution across index groups.
    point_colour : tuple of int
        RGB colour of rendered landmark points, with values in [0, 255].
    point_radius : int
        The radius of the rendered points.
    display_history_vectors : bool
        If ``True``, overlays motion history vectors between landmark
        positions across frames.
    history_method : str or int, default = `SHOW_HISTORY_ORIGIN` Literal[32]
        Method used to render motion history:
        - ``"origin"``: vectors drawn between current points and initial frame origin points.
        - ``"relative"``: accumulated point displacement over a temporal window.   
    history_window_msec : int, default = 500
        Temporal window (in milliseconds) used for relative history
        accumulation. Ignored if ``history_method="origin"``.
    history_vector_colour : tuple of int
        RGB colour of history vectors, with values in [0, 255].
    maintain_background : bool
        If ``True``, overlays points on the original frame. Otherwise,
        renders on a blank (black) background.
    invert_colours : bool
        If ``True``, inverts the final output image.

    Raises
    ------
    ValueError
        If colour values fall outside [0, 255], or if numeric parameters
        are outside their valid ranges.
    TypeError
        If ``history_method`` is not of type ``int`` or ``str``.
    """

    landmark_paths:Union[List[List[Tuple[int,...]]], List[Tuple[int,...]]]
    point_density:PositiveFloat
    point_colour:tuple[NonNegativeInt, NonNegativeInt, NonNegativeInt]
    point_radius:int
    display_history_vectors:bool
    history_method:Union[int,str] = SHOW_HISTORY_ORIGIN
    history_window_msec:NonNegativeInt = 500
    history_vector_colour:tuple[NonNegativeInt, NonNegativeInt, NonNegativeInt]
    maintain_background:bool
    invert_colours:bool

    @field_validator("point_colour", "history_vector_colour")
    @classmethod
    def check_in_range(cls, value, info:ValidationInfo):
        field_name = info.field_name
        for elem in value:
            if not (0 <= elem <= 255):
                raise ValueError(f"{field_name} values must lie in the range 0 - 255.")
        
        return value
    
    @field_validator("point_density")
    @classmethod
    def check_normal_range(cls, value, info:ValidationInfo):
        field_name = info.field_name
        if not (0.0 < value <= 1.0):
            raise ValueError(f"Invalid value for parameter {field_name}. Must lie in the range 0.0 - 1.0.")
        
        return value

    @field_validator("point_radius")
    @classmethod
    def check_radius_range(cls, value, info:ValidationInfo):
        field_name = info.field_name
        if not (1 <= value <= 25):
            raise ValueError(f"{field_name} must lie in the range 1 - 25.")

        return value
    
    @field_validator("history_method", mode="before")
    @classmethod
    def check_accepted_value(cls, value, info:ValidationInfo):
        field_name = info.field_name

        if isinstance(value, str):
            if value not in {"origin", "relative"}:
                raise ValueError(f"Unrecognized value for parameter {field_name}.")
            return value
        
        elif isinstance(value, int):
            if value not in {32, 33}:
                raise ValueError(f"Unrecognized value for parameter {field_name}.")
            return value
        
        raise TypeError(f"Invalid type for parameter {field_name}. Expected int or str.")

class LayerStylisePointLight(Layer):
    """
    Stylisation layer for generating facial point-light displays.

    This layer renders selected facial landmarks as discrete points
    (dots) on either a blank or original background. Optionally,
    it visualizes landmark motion over time using vector histories.

    Parameters
    ----------
    timing_configuration : TimingConfiguration
        Temporal configuration controlling when the layer is active.
    point_light_parameters : PointLightParameters
        Validated parameter set defining point rendering and history behavior.

    Attributes
    ----------
    frame_history : list of ndarray
        Buffer of past history masks used for relative motion visualization.
    prev_points : list of tuple of int or None
        Landmark positions from the previous frame.
    idx_to_display : ndarray
        Indices of landmarks selected for rendering based on density.
    point_density : float
        Proportion of landmarks displayed.
    point_colour : tuple of int
        RGB colour of rendered points.
    point_radius : int
        The radius of the rendered points.
    maintain_background : bool
        Whether to preserve the original frame.
    display_history_vectors : bool
        Whether motion vectors are rendered.
    history_method : str or int
        Method used for motion visualization.
    history_window_msec : int
        Time window for relative history accumulation.
    history_colour : tuple of int
        RGB colour of motion vectors.
    landmark_paths : list
        Landmark region definitions.
    invert_colours : bool
        Whether to invert the final output image.

    Notes
    -----
    - This layer does not support temporal weighting; the sketch effect is
      applied as a binary on/off effect governed solely by onset and offset
      times.
    """

    def __init__(self, timing_configuration:TimingConfiguration, point_light_parameters:PointLightParameters):
        """
        Initialize a point light display stylisation layer.

        Parameters
        ----------
        timing_configuration: TimingConfiguration
            Timing configuration controlling when the point-light effect
            comes on and off.
        point_light_parameters: PointLightParameters
            Parameters defining the region of application, point size, 
            point density and method of history vector display.

        Notes
        -----
        - The timing configuration is passed to the superclass ``Layer``.
        - A snapshot of the initial state is taken after initialization to
          allow safe resetting between independent applications.
        """
        self.time_config = timing_configuration
        self.pl_params = point_light_parameters

        # Initialise superclass
        super().__init__(self.time_config)

        # Declare class parameters
        self.frame_history = []
        self.prev_points = None
        self.point_density = self.pl_params.point_density
        self.point_colour = self.pl_params.point_colour
        self.point_radius = self.pl_params.point_radius
        self.maintain_background = self.pl_params.maintain_background
        self.display_history_vectors = self.pl_params.display_history_vectors
        self.history_method = self.pl_params.history_method
        self.history_window_msec = self.pl_params.history_window_msec
        self.history_colour = self.pl_params.history_vector_colour
        self.landmark_paths = self.pl_params.landmark_paths
        self.invert_colours = self.pl_params.invert_colours

        # Snapshot of initial state
        self._snapshot_state()
    
    def supports_weight(self):
        """
        Indicate whether the layer supports temporal weighting.

        Returns
        -------
        bool
            ``False``, as pencil sketch stylisation operates as a binary
            on/off effect and does not support continuous rise/fall
            weighting.
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
            combining both timing and pencil sketch configuration fields.
        """
        # Dump the pydantic models to get dict of full parameter list
        self._layer_parameters = self.time_config.model_dump()
        self._layer_parameters.update(self.pl_params.model_dump())
        self._layer_parameters["onset_time_msec"] = self.onset_t
        self._layer_parameters["offset_time_msec"] = self.offset_t
        return dict(self._layer_parameters)

    def _convert_landmarks_to_coord_list(self, landmarker_coordinates:Any, landmark:list) -> list[tuple[int,int]]:
        """
        Takes a raw landmark path or list of paths, and returns a flattened
        list of all (x,y) pixel coordinates of the landmarks passed.

        Parameters
        ----------
        landmarker_coordinates: list
            The list object returned by the mediapipe face landmarker
            instance, containing the (x,y) coordinates of all 478 
            facial landmarks.
        landmark: list
            A list of tuples or a list of list of tuples containing
            one or more landmark paths.
        
        Returns
        -------
        List of tuple of int
            A flattened list of screen pixel coordinates.
        
        """
        lm_coordinates = []

        if isinstance(landmark[0], list):
            # Multiple landmarks
            for path in landmark:
                lm_coordinates.extend(get_relative_landmark_coordinates(landmarker_coordinates, path))
        else:
            lm_coordinates.extend(get_relative_landmark_coordinates(landmarker_coordinates, landmark))

        return lm_coordinates

    def apply_layer(self, landmarker_coordinates:list[tuple[int,int]], frame:cv.typing.MatLike, dt:float):
        """
        Apply the point-light stylisation effect to a single frame.

        Parameters
        ----------
        landmarker_coordinates : list of tuple of int
            List of (x, y) landmark coordinates for the current frame.
        frame : MatLike
            Input image frame in OpenCV-compatible format.
        dt : float or None
            Time delta (in seconds or milliseconds depending on pipeline)
            used to compute temporal weighting. If ``None``, the layer is
            applied with full weight.

        Returns
        -------
        MatLike
            Output frame with point-light stylisation applied.

        Notes
        -----
        - If ``point_density < 1.0``, a subset of landmarks is sampled once
        and reused across frames.
        - If ``maintain_background=False``, the output is rendered on a
        blank frame.
        - If ``invert_colours=True``, the final image is inverted.

        Raises
        ------
        ValueError
            If invalid landmark indices are encountered during masking.
        """
        if dt is None:
            weight = 1.0
        else:
            weight = super().compute_weight(dt, self.supports_weight())

        if weight == 0.0:
            return frame
        
        mask = np.zeros_like(frame, dtype=np.uint8)
        frame_history_count = round(30 * (self.history_window_msec/1000))
        # Update later for variable frame rates

        if self.maintain_background:
            output_img = frame.copy()
        else:
            output_img = np.zeros_like(frame, dtype=np.uint8)

        cur_points = self._convert_landmarks_to_coord_list(landmarker_coordinates, self.landmark_paths)
        history_mask = np.zeros_like(frame, dtype=np.uint8)
        
        if self.prev_points == None or self.display_history_vectors == False:
            self.prev_points = cur_points.copy()

            # Draw a circle (point) over each of the current_points
            for point in cur_points:
                x1, y1 = point
                if x1 > 0 and y1 > 0:
                    cv.circle(output_img, (x1, y1), self.point_radius, self.point_colour, -1)

        elif self.history_method == SHOW_HISTORY_ORIGIN:
            # If show_history is true, display vector paths of all points;
            # On top of the points themselves
            for (old, new) in zip(self.prev_points, cur_points):
                x0, y0 = old
                x1, y1 = new
                cv.line(mask, (int(x0), int(y0)), (int(x1), int(y1)), self.history_colour, 2)
                cv.circle(output_img, (int(x1), int(y1)), self.point_radius, self.point_colour, -1)

            #self.prev_points = cur_points.copy()
            output_img = cv.add(output_img, mask)
            mask = np.zeros_like(frame, dtype=np.uint8)

        else:
            # If show_history is true, display vector paths of all points
            for (old, new) in zip(self.prev_points, cur_points):
                x0, y0 = old
                x1, y1 = new
                cv.line(mask, (int(x0), int(y0)), (int(x1), int(y1)), self.history_colour, 2)
                cv.circle(output_img, (int(x1), int(y1)), self.point_radius, self.point_colour, -1)

            # Relative vector history only displays up to history_window_msec seconds of history
            if len(self.frame_history) < frame_history_count:
                self.frame_history.append(mask)
                for img in self.frame_history:
                    history_mask = cv.bitwise_or(history_mask, img)
            else:
                self.frame_history.append(mask)
                self.frame_history.pop(0)
                for img in self.frame_history:
                    history_mask = cv.bitwise_or(history_mask, img)

            self.prev_points = cur_points.copy()
            output_img = cv.add(output_img, history_mask)
            mask = np.zeros_like(frame, dtype=np.uint8)

        if self.invert_colours:
            return cv.bitwise_not(output_img)
        return output_img
        
def layer_stylise_point_light(timing_configuration:TimingConfiguration | None = None, landmark_paths:list[list[tuple[int,int]]] | list[tuple[int,int]]=LANDMARK_FACE_OVAL, 
                              point_density:float = 1.0, point_colour:tuple[int,int,int] = (255,255,255), point_radius:int = 3, display_history_vectors:bool = False, 
                              history_window_msec:int = 500, history_method:int|str = SHOW_HISTORY_ORIGIN, history_vector_colour:tuple[int,int,int] = (0,0,255), 
                              maintain_background:bool = False, invert_colours:bool = False):
    """
    Factory function for the point-light stylisation layer. This
    function validates input parameters, constructs and returns a
    ``LayerStylisePointLight`` instance. 

    This layer renders selected facial landmarks as discrete points
    (dots) on either a blank or original background. Optionally,
    it visualizes landmark motion over time using historical motion vectors.

    Parameters
    ----------
    timing_configuration : TimingConfiguration, optional
        Temporal configuration for the layer. If ``None``, a default
        configuration is used.
    landmark_paths : list, optional
        Landmark region definitions used to select visible points.
    point_density : float, optional
        Proportion of landmarks to render, in the range (0.0, 1.0].
    point_colour : tuple of int, optional
        RGB colour of rendered points.
    point_radius : int
        The radius of the rendered points.
    display_history_vectors : bool, optional
        Whether to display motion history vectors.
    history_window_msec : int, optional
        Time window for relative motion history accumulation.
    history_method : {"origin", "relative"} or int, optional
        Method used to compute motion history.
    history_vector_colour : tuple of int, optional
        RGB colour of motion vectors.
    maintain_background : bool, optional
        Whether to overlay points on the original frame.
    invert_colours : bool, optional
        Whether to invert the final output image.

    Returns
    -------
    LayerStylisePointLight
        Configured point-light stylisation layer.

    Raises
    ------
    ValueError
        If parameter validation fails.
    """
    # Populate with defaults if None
    time_config = timing_configuration or TimingConfiguration()

    # Validate input params
    try:
        params = PointLightParameters(
            landmark_paths=landmark_paths, 
            point_density=point_density, 
            point_colour=point_colour, 
            point_radius=point_radius,
            display_history_vectors=display_history_vectors, 
            history_method=history_method, 
            history_window_msec=history_window_msec,
            history_vector_colour=history_vector_colour, 
            maintain_background=maintain_background, 
            invert_colours=invert_colours
        )
    except ValidationError as e:
        raise ValueError(f"Invalid parameters for {LayerStylisePointLight.__name__}: {e}")
    
    return LayerStylisePointLight(time_config, params)

__all__ = ["layer_stylise_point_light", "PointLightParameters"]