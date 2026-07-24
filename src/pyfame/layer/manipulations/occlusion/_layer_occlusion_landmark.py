from pydantic import BaseModel, field_validator, ValidationInfo, ValidationError
from typing import Union, List, Tuple
from pyfame.landmark.facial_landmarks import *
from pyfame.landmark.get_landmark_coordinates import get_relative_landmark_coordinates
from pyfame.layer._layer import Layer, TimingConfiguration
from pyfame.layer.manipulations.mask import mask_from_landmarks
from pyfame.utils.constants import *
import cv2 as cv
import numpy as np

### Note: possibly expand fill_method options to include colour presets in the future

class LandmarkOcclusionParameters(BaseModel):
    """
    Configuration model defining the control parameters for applying
    a landmark-based occlusion to a frame or image.

    This class inherits from pydantic's `BaseModel` to provide validation
    and default handling of landmark occlusion parameters.

    Attributes
    ----------
    fill_method : str or int
        The method used to fill the occluded region. Accepted string values
        are ``"black"`` and ``"mean"``. Accepted integer values are ``8``
        (black) and ``9`` (mean). ``"black"`` fills the region with solid
        black pixels; ``"mean"`` fills it with the mean BGR colour of the
        detected facial region.
    landmark_paths : list of list of tuple of int or list of tuple of int
        A list of one or more closed landmark paths representing the
        region(s) to be occluded.

    Notes
    -----
    - Additional fill methods, such as colour presets, are planned for a
      future release.
    """
    fill_method:Union[int,str]
    landmark_paths:Union[List[List[Tuple[int,...]]], List[Tuple[int,...]]]

    @field_validator("fill_method", mode="before")
    @classmethod
    def check_accepted_value(cls, value, info:ValidationInfo):
        field_name = info.field_name
        occlusion_method_mapping = {8:"black", 9:"mean"}

        if isinstance(value, str):
            value = str.lower(value)
            if value not in {"black", "mean"}:
                raise ValueError(f"Unrecognized value for parameter {field_name}.")
            return value
        
        elif isinstance(value, int):
            if value not in {8,9}:
                raise ValueError(f"Unrecognized value for parameter {field_name}.")
            return occlusion_method_mapping.get(value)
        
        raise TypeError(f"Invalid type provided for {field_name}. Must be one of int or str.")   

class LayerOcclusionLandmark(Layer):
    """
    Manipulation layer that occludes one or more landmark-defined facial
    regions by replacing their pixel values with a uniform fill.

    This layer isolates a region of interest using a landmark-derived binary
    mask and replaces all pixels within that region with either solid black
    or the mean BGR colour of the detected face. The mean fill method
    computes a per-frame facial mean from the full face oval region, making
    the occlusion perceptually less salient than a solid black fill while
    still removing local feature information.

    Parameters
    ----------
    timing_configuration : TimingConfiguration
        Timing configuration controlling onset, offset, and rise/fall durations.
    occlusion_parameters : LandmarkOcclusionParameters
        Configuration model specifying the fill method and landmark region(s).

    Attributes
    ----------
    time_config : TimingConfiguration
        Timing configuration used by the layer.
    occlude_params : LandmarkOcclusionParameters
        Landmark occlusion-specific configuration parameters.
    fill_method : str or int
        The method used to fill the occluded region (``"black"`` or
        ``"mean"``).
    landmark_paths : list of list of tuple of int or list of tuple of int
        Landmark paths defining the region(s) to be occluded.

    Notes
    -----
    - This layer does not support temporal weighting; occlusion is applied
      as a binary on/off effect governed solely by onset and offset times.
    - For the ``"mean"`` fill method, the facial mean is computed fresh
      each frame from the full face oval region using a convex polygon mask,
      ensuring the fill colour adapts to changes in lighting and head
      position over time.
    """

    def __init__(self, timing_configuration:TimingConfiguration, occlusion_parameters:LandmarkOcclusionParameters):
        """
        Initialize a landmark occlusion manipulation layer.

        Parameters
        ----------
        timing_configuration : TimingConfiguration
            Timing configuration controlling when the occlusion effect
            is applied.
        occlusion_parameters : LandmarkOcclusionParameters
            Parameters defining the fill method and target landmark
            region(s).

        Notes
        -----
        - The timing configuration is passed to the superclass ``Layer``.
        - A snapshot of the initial state is taken after initialization to
          allow safe resetting between independent applications.
        """
        self.time_config = timing_configuration
        self.occlude_params = occlusion_parameters

        # Initialise superclass
        super().__init__(self.time_config)

        # Declaring class parameters
        self.fill_method = self.occlude_params.fill_method
        self.landmark_paths = self.occlude_params.landmark_paths

        # Snapshot of initial state
        self._snapshot_state()

    def supports_weight(self):
        """
        Indicate whether the layer supports temporal weighting.

        Returns
        -------
        bool
            ``False``, as landmark occlusion operates as a binary on/off
            effect and does not support continuous rise/fall weighting.
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
            combining both timing and landmark occlusion configuration fields.
        """
        # Dump the pydantic models to get dict of full parameter list
        self._layer_parameters = self.time_config.model_dump()
        self._layer_parameters.update(self.occlude_params.model_dump())
        self._layer_parameters["onset_time_msec"] = self.onset_t
        self._layer_parameters["offset_time_msec"] = self.offset_t
        return dict(self._layer_parameters)
    
    def apply_layer(self, landmarker_coordinates:list[tuple[int,int]], frame:cv.typing.MatLike, dt:float = None):
        """
        Apply the landmark occlusion manipulation to a single frame.

        A binary mask is derived from the configured landmark paths and used
        to identify the region of interest. Pixels within this region are
        replaced according to the configured fill method: either solid black,
        or the mean BGR colour of the face computed from the full face oval
        region for that frame.

        Parameters
        ----------
        landmarker_coordinates : list of tuple of int
            Facial landmark coordinates associated with the current frame.
        frame : MatLike
            Input image frame to which the occlusion is applied.
        dt : float, optional
            Current time (in milliseconds). If ``None``, a weight of 1.0
            is used, applying the occlusion unconditionally.

        Returns
        -------
        MatLike
            The frame with the landmark-defined region filled according to
            the configured fill method, and original pixel values preserved
            outside it.

        Notes
        -----
        - For the ``"mean"`` fill method, the facial mean colour is computed
          per-frame using ``cv.fillConvexPoly`` on the face oval coordinates
          and ``cv.mean`` with the resulting mask, ensuring the fill adapts
          to lighting changes across frames.
        """
        if dt is None:
            weight = 1.0
        else:
            weight = super().compute_weight(dt, self.supports_weight())
        
        if weight == 0.0:
            return frame
        
        # Mask out the region of interest
        mask = mask_from_landmarks(frame, self.landmark_paths, landmarker_coordinates)
        mask = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))

        match self.fill_method:
            case "black":
                occluded = np.where(mask == 255, (0,0,0), frame)
                return occluded
            
            case "mean":
                fo_coords = get_relative_landmark_coordinates(landmarker_coordinates, LANDMARK_FACE_OVAL)

                # Creating boolean masks for the facial landmarks 
                bool_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                bool_mask = cv.fillConvexPoly(bool_mask, np.array(fo_coords), 1)
                bool_mask = bool_mask.astype(bool)

                # Extracting the mean pixel value of the face
                bin_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                bin_mask[bool_mask] = 255
                mean = cv.mean(frame, bin_mask)

                # Fill occlusion regions with facial mean
                mean_img = np.zeros_like(frame, dtype=np.uint8)
                mean_img[:] = mean[:3]
                occluded = np.where(mask == 255, mean_img, frame)
                return occluded

def layer_occlusion_landmark(timing_configuration:TimingConfiguration | None = None, landmark_paths:list[list[tuple[int,...]]] | list[tuple[int,...]]=LANDMARK_FACE_OVAL, fill_method:int|str = OCCLUSION_FILL_BLACK) -> LayerOcclusionLandmark:
    """
    Factory function for the landmark occlusion manipulation layer.
    `LayerOcclusionLandmark` occludes one or more landmark-defined facial
    regions by replacing pixel values within those regions with a uniform
    fill. Two fill methods are supported: solid black, which provides
    complete feature removal, and mean colour, which fills the region with
    the per-frame mean BGR value of the detected face, producing a less
    visually salient occlusion.

    Parameters
    ----------
    timing_configuration : TimingConfiguration or None, optional
        A pydantic model containing timing configurations controlling onset
        and offset. If ``None``, a default ``TimingConfiguration`` is
        instantiated. The default instantiation assumes onset at 0.0 and
        offset at the video's duration.
    landmark_paths : list of list of tuple of int or list of tuple of int, default=LANDMARK_FACE_OVAL
        A list of one or more closed landmark paths representing the
        region(s) to be occluded.
    fill_method : str or int, default=OCCLUSION_FILL_BLACK
        The method used to fill the occluded region. Accepted string values
        are ``"black"`` and ``"mean"``; accepted integer values are ``8``
        (black) and ``9`` (mean).

    Returns
    -------
    LayerOcclusionLandmark
        An instance of the landmark occlusion manipulation layer.

    Raises
    ------
    ValueError
        When provided invalid or unrecognized parameter values.
    """

    # Populate with defaults if None
    time_config = timing_configuration or TimingConfiguration()

    # Validate input parameters
    try:
        params = LandmarkOcclusionParameters(
            fill_method=fill_method, 
            landmark_paths=landmark_paths
        )
    except ValidationError as e:
        raise ValueError(f"Invalid parameters for {LayerOcclusionLandmark.__name__}: {e}")

    return LayerOcclusionLandmark(time_config, params)

__all__ = ["layer_occlusion_landmark", "LandmarkOcclusionParameters"]