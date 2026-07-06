from pydantic import BaseModel, ValidationError, NonNegativeFloat, field_validator, ValidationInfo
from typing import Union, List, Tuple, Optional
from pyfame.layer.layer import Layer, TimingConfiguration
from pyfame.layer.manipulations.mask.mask_from_landmarks import mask_from_landmarks
from pyfame.landmark.facial_landmarks import *
from pyfame.landmark.get_landmark_coordinates import get_relative_landmark_coordinates
import cv2 as cv
import numpy as np
from operator import itemgetter

class PencilSketchParameters(BaseModel):
    """
    Configuration model defining the control parameters for applying
    a pencil sketch stylisation to a frame or image.

    This class inherits from pydantic's `BaseModel` to provide validation
    and default handling of pencil sketch parameters.

    Attributes
    ----------
    landmark_paths : list of list of tuple of int or list of tuple of int or None
        An optional list of one or more closed landmark paths defining the
        region in which the stylisation is applied. If ``None``, the effect
        is applied to the entire frame.
    detail_level : float
        A normalised value in the range [0.0, 1.0] controlling the spatial
        scales of illumination normalisation, bilateral smoothing, and
        adaptive thresholding. Lower values produce coarser, bolder sketches;
        higher values preserve finer facial detail.
    threshold_bias : float
        A non-negative constant subtracted from the local mean in the
        adaptive thresholding step, controlling the sensitivity of edge
        detection. Higher values produce sparser, lighter edges; lower
        values produce denser, darker edges.
    """

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
    """
    Manipulation layer that applies a pencil sketch stylisation to a frame
    or a landmark-defined facial region.

    The sketch effect is produced through a three-stage pipeline. First,
    illumination gradients are estimated by low-pass Gaussian filtering and
    used to normalise the greyscale frame, reducing the influence of uneven
    lighting on edge detection. Second, a bilateral filter smooths the
    normalised image while preserving landmark edges, reducing noise-driven
    spurious contours. Third, adaptive mean thresholding extracts local
    contrast boundaries to produce the final sketch-like contour image.

    All three spatial scales (illumination blur sigma, bilateral filter
    diameter, and adaptive threshold block size) are derived from a single
    ``detail_level`` parameter via a power-law mapping, and are further
    scaled by the detected face width to maintain consistent visual
    granularity across subjects at different distances from the camera.

    If ``landmark_paths`` is provided, the sketch effect is composited only
    within the specified region, leaving the remainder of the frame unchanged.

    Parameters
    ----------
    timing_configuration : TimingConfiguration
        Timing configuration controlling onset, offset, and rise/fall
        durations.
    ps_parameters : PencilSketchParameters
        Configuration model specifying the landmark region, detail level,
        and threshold bias.

    Attributes
    ----------
    time_config : TimingConfiguration
        Timing configuration used by the layer.
    ps_params : PencilSketchParameters
        Pencil sketch-specific configuration parameters.
    landmark_paths : list of list of tuple of int or list of tuple of int or None
        Landmark paths defining the region in which the sketch is applied,
        or ``None`` if the effect is applied to the full frame.
    detail_level : float
        Normalised detail level controlling spatial filter scales.
    illum_scale : float
        Sigma of the Gaussian blur used for illumination estimation,
        expressed as a fraction of face width. Derived from ``detail_level``
        via ``map_detail_to_spatial_scales``.
    filter_scale : float
        Diameter of the bilateral filter as a fraction of face width.
        Derived from ``detail_level`` via ``map_detail_to_spatial_scales``.
    thresh_scale : float
        Block size of the adaptive threshold as a fraction of face width.
        Derived from ``detail_level`` via ``map_detail_to_spatial_scales``.
    thresh_const : float
        Constant subtracted from the local mean in the adaptive thresholding
        step, controlling edge detection sensitivity.

    Notes
    -----
    - This layer does not support temporal weighting; the sketch effect is
      applied as a binary on/off effect governed solely by onset and offset
      times.
    - Spatial filter scales are computed once at initialisation from
      ``detail_level`` and remain fixed across frames. Face width is
      re-estimated each frame to scale these fixed proportions to the
      subject's current size in the frame.
    """

    def __init__(self, timing_configuration:TimingConfiguration, ps_parameters:PencilSketchParameters):
        """
        Initialize a pencil sketch stylisation layer.

        Parameters
        ----------
        timing_configuration : TimingConfiguration
            Timing configuration controlling when the sketch effect is
            applied.
        ps_parameters : PencilSketchParameters
            Parameters defining the landmark region, detail level, and
            threshold bias.

        Notes
        -----
        - The timing configuration is passed to the superclass ``Layer``.
        - Spatial scales are derived from ``detail_level`` once at
          initialisation via ``map_detail_to_spatial_scales`` and stored
          as ``illum_scale``, ``filter_scale``, and ``thresh_scale``.
        - A snapshot of the initial state is taken after initialization to
          allow safe resetting between independent applications.
        """
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
        self._layer_parameters.update(self.ps_params.model_dump())
        self._layer_parameters["onset_time_msec"] = self.onset_t
        self._layer_parameters["offset_time_msec"] = self.offset_t
        return dict(self._layer_parameters)
    
    def map_detail_to_spatial_scales(self, d, gamma=1.5) -> Tuple[float,float,float]:
        """
        Map a normalised detail level to the three spatial filter scales
        used by the sketch pipeline.

        Each scale is computed via an exponential interpolation between
        its minimum and maximum values, raised to the power of
        ``d ** gamma``. This power-law transformation produces a more
        perceptually linear response to changes in ``detail_level`` than
        direct linear interpolation.

        Parameters
        ----------
        d : float
            Normalised detail level in the range [0.0, 1.0].
        gamma : float, default=1.5
            Exponent applied to ``d`` before interpolation, controlling
            the non-linearity of the mapping. Values greater than 1.0
            compress the low end of the detail range; values less than
            1.0 compress the high end.

        Returns
        -------
        scale_i : float
            Illumination blur sigma as a fraction of face width, in the
            range [0.02, 0.15].
        scale_b : float
            Bilateral filter diameter as a fraction of face width, in the
            range [0.01, 0.05].
        scale_t : float
            Adaptive threshold block size as a fraction of face width, in
            the range [0.02, 0.1].
        """
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
        """
        Apply the pencil sketch stylisation to a single frame.

        The face width is estimated from the face oval landmark bounding
        box and used to scale all spatial filter parameters for the current
        frame. The frame is converted to greyscale, illumination-normalised
        via Gaussian low-pass filtering, smoothed with a bilateral filter,
        and thresholded using adaptive mean thresholding to produce a
        contour image. If ``landmark_paths`` is set, the contour image is
        composited onto the original frame within the masked region only;
        otherwise it replaces the full frame.

        Parameters
        ----------
        landmarker_coordinates : list of tuple of int
            Facial landmark coordinates associated with the current frame.
        frame : MatLike
            Input image frame to which the stylisation is applied.
        dt : float
            Current time (in milliseconds).

        Returns
        -------
        MatLike
            The stylised frame, with the pencil sketch effect applied either
            within the landmark-defined region or across the full frame,
            depending on whether ``landmark_paths`` is set.

        Notes
        -----
        - The bilateral filter diameter is clamped to a minimum of 3 pixels
          to ensure smoothing still occurs for subjects with small face
          widths in the frame.
        - The adaptive threshold block size is clamped to a minimum of 3
          and forced to an odd value via bitwise OR with 1, as required
          by ``cv.adaptiveThreshold``.
        - The sketch output is a three-channel greyscale image (converted
          from the single-channel threshold output via ``COLOR_GRAY2BGR``),
          ensuring compatibility with the BGR frame format expected by
          downstream processing.
        """
        if dt is None:
            weight = 1.0
        else:
            weight = super().compute_weight(dt, self.supports_weight())

        if weight == 0.0:
            return frame
        
        # Compute facial width for filter scaling downstream
        fo_coords = get_relative_landmark_coordinates(landmarker_coordinates, LANDMARK_FACE_OVAL)

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
    """
    Factory function for the pencil sketch stylisation layer.
    `LayerStylisePencilSketch` applies a sketch-like stylisation to a frame
    or landmark-defined facial region by combining illumination normalisation,
    bilateral smoothing, and adaptive mean thresholding. All spatial filter
    scales are derived from a single ``detail_level`` parameter and scaled
    proportionally to face width, ensuring consistent visual granularity
    across subjects at different distances from the camera.

    Parameters
    ----------
    timing_configuration : TimingConfiguration or None, optional
        A pydantic model containing timing configurations controlling onset
        and offset. If ``None``, a default ``TimingConfiguration`` is
        instantiated. The default instantiation assumes onset at 0.0 and
        offset at the video's duration.
    landmark_paths : list of list of tuple of int or list of tuple of int or None, default=None
        An optional list of one or more closed landmark paths defining the
        region in which the stylisation is applied. If ``None``, the effect
        is applied to the entire frame.
    detail_level : float, default=0.35
        A normalised value in the range [0.0, 1.0] controlling the spatial
        granularity of the sketch. Lower values produce coarser, bolder
        sketches; higher values preserve finer facial detail.
    threshold_bias : float, default=7.0
        A non-negative constant subtracted from the local mean in the
        adaptive thresholding step. Higher values produce sparser, lighter
        edges; lower values produce denser, darker edges.

    Returns
    -------
    LayerStylisePencilSketch
        An instance of the pencil sketch stylisation layer.

    Raises
    ------
    ValueError
        When provided invalid or out-of-range parameter values.
    """
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

__all__ = ["layer_stylise_pencil_sketch", "PencilSketchParameters"]