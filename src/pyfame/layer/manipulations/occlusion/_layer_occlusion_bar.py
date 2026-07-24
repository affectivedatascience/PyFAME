from pydantic import BaseModel, field_validator, ValidationError, ValidationInfo, NonNegativeInt
from typing import Tuple, List
from pyfame.utils.general_utilities import compute_rotation_angle, compute_slope
from pyfame.utils.constants import *
from pyfame.layer._layer import Layer, TimingConfiguration
from pyfame.landmark.facial_landmarks import *
from pyfame.landmark.get_landmark_coordinates import get_relative_landmark_coordinates
import cv2 as cv
import numpy as np

class BarOcclusionParameters(BaseModel):
    """
    Configuration model defining the control parameters for 
    applying a bar occlusion to a frame or image.

    This class inherits from pydantic's `BaseModel` to provide validation 
    and default handling of bar occlusion parameters.

    Attributes
    ----------
    bar_colour : tuple of int
        A BGR colour tuple in the range [0, 255] per channel, defining
        the fill colour of the occlusion bar.
    vertical_bar_span : int
        The height of the occlusion bar in pixels. Must be non-negative.
    landmark_paths : list of tuple of int
        A landmark path defining the facial region over which the bar
        is centered. Must be one of: ``LANDMARK_LEFT_EYE_REGION``,
        ``LANDMARK_RIGHT_EYE_REGION``, ``LANDMARK_BOTH_EYE_REGIONS``,
        ``LANDMARK_NOSE``, or ``LANDMARK_MOUTH_REGION``.
    y_offset : int
        A vertical pixel offset applied to the bar's computed center
        position, allowing fine adjustment of bar placement along the
        vertical axis. May be negative.
    horizontal_bar_margin : int, default=30
        A horizontal pixel margin added to the occluding bars width. The margin begins
        at the edges of the bounding landmark points, and extends negatively from the 
        left edge, and positively from the right edge.
    """

    bar_colour:Tuple[int,int,int]
    vertical_bar_span:NonNegativeInt
    landmark_paths:List[Tuple[int,...]]
    y_offset:int
    horizontal_bar_margin:int

    @field_validator("bar_colour")
    @classmethod
    def check_in_range(cls, value, info:ValidationInfo):
        field_name = info.field_name
        for elem in value:
            if not (0 <= elem <= 255):
                raise ValueError(f"{field_name} values must lie between 0 and 255.")
        
        return value
    
    @field_validator("landmark_paths", mode="before")
    @classmethod
    def check_compatible_path(cls, value, info:ValidationInfo):
        valid_paths = [LANDMARK_LEFT_EYE_REGION, LANDMARK_RIGHT_EYE_REGION, LANDMARK_BOTH_EYE_REGIONS, LANDMARK_NOSE, LANDMARK_MOUTH_REGION]
        field_name = info.field_name
                
        if value not in valid_paths:
            raise ValueError(f"Incompatible path provided in {field_name}. Please provide one of: LANDMARK_LEFT_EYE_REGION, LANDMARK_RIGHT_EYE_REGION, LANDMARK_BOTH_EYE_REGIONS, LANDMARK_NOSE, LANDMARK_MOUTH_REGION.")
        
        return value

class LayerOcclusionBar(Layer):
    """
    Manipulation layer that overlays a solid-colour rectangular bar over 
    a landmark-defined facial region, rotating to follow head orientation.

    This layer occludes a specified facial region (such as the eye region or
    mouth) by drawing a filled rectangle centered on the region's bounding
    geometry. The bar is rotated to remain aligned with the face plane by
    computing the head's roll angle from a pair of reference landmarks. A
    vertical offset parameter allows fine-grained control over bar placement
    along the face's vertical axis.

    The bounding landmark indices used to determine bar position are computed
    lazily on first application and cached for subsequent frames, avoiding
    redundant computation during video processing.

    Parameters
    ----------
    timing_configuration : TimingConfiguration
        Timing configuration controlling onset, offset, and rise/fall durations.
    occlusion_parameters : BarOcclusionParameters
        Configuration model specifying bar colour, span, landmark region,
        and vertical offset.

    Attributes
    ----------
    time_config : TimingConfiguration
        Timing configuration used by the layer.
    occlude_params : BarOcclusionParameters
        Bar occlusion-specific configuration parameters.
    bar_color : tuple of int
        BGR colour tuple used to fill the occlusion bar.
    vertical_bar_span : int
        Height of the occlusion bar in pixels.
    landmark_paths : list of tuple of int
        Landmark path defining the facial region to be occluded.
    y_offset : int
        Vertical pixel offset applied to the bar's computed center position.
    horizontal_bar_margin: int
        A horizontal pixel margin added to the occluding bars width. The margin begins
        at the edges of the bounding landmark points, and extends negatively from the 
        left edge, and positively from the right edge.
    min_x_lm_id : int
        Index of the landmark with the minimum x-coordinate within the
        region, used to determine the left extent of the bar. Initialised
        to -1 and computed on first application.
    max_x_lm_id : int
        Index of the landmark with the maximum x-coordinate within the
        region, used to determine the right extent of the bar. Initialised
        to -1 and computed on first application.
    min_y_lm_id : int
        Index of the landmark with the minimum y-coordinate within the
        region, used to determine the vertical center of the bar. Initialised
        to -1 and computed on first application.
    max_y_lm_id : int
        Index of the landmark with the maximum y-coordinate within the
        region, used to determine the vertical center of the bar. Initialised
        to -1 and computed on first application.

    Notes
    -----
    - This layer does not support temporal weighting; the bar is applied
      as a binary on/off effect governed solely by onset and offset times.
    - Head roll is estimated from landmarks 162 and 389, which form a line
      approximately parallel to the x-axis when the face is vertically
      upright. The resulting rotation angle is used to align the bar with
      the face plane.
    - Bounding landmark indices are resolved once and cached after the first
      call to ``apply_layer``, avoiding redundant computation on subsequent
      frames.
    """

    def __init__(self, timing_configuration:TimingConfiguration, occlusion_parameters:BarOcclusionParameters):
        """
        Initialize a bar occlusion manipulation layer.

        Parameters
        ----------
        timing_configuration : TimingConfiguration
            Timing configuration controlling when the bar occlusion effect
            is applied.
        occlusion_parameters : BarOcclusionParameters
            Parameters defining bar colour, vertical span, target landmark
            region, and vertical offset.

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

        # Define class parameters
        self.bar_color = self.occlude_params.bar_colour
        self.vertical_bar_span = self.occlude_params.vertical_bar_span
        self.landmark_paths = self.occlude_params.landmark_paths
        self.y_offset = self.occlude_params.y_offset
        self.horizontal_bar_margin = self.occlude_params.horizontal_bar_margin
        self.min_x_lm_id = -1
        self.max_x_lm_id = -1
        self.min_y_lm_id = -1
        self.max_y_lm_id = -1

        # Snapshot of initial state
        self._snapshot_state()
            
    def supports_weight(self):
        """
        Indicate whether the layer supports temporal weighting.

        Returns
        -------
        bool
            ``False``, as bar occlusion operates as a binary on/off effect
            and does not support continuous rise/fall weighting.
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
            combining both timing and bar occlusion configuration fields.
        """
        # Dump the pydantic models to get dict of full parameter list
        self._layer_parameters = self.time_config.model_dump()
        self._layer_parameters.update(self.occlude_params.model_dump())
        self._layer_parameters["onset_time_msec"] = self.onset_t
        self._layer_parameters["offset_time_msec"] = self.offset_t
        return dict(self._layer_parameters)
    
    def set_min_max_landmarks(self, landmarker_coordinates, coordinate_array):
        """
        Resolve and cache the landmark indices defining the bounding extents
        of the target region.

        For each axis, candidate landmarks sharing the extreme coordinate value
        are evaluated as pairs. The pair minimising vertical displacement is
        selected for the x-axis extents, and the pair maximising horizontal
        displacement is selected for the y-axis extents.

        Parameters
        ----------
        landmarker_coordinates : list of tuple of int
            Full list of facial landmark coordinates for the current frame.
        coordinate_array : numpy.ndarray of shape (N, 2)
            Array of pixel coordinates corresponding to the landmarks within
            the target region, used to compute bounding extents.
        """
        # Determine the landmark id's containing the min/max x-values
        xs = coordinate_array[:, 0]
        ys = coordinate_array[:, 1]
        min_x = int(xs.min())
        max_x = int(xs.max())
        min_y = int(ys.min())
        max_y = int(ys.max())
        
        min_x_candidates = [(i, lm) for i,lm in enumerate(landmarker_coordinates) if lm[0] == min_x]
        max_x_candidates = [(i, lm) for i,lm in enumerate(landmarker_coordinates) if lm[0] == max_x]
        min_y_candidates = [(i, lm) for i,lm in enumerate(landmarker_coordinates) if lm[1] == min_y]
        max_y_candidates = [(i, lm) for i,lm in enumerate(landmarker_coordinates) if lm[1] == max_y]

        best_pair_x = None
        best_pair_y = None
        min_vert_diff = float("inf")
        max_horiz_diff = 0

        for i1, lm1 in min_x_candidates:
            for i2, lm2 in max_x_candidates:
                dy = abs(lm2[1] - lm1[1])
                if dy < min_vert_diff:
                    min_vert_diff = dy
                    best_pair_x = (i1, i2)
        
        for i1, lm1 in min_y_candidates:
            for i2, lm2 in max_y_candidates:
                dx = abs(lm2[0] - lm1[0])
                if dx > max_horiz_diff:
                    max_horiz_diff = dx
                    best_pair_y = (i1, i2)
        
        self.min_x_lm_id, self.max_x_lm_id = best_pair_x
        self.min_y_lm_id, self.max_y_lm_id = best_pair_y
    
    def apply_layer(self, landmarker_coordinates:list[tuple[int,int]], frame:cv.typing.MatLike, dt:float):
        """
        Apply the bar occlusion manipulation to a single frame.

        A filled rectangular bar is drawn over the configured facial region,
        centered on the region's bounding geometry and rotated to follow the
        estimated head roll angle. The bar extends horizontally from the
        leftmost to the rightmost landmark of the region, with a 30-pixel
        margin on each side, and spans ``vertical_bar_span`` pixels vertically
        around the computed center, shifted by ``y_offset``.

        Parameters
        ----------
        landmarker_coordinates : list of tuple of int
            Facial landmark coordinates associated with the current frame.
        frame : MatLike
            Input image frame to which the bar occlusion is applied.
        dt : float
            Current time (in milliseconds).

        Returns
        -------
        MatLike
            The frame with the occlusion bar rendered over the configured
            facial region.
        """
        # Bar occlusion does not support weight, so weight will always be 0.0 or 1.0
        if dt is None:
            weight = 1.0
        else:
            weight = super().compute_weight(dt, self.supports_weight())

        if weight == 0.0:
            return frame
        
        h,w = frame.shape[:2]
        # Replace placeholder concave path with its convex sub-paths
        roi_coordinates = get_relative_landmark_coordinates(landmarker_coordinates, self.landmark_paths)
        roi_arr = np.array(roi_coordinates, dtype=int)

        if self.min_x_lm_id == -1 or self.max_x_lm_id == -1 or self.min_y_lm_id == -1 or self.max_y_lm_id == -1:
            self.set_min_max_landmarks(landmarker_coordinates, roi_arr)

        # Calculate the slope of the connecting line & angle to the horizontal
        # landmarks 162, 389 form a paralell line to the x-axis when the face is vertical
        p1 = landmarker_coordinates[162]
        p2 = landmarker_coordinates[389]
        slope = compute_slope(p1, p2)
        rot_angle = compute_rotation_angle(slope_1=slope)
        
        # Compute the center bisecting line of the landmark
        min_x_lm = landmarker_coordinates[self.min_x_lm_id]
        max_x_lm = landmarker_coordinates[self.max_x_lm_id]
        min_y_lm = landmarker_coordinates[self.min_y_lm_id]
        max_y_lm = landmarker_coordinates[self.max_y_lm_id]
        cx = int(round((min_x_lm[0] + max_x_lm[0])/2.0))
        cy = int(round((min_y_lm[1] + max_y_lm[1])/2.0))
        
        # Generate the rectangle
        masked_frame = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        x1 = max(0, min_x_lm[0] - self.horizontal_bar_margin)
        x2 = min(w-1, max_x_lm[0] + self.horizontal_bar_margin)
        y1 = max(0, cy - (self.vertical_bar_span//2))
        y2 = min(h-1, cy + (self.vertical_bar_span//2))
        cv.rectangle(masked_frame, (x1, y1 + self.y_offset), (x2, y2 + self.y_offset), (255,255,255), -1)
        
        # Generate rotation matrix and rotate the rectangle
        rot_mat = cv.getRotationMatrix2D((cx,cy), (rot_angle), 1.0)
        rot_mask = cv.warpAffine(masked_frame, rot_mat, (w,h), flags=cv.INTER_NEAREST)
        rot_mask = rot_mask.astype(bool)

        output_frame = frame.copy().astype(np.uint8)
        output_frame[rot_mask] = self.bar_color
        
        return output_frame

def layer_occlusion_bar(timing_configuration:TimingConfiguration | None = None, landmark_paths:list[tuple[int,...]] = LANDMARK_BOTH_EYE_REGIONS, 
                        bar_colour:tuple[int,int,int] = (0,0,0), vertical_bar_span:int = 100, y_offset:int = 15, horizontal_bar_margin:int = 30) -> LayerOcclusionBar:
    """
    Factory function for the bar occlusion manipulation layer. `LayerOcclusionBar`
    overlays a solid-colour rectangular bar over a specified facial region, rotating
    it to remain aligned with the face plane by estimating head roll from a pair of
    reference landmarks. The bar position can be fine-tuned vertically using the
    ``y_offset`` parameter.

    Parameters
    ----------
    timing_configuration : TimingConfiguration or None, optional
        A pydantic model containing timing configurations controlling onset
        and offset. If ``None``, a default ``TimingConfiguration`` is
        instantiated. The default instantiation assumes onset at 0.0 and
        offset at the video's duration.
    landmark_paths : list of tuple of int, default=LANDMARK_BOTH_EYE_REGIONS
        A landmark path defining the facial region over which the bar is
        centered. Must be one of: ``LANDMARK_LEFT_EYE_REGION``,
        ``LANDMARK_RIGHT_EYE_REGION``, ``LANDMARK_BOTH_EYE_REGIONS``,
        ``LANDMARK_NOSE``, or ``LANDMARK_MOUTH_REGION``.
    bar_colour : tuple of int, default=(0, 0, 0)
        A BGR colour tuple in the range [0, 255] per channel, defining the
        fill colour of the occlusion bar. Defaults to black.
    vertical_bar_span : int, default=100
        The height of the occlusion bar in pixels. Must be non-negative.
    y_offset : int, default=15
        A vertical pixel offset applied to the bar's computed center position.
        Positive values shift the bar downward; negative values shift it upward.
    horizontal_bar_margin : int, default=30
        A horizontal pixel margin added to the occluding bars width. The margin begins
        at the edges of the bounding landmark points, and extends negatively from the 
        left edge, and positively from the right edge.

    Returns
    -------
    LayerOcclusionBar
        An instance of the bar occlusion manipulation layer.

    Raises
    ------
    ValueError
        When provided invalid, out-of-range, or incompatible parameter values.
    """
    # Populate with defaults if None
    time_config = timing_configuration or TimingConfiguration()

    # Validate input parameters
    try:
        params = BarOcclusionParameters(
            bar_colour=bar_colour, 
            vertical_bar_span=vertical_bar_span,
            landmark_paths=landmark_paths,
            y_offset=y_offset,
            horizontal_bar_margin=horizontal_bar_margin
        )
    except ValidationError as e:
        raise ValueError(f"Invalid parameters for {LayerOcclusionBar.__name__}: {e}")

    return LayerOcclusionBar(time_config, params)

__all__ = ["layer_occlusion_bar", "BarOcclusionParameters"]