from pydantic import BaseModel, field_validator, ValidationError, ValidationInfo, NonNegativeInt, PositiveFloat
from typing import Union, Optional, Any
from pyfame.landmark.facial_landmarks import *
from pyfame.landmark.blendshape_smoother import EyeBlendshapeSmoother
from pyfame.landmark.get_landmark_coordinates import get_relative_landmark_coordinates
from pyfame.file_access import *
from pyfame.utils import compute_rotation_angle, compute_slope
from pyfame.layer.layer import Layer, TimingConfiguration
from pyfame.layer.manipulations.mask.mask_from_landmarks import mask_from_landmarks
from pyfame.utils.exceptions import FileReadError
import cv2 as cv
import numpy as np
from pathlib import Path
import os
from collections import deque

_OVERLAY_DIR = Path(__file__).parent / "overlay_images"
# (127, 6) for right cheek scaling, center lm 119
# (6, 356) for left cheek scaling, center 348
# (127, 356) for facial-width scaling, center lm 6
# consider adding overlay-specific y-offsets
_OVERLAY_MAPPING = {
    "sunglasses": {
        "path": _OVERLAY_DIR / "sunglasses.png",
        "anchor_landmarks": (127, 356),
        "center_landmark": 6,
        "scale_factor": None
    },
    "glasses": {
        "path": _OVERLAY_DIR / "glasses.png",
        "anchor_landmarks": (127, 356),
        "center_landmark": 6,
        "scale_factor": None
    },
    "teardrop_short_1": { # Update this with hard coded tear_left_cheek, tear_right_cheek
        "path": _OVERLAY_DIR / "teardrops" / "teardrop_1.png",
        "anchor_landmarks": (6, 356),
        "center_landmark": 348,
        "scale_factor": 0.2
    },
    "face_mask": {
        "path": _OVERLAY_DIR / "face_mask.png",
        "anchor_landmarks": (127, 356),
        "center_landmark": 6,
        "scale_factor": None
    }
}

_OVERLAY_CACHE: dict[str, cv.typing.MatLike] = {}

def compute_scale(landmarker_coordinates:list[dict], anchor_landmarks:tuple[int,int], scale_factor:float = 1.0) -> float:
    """
    Compute a scale value from the Euclidean distance between two anchor
    landmarks, optionally weighted by a scale factor.

    The distance between the two anchor landmarks is used as a proxy for
    the size of the facial region of interest, allowing overlay images to
    be resized proportionally to the subject's face across frames.

    Parameters
    ----------
    landmarker_coordinates : list of dict
        Full list of facial landmark coordinates for the current frame.
    anchor_landmarks : tuple of int
        A pair of landmark indices whose Euclidean distance defines the
        reference scale.
    scale_factor : float, default=1.0
        A multiplier applied to the computed distance. Values greater than
        1.0 enlarge the derived scale; values less than 1.0 reduce it.
        If ``None`` is passed, defaults to 1.0.

    Returns
    -------
    float
        The scaled Euclidean distance between the two anchor landmarks.
    """
    if scale_factor is None:
        scale_factor = 1.0
    
    p1 = np.array([
        landmarker_coordinates[anchor_landmarks[0]][0],
        landmarker_coordinates[anchor_landmarks[0]][1]
    ])

    p2 = np.array([
        landmarker_coordinates[anchor_landmarks[1]][0],
        landmarker_coordinates[anchor_landmarks[1]][1]
    ])
    
    return np.linalg.norm(p1-p2) * scale_factor

def load_overlay(overlay_name:str, landmarker_coordinates:list[tuple[int,int]], scale_factor:float | None = None) -> tuple[cv.typing.MatLike, tuple, tuple, float]:
    """
    Load and prepare an overlay image for compositing onto a frame.

    Retrieves the overlay image from a cache, falling back to disk on the
    first access. The overlay's configuration is looked up from a predefined
    mapping by name, which specifies the file path, scale factor, anchor
    landmarks, and center landmark. The center point and scale are computed
    from the current frame's landmark coordinates to support overlay
    positioning and sizing proportional to the subject's face.

    Parameters
    ----------
    overlay_name : str
        The name of the overlay to load, used as a key into the predefined
        overlay configuration mapping and the image cache.
    landmarker_coordinates : list of tuple of int
        Facial landmark coordinates for the current frame, used to compute
        the overlay center point and scale.
    scale_factor : float or None, optional
        An optional multiplier applied to the computed anchor distance when
        deriving the overlay scale. If ``None``, the scale factor defined
        in the overlay's configuration mapping is used.

    Returns
    -------
    img : MatLike
        The loaded overlay image, including its alpha channel if present.
    anchor_landmarks : tuple
        The pair of landmark indices used to derive the overlay scale.
    center_point : tuple of int
        The pixel coordinates of the landmark used to center the overlay.
    scale : float
        The computed scale value used to resize the overlay image.

    Raises
    ------
    FileReadError
        If the overlay image file cannot be read from disk.
    """
    global _OVERLAY_CACHE

    # Pre-defined overlay type
    config = _OVERLAY_MAPPING[overlay_name]
    file_path = str(config["path"])
    scale_factor = config["scale_factor"]
    anchor_landmarks = config["anchor_landmarks"]
    center_landmark = config["center_landmark"]

    # lazy-loading the image
    if overlay_name not in _OVERLAY_CACHE:
        _OVERLAY_CACHE[overlay_name] = cv.imread(file_path, cv.IMREAD_UNCHANGED)
        if _OVERLAY_CACHE[overlay_name] is None:
            raise FileReadError("Error reading in file.")
    
    img = _OVERLAY_CACHE[overlay_name]
        
    center_point = landmarker_coordinates[center_landmark]
    
    if scale_factor is None:
        scale = compute_scale(landmarker_coordinates, anchor_landmarks)
    else:
        scale = compute_scale(landmarker_coordinates, anchor_landmarks, scale_factor)
    
    return (img, anchor_landmarks, center_point, scale)
        
class OverlayParameters(BaseModel):
    """
    Configuration model defining the control parameters for compositing
    an overlay image onto a frame.

    This class inherits from pydantic's `BaseModel` to provide validation
    and default handling of overlay parameters.

    Attributes
    ----------
    overlay_type : str or int
        The overlay to apply. Accepted string values are ``"sunglasses"``,
        ``"glasses"``, ``"teardrop"``, ``"face_mask"``, and ``"pupils"``. Accepted
        integer values are ``43`` (sunglasses), ``44`` (glasses), ``45``
        (teardrop), ``46`` (face mask), and ``47`` (pupils). A file path to a 
        custom overlay image may also be provided as a string. Integer inputs 
        are normalised to their string equivalents on validation.
    overlay_scale_factor : float or None, optional
        An optional multiplier applied to the computed anchor distance when
        deriving the overlay scale. If ``None``, the scale factor defined
        in the overlay's configuration mapping is used.
    y_offset : int
        A vertical pixel offset applied to the overlay's computed center
        position, allowing fine adjustment of placement along the vertical
        axis. May be negative.
    x_offset : int
        A horizontal pixel offset applied to the overlay's computed center
        position, allowing fine adjustment of placement along the 
        horizontal axis. May be negative.
    pupil_scale_factor : float
        The ratio of pupil diameter to eye canthal width, used to derive
        the pupil radius for the ``"pupils"`` overlay. Must lie in the
        normalised range (0.0, 1.0]. Based on adult anatomical ratios,
        typical values lie in the range [0.130, 0.300].
    """

    overlay_type:Union[NonNegativeInt, str]
    overlay_scale_factor:Optional[float] = None
    y_offset:int
    x_offset:int
    pupil_scale_factor:PositiveFloat

    @field_validator("overlay_type", mode="before")
    @classmethod
    def check_accepted_value(cls, value, info:ValidationInfo):
        field_name = info.field_name

        if isinstance(value, str):
            value = str.lower(value)
            if value in {"sunglasses", "glasses", "teardrop", "face_mask", "pupils"}:
                return value
            
            elif os.path.isfile(value):
                return value
            
            raise ValueError(f"Unrecognized value or invalid file path provided to parameter {field_name}.")
            
        elif isinstance(value, int):
            if value not in {43,44,45,46,47}:
                raise ValueError(f"Unrecognized value for parameter {field_name}.")
            
            mapping = {43:"sunglasses", 44:"glasses", 45:"teardrop", 46:"face_mask", 47:"pupils"}
            return mapping.get(value)
        
        raise TypeError(f"Invalid type for parameter {field_name}. Expected int or str.")
    
    @field_validator("pupil_scale_factor")
    @classmethod
    def check_normalised_range(cls, value, info:ValidationInfo):
        field_name = info.field_name

        if not 0.0 < value <= 1.0:
            raise ValueError(f"Parameter {field_name} must lie in the normalised range (0.0-1.0].")
        
        return value
    
class LayerOverlay(Layer):
    """
    Manipulation layer that composites a pre-defined or custom overlay image
    onto a frame, aligned and scaled to follow the subject's facial geometry.

    For image-based overlays (sunglasses, glasses, teardrops, masks), the overlay
    is loaded from disk on first use and cached for subsequent frames. It is
    scaled proportionally to the distance between a pair of anchor landmarks,
    rotated to match the estimated head roll angle, and alpha-blended onto the
    frame centered on a designated landmark.

    For the ``"pupils"`` overlay, synthetic circular pupils are drawn onto
    the iris regions of both eyes. Pupil radius is derived from the canthal
    width of each eye and the configured ``pupil_scale_factor``, smoothed
    over a short temporal window to reduce per-frame jitter. Blink-aware
    logic via blendshape smoothing ensures pupils are only rendered when
    the corresponding eye is open. Pupil size is modulated by temporal
    weight, supporting gradual onset and offset transitions.

    Parameters
    ----------
    timing_configuration : TimingConfiguration
        Timing configuration controlling onset, offset, and rise/fall
        durations.
    overlay_parameters : OverlayParameters
        Configuration model specifying the overlay type, scale factor,
        vertical offset, and pupil scale factor.

    Attributes
    ----------
    time_config : TimingConfiguration
        Timing configuration used by the layer.
    overlay_params : OverlayParameters
        Overlay-specific configuration parameters.
    overlay_type : str
        The overlay to apply (e.g. ``"sunglasses"``, ``"pupils"``).
    overlay_scale_factor : float or None
        Optional scale multiplier applied when deriving the overlay size
        from anchor landmark distance.
    y_offset : int
        Vertical pixel offset applied to the overlay's computed center.
    x_offset : int
        Horizontal pixel offset applied to the overlay's computed center.
    pupil_scale_factor : float
        Ratio of pupil diameter to eye canthal width, used to derive
        pupil radius for the ``"pupils"`` overlay.
    eye_blendshape_smoother : EyeBlendshapeSmoother
        Temporal smoother used to detect eye openness for blink-aware
        pupil rendering.
    left_pupil_radius_deque : deque of int
        Rolling window of recent left pupil radius values, smoothed to
        reduce per-frame jitter. Maximum length of 5.
    right_pupil_radius_deque : deque of int
        Rolling window of recent right pupil radius values, smoothed to
        reduce per-frame jitter. Maximum length of 5.

    Notes
    -----
    - Image-based overlays are loaded lazily and cached globally in
      ``_OVERLAY_CACHE`` across layer instances and frames.
    - Overlay rotation is performed on a zero-padded square canvas sized
      to the overlay's diagonal, preventing corner clipping during rotation.
    - The ``"pupils"`` overlay is the only overlay type that supports
      continuous temporal weighting; all other overlay types operate as
      binary on/off effects.
    """

    def __init__(self, timing_configuration:TimingConfiguration, overlay_parameters:OverlayParameters):
        """
        Initialize an overlay manipulation layer.

        Parameters
        ----------
        timing_configuration : TimingConfiguration
            Timing configuration controlling when the overlay effect is
            applied and, for the ``"pupils"`` overlay, how its intensity
            transitions over time.
        overlay_parameters : OverlayParameters
            Parameters defining the overlay type, scale factor, vertical
            offset, and pupil scale factor.

        Notes
        -----
        - Rolling deques for left and right pupil radii are initialised
          with a maximum length of 5 to smooth radius estimates across
          frames.
        - A snapshot of the initial state is taken after initialization to
          allow safe resetting between independent applications.
        """
        self.time_config = timing_configuration
        self.overlay_params = overlay_parameters

        # Initialise superclass
        super().__init__(self.time_config)
        
        # Declare class parameters
        self.overlay_type = self.overlay_params.overlay_type
        self.overlay_scale_factor = self.overlay_params.overlay_scale_factor
        self.y_offset = self.overlay_params.y_offset
        self.x_offset = self.overlay_params.x_offset
        self.pupil_scale_factor = self.overlay_params.pupil_scale_factor
        
        # For eye related overlays
        self.eye_blendshape_smoother = EyeBlendshapeSmoother(frame_window_size=1)
        self.left_pupil_radius_deque = deque(maxlen=5)
        self.right_pupil_radius_deque = deque(maxlen=5)

        # Snapshot of initial state
        self._snapshot_state()

    def supports_weight(self):
        """
        Indicate whether the layer supports temporal weighting.

        Returns
        -------
        bool
            ``False`` for all overlay types except ``"pupils"``, which
            modulates the rendered pupil radius by the current temporal
            weight. This return value is bypassed internally for the
            ``"pupils"`` overlay in ``apply_layer``.
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
            combining both timing and overlay configuration fields.
        """
        # Dump the pydantic models to get dict of full parameter list
        self._layer_parameters = self.time_config.model_dump()
        self._layer_parameters.update(self.overlay_params.model_dump())
        self._layer_parameters["onset_time_msec"] = self.onset_t
        self._layer_parameters["offset_time_msec"] = self.offset_t
        return dict(self._layer_parameters)
    
    def apply_layer(self, landmarker_coordinates:list[tuple[int,int]], frame:cv.typing.MatLike, dt:float, blendshapes:Any):
        """
        Apply the overlay manipulation to a single frame.

        For image-based overlays, the configured overlay image is loaded,
        scaled to the anchor landmark distance, rotated to match head roll,
        and alpha-blended onto the frame at the designated center landmark,
        adjusted by ``y_offset``.

        For the ``"pupils"`` overlay, synthetic circular pupils are drawn
        onto the iris regions of both eyes. Pupil radius is derived from
        the canthal width of each eye, smoothed over a rolling window, and
        scaled by the current temporal weight. Pupils are only rendered
        when the corresponding eye is detected as open by the blendshape
        smoother. The result is blended with the original frame using a
        fixed 70/30 weighted average.

        Parameters
        ----------
        landmarker_coordinates : list of tuple of int
            Facial landmark coordinates associated with the current frame.
        frame : MatLike
            Input image frame to which the overlay is applied.
        dt : float
            Current time (in milliseconds).
        blendshapes : Any
            A dictionary of blendshape scores returned by the mediapipe
            ``FaceLandmarker``, used to determine eye openness for
            blink-aware rendering.

        Returns
        -------
        MatLike
            The frame with the configured overlay composited onto it.

        Notes
        -----
        - For image-based overlays, the overlay is padded to a square
          canvas sized to its diagonal before rotation, preventing corner
          clipping. The alpha channel is used as a per-pixel blend mask
          during compositing.
        - For the ``"pupils"`` overlay, each pupil is constrained to the
          visible scleral region via a bitwise intersection of the pupil
          circle mask and the eye landmark mask, preventing the pupil from
          extending beyond the visible eye boundary.
        """
        if dt is None:
            weight = 1.0
        elif self.overlay_type == "pupils":
            weight = super().compute_weight(dt, True)
        else:
            weight = super().compute_weight(dt, self.supports_weight())

        if weight == 0.0:
            return frame
        
        overlayed_frame = frame.copy()
        left_eye_open, right_eye_open = self.eye_blendshape_smoother.update(blendshapes)

        if self.overlay_type == "pupils":

            left_iris_arr = np.array(get_relative_landmark_coordinates(landmarker_coordinates, LANDMARK_LEFT_IRIS))
            right_iris_arr = np.array(get_relative_landmark_coordinates(landmarker_coordinates, LANDMARK_RIGHT_IRIS))

            li_xs = left_iris_arr[:, 0]
            li_ys = left_iris_arr[:, 1]
            ri_xs = right_iris_arr[:, 0]
            ri_ys = right_iris_arr[:, 1]

            # Use the centroid of the iris landmarks to center the pupil overlay
            centroid_left_pupil = (int(round(np.mean(li_xs))), int(round(np.mean(li_ys)) + 3))
            centroid_right_pupil = (int(round(np.mean(ri_xs))), int(round(np.mean(ri_ys)) + 3))

            # left and right cantha of the left eye: landmarks 362, 263
            p1 = landmarker_coordinates[362]
            p2 = landmarker_coordinates[263]
            left_eye_canthal_width = np.linalg.norm(np.array(p1) - np.array(p2))

            # left and right cantha of the right eye: landmarks 33, 133
            p1 = landmarker_coordinates[33]
            p2 = landmarker_coordinates[133]
            right_eye_canthal_width = np.linalg.norm(np.array(p1) - np.array(p2))

            # Scale pupil radius by the eye width, according to average 
            # eye-width to pupil diameter ratios of adults found in the literature. 
            # Typical adult eye width is 27-32mm, while typical diameter of a dilated pupil is 4-8mm.
            # Thus, typical pupil diameter:eye-width ratios in adults [0.130-0.300]
            self.left_pupil_radius_deque.append(int((left_eye_canthal_width * self.pupil_scale_factor)/2))
            self.right_pupil_radius_deque.append(int((right_eye_canthal_width * self.pupil_scale_factor)/2))
            left_pupil_radius = int(np.mean(self.left_pupil_radius_deque))
            right_pupil_radius = int(np.mean(self.right_pupil_radius_deque))
            pupil_radius = int((left_pupil_radius + right_pupil_radius)/2)

            pupil_overlay = frame.copy()

            if left_eye_open:
                sclera_mask = mask_from_landmarks(frame, LANDMARK_LEFT_EYE, landmarker_coordinates)
                pupil_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                cv.circle(pupil_mask, centroid_left_pupil, int(weight * pupil_radius), (255,255,255), -1)

                # Overlay the pupil only where it lies in the visible sclera
                masked_pupil = cv.bitwise_and(pupil_mask, sclera_mask)
                pupil_overlay[masked_pupil > 0] = (0,0,0)

            if right_eye_open:
                sclera_mask = mask_from_landmarks(frame, LANDMARK_RIGHT_EYE, landmarker_coordinates)
                pupil_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                cv.circle(pupil_mask, centroid_right_pupil, int(weight * pupil_radius), (255,255,255), -1)

                # Overlay the pupil only where it lies in the visible sclera
                masked_pupil = cv.bitwise_and(pupil_mask, sclera_mask)
                pupil_overlay[masked_pupil > 0] = (0,0,0)
            
            overlayed_frame = cv.addWeighted(pupil_overlay, 0.7, overlayed_frame, 0.3, 0)

        else:
            overlay, anchor_lms, center_point, scale = load_overlay(
                overlay_name=self.overlay_type,
                landmarker_coordinates=landmarker_coordinates,
                scale_factor=self.overlay_scale_factor
            )

            # Rescaling the overlay to match 
            scaling_factor = 1/(overlay.shape[1]/(scale))
            overlay = cv.resize(src=overlay, dsize=None, fx=scaling_factor, fy=scaling_factor, interpolation=cv.INTER_AREA)

            # Save the overlay img dimensions for later
            overlay_width = overlay.shape[1]
            overlay_height = overlay.shape[0]
            
            # Compute the angle from the x axis 
            p1 = landmarker_coordinates[anchor_lms[0]]
            p2 = landmarker_coordinates[anchor_lms[1]]
            cur_slope = compute_slope(p1, p2)
            rotation_angle = compute_rotation_angle(slope_1=cur_slope)
                
            # Add transparent padding prior to rotation
            diag_size = int(np.ceil(np.sqrt(overlay_height**2 + overlay_width**2)))
            pad_h = (diag_size-overlay_height)//2
            pad_w = (diag_size-overlay_width)//2
            padded = np.zeros((diag_size, diag_size, 4), dtype=np.uint8)
            padded[pad_h:pad_h+overlay_height, pad_w:pad_w + overlay_width] = overlay

            # Get center point of padded overlay
            padded_height = padded.shape[0]
            padded_width = padded.shape[1]
            padded_center = (padded_width//2, padded_height//2)

            # Rotate the overlay to match the angle of inclination of the head
            rot_mat = cv.getRotationMatrix2D(padded_center, rotation_angle, 1)
            overlay = cv.warpAffine(padded, rot_mat, (padded_width, padded_height), 
                                    flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=(0,0,0,0))

            # Generate a binary mask of the overlay for addition onto original frame
            overlay_img = overlay[:,:,:3]
            overlay_mask = overlay[:,:,3] / 255.0
            overlay_mask = overlay_mask[:,:,np.newaxis]

            x_pos = center_point[0] - padded_width//2 + self.x_offset
            y_pos = center_point[1] - padded_height//2 + self.y_offset

            roi = frame[y_pos:y_pos + padded_height, x_pos:x_pos + padded_width]
            blended = (1.0 - overlay_mask) * roi + overlay_mask * overlay_img

            overlayed_frame[y_pos:y_pos + padded_height, x_pos:x_pos + padded_width] = blended.astype(np.uint8)

        return overlayed_frame
        
def layer_overlay(timing_configuration:TimingConfiguration | None = None, overlay_type:int|str = "sunglasses", 
                  overlay_scale_factor:float | None = None, y_offset:int = 10, x_offset:int = 0, pupil_scale_factor:float = 0.25) -> LayerOverlay:
    """
    Factory function for the overlay manipulation layer. `LayerOverlay`
    composites a pre-defined or custom overlay image onto a frame, aligned
    and scaled to the subject's facial geometry. Four built-in overlay types
    are provided: sunglasses, glasses, a teardrop shape, and synthetic pupils.
    Custom overlay images may also be supplied via a file path.

    Parameters
    ----------
    timing_configuration : TimingConfiguration or None, optional
        A pydantic model containing timing configurations controlling onset
        and offset. If ``None``, a default ``TimingConfiguration`` is
        instantiated. The default instantiation assumes onset at 0.0 and
        offset at the video's duration.
    overlay_type : str or int, default="sunglasses"
        The overlay to apply. Accepted string values are ``"sunglasses"``,
        ``"glasses"``, ``"teardrop"``, ``"face_mask"``, and ``"pupils"``. 
        Accepted integer values are ``43`` (sunglasses), ``44`` (glasses), 
        ``45`` (teardrop), ``46`` (face_mask), and ``47`` (pupils). A file path 
        to a custom BGRA overlay image may also be provided as a string.
    overlay_scale_factor : float or None, default=None
        An optional multiplier applied to the anchor landmark distance when
        deriving the overlay scale. If ``None``, the scale factor defined
        in the overlay's configuration mapping is used.
    y_offset : int, default=10
        A vertical pixel offset applied to the overlay's computed center
        position. Positive values shift the overlay downward; negative
        values shift it upward.
    x_offset : int, default=0
        A horizontal pixel offset applied to the overlay's computed center
        position. Positive values shift the overlay to the right; negative
        values shift it to the left.
    pupil_scale_factor : float, default=0.25
        The ratio of pupil diameter to eye canthal width, used to derive
        pupil radius for the ``"pupils"`` overlay. Must lie in the
        normalised range (0.0, 1.0]. Based on adult anatomical ratios,
        typical values lie in the range [0.130, 0.300].

    Returns
    -------
    LayerOverlay
        An instance of the overlay manipulation layer.

    Raises
    ------
    ValueError
        When provided invalid, unrecognized, or out-of-range parameter values.
    """
    # Populate with defaults if None
    time_config = timing_configuration or TimingConfiguration()

    # Validate input parameters
    try:
        params = OverlayParameters(
            overlay_type=overlay_type,
            overlay_scale_factor=overlay_scale_factor, 
            y_offset = y_offset,
            x_offset = x_offset,
            pupil_scale_factor=pupil_scale_factor
        )
        
    except ValidationError as e:
        raise ValueError(f"Invalid parameters for {LayerOverlay.__name__}: {e}")
    
    return LayerOverlay(time_config, params)

__all__ = ["layer_overlay", "OverlayParameters"]