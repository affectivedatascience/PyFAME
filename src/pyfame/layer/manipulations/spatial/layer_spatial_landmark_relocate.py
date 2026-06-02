from pydantic import BaseModel, ValidationError, ValidationInfo, field_validator, PositiveInt
from typing import Optional, Tuple, Dict
from pyfame.landmark.facial_landmarks import *
from pyfame.landmark.get_landmark_coordinates import get_pixel_coordinates_from_landmark
from pyfame.layer.layer import Layer, TimingConfiguration
from pyfame.layer.manipulations.mask import mask_from_landmarks
from pyfame.layer.manipulations.spatial.face_anchors import FaceAnchor
from pyfame.utils.constants import *
import cv2 as cv
import numpy as np
from operator import itemgetter

### TODO investigate scaling inpainting radius with facial width

class LandmarkRelocateSpec(BaseModel):
    """
    Specification model defining the relocation parameters for a single
    facial landmark region.

    This class inherits from pydantic's `BaseModel` to provide validation
    and default handling of per-landmark relocation parameters. One
    ``LandmarkRelocateSpec`` is defined per landmark region, collected
    into a ``RelocateParameters.user_specs`` dictionary keyed by landmark
    index.

    Attributes
    ----------
    anchor : FaceAnchor
        The facial anchor point to which this landmark region is relocated.
        Anchors are defined in normalised face coordinates relative to the
        face center and dimensions.
    rotatation_deg : float, default=0.0
        The rotation angle in degrees applied to the landmark crop before
        it is cloned into its new position. Positive values rotate
        clockwise; negative values rotate counter-clockwise.
    offsets : tuple of float, default=(0.0, 0.0)
        Normalised (x, y) offsets applied to the anchor position, expressed
        as fractions of face width and face height respectively. Both values
        must lie in the range [-1.0, 1.0]. Positive x shifts the landmark
        rightward; positive y shifts it downward.

    Notes
    -----
    """

    anchor:FaceAnchor
    rotation_deg:float = 0.0
    offsets:Tuple[float,float] = (0.0, 0.0)  # Normalised x,y offsets

    @field_validator('offsets')
    @classmethod
    def check_normalised_offset_values(cls, value, info:ValidationInfo):
        field_name = info.field_name

        if not (-1.0 <= value[0] <= 1.0):
            raise ValidationError(f"X - {field_name} must be a normalised float in the range [-1,1].")
        if not (-1.0 <= value[1] <= 1.0):
            raise ValidationError(f"Y - {field_name} must be a normalised float in the range [-1,1].")
        
        return value

class RelocateParameters(BaseModel):
    """
    Configuration model defining the control parameters for the landmark
    relocation spatial manipulation.

    This class inherits from pydantic's `BaseModel` to provide validation
    and default handling of relocation parameters.

    Attributes
    ----------
    random_seed : int or None, optional
        An optional positive integer seed for the random number generator,
        enabling reproducible random relocation specifications across runs.
        If ``None``, a seed is sampled uniformly from [0, 1000) at
        initialisation time.
    user_specs : dict of {int : LandmarkRelocateSpec} or None, optional
        A dictionary mapping landmark keys to their relocation
        specifications. Keys are integers in the range [0, 3], corresponding
        to left eye (0), right eye (1), nose (2), and mouth (3). If ``None``,
        a random specification is generated from ``max_random_offset`` at
        layer initialisation.
    max_random_offset : float
        The maximum absolute normalised offset applied to each landmark
        when generating a random relocation specification. Must lie in the
        range [0.0, 1.0]. Ignored when ``user_specs`` is provided.
    out_greyscale : bool
        If ``True``, the output frame is converted to greyscale before
        being returned, while preserving the three-channel BGR format
        required by downstream processing.
    """

    random_seed:Optional[PositiveInt] = None
    user_specs:Optional[Dict[int, LandmarkRelocateSpec]] = None
    max_random_offset:float
    out_greyscale:bool

    @field_validator('max_random_offset')
    @classmethod
    def check_normalised_float(cls, value, info:ValidationInfo):
        field_name = info.field_name

        if not (0.0 <= value <= 1.0):
            raise ValidationError(f"{field_name} must be a normalised float in the range [0,1].")
        
        return value

class LayerSpatialLandmarkRelocate(Layer):
    """
    Manipulation layer that extracts the four primary facial feature regions
    (left eye, right eye, nose, mouth), removes them from their original
    positions using inpainting, and recomposites them at new positions on
    the face according to a relocation specification.

    Each landmark region is cut from the frame, its vacated area is filled
    using Navier-Stokes inpainting to restore a plausible skin texture, and
    the crop is rotated and seamlessly cloned onto the face at the position
    defined by a facial anchor point and normalised offsets. The result
    is optionally converted to greyscale.

    A relocation specification may be provided explicitly as a dictionary
    of ``LandmarkRelocateSpec`` objects, one per landmark region, or
    generated randomly from a configurable maximum offset magnitude. The
    random number generator is seeded once at initialisation, ensuring
    a consistent random specification is used across all frames of a
    sequence.

    Parameters
    ----------
    timing_configuration : TimingConfiguration
        Timing configuration controlling onset, offset, and rise/fall
        durations.
    relocation_parameters : RelocateParameters
        Configuration model specifying the relocation specification,
        random seed, maximum random offset, and greyscale output flag.

    Attributes
    ----------
    time_config : TimingConfiguration
        Timing configuration used by the layer.
    relocate_params : RelocateParameters
        Relocation-specific configuration parameters.
    rand_seed : int
        Seed used for the random number generator. If ``None`` was provided,
        a seed is sampled from [0, 1000) at initialisation time.
    max_random_offset : float
        Maximum absolute normalised offset used when generating a random
        relocation specification.
    out_greyscale : bool
        If ``True``, the output frame is returned in greyscale.
    pad_map : dict of {int : int}
        Per-landmark padding in pixels added to the bounding box of each
        crop. Keys correspond to landmark indices: 0 (left eye, 10px),
        1 (right eye, 10px), 2 (nose, 20px), 3 (mouth, 20px).
    rng : numpy.random.Generator
        Random number generator instance seeded with ``rand_seed``, used
        to produce the random relocation specification.

    Notes
    -----
    - This layer does not support temporal weighting; relocation is applied
      as a binary on/off effect governed solely by onset and offset times.
    - A weighted blend between the inpainted frame and a face-mean-toned
      overlay is applied before landmark cloning, to reduce colour
      discontinuities introduced by inpainting.
    - Seamless cloning via ``cv.seamlessClone`` is used for final landmark
      compositing, which may fail or produce artefacts if the destination
      center point is too close to the frame boundary.
    """

    def __init__(self, timing_configuration:TimingConfiguration, relocation_parameters:RelocateParameters):
        """
        Initialize a landmark relocation spatial manipulation layer.

        Parameters
        ----------
        timing_configuration : TimingConfiguration
            Timing configuration controlling when the relocation effect
            is applied.
        relocation_parameters : RelocateParameters
            Parameters defining the relocation specification, random seed,
            maximum random offset, and greyscale output flag.

        Notes
        -----
        - If ``random_seed`` is ``None``, a seed is sampled from [0, 1000)
          using ``numpy.random.randint`` before the seeded generator is
          instantiated, ensuring reproducibility even when no explicit seed
          is provided by the caller.
        - If ``user_specs`` is ``None``, a random relocation specification
          is generated immediately via ``_get_random_relocate_spec`` and
          stored in ``relocate_params.user_specs``.
        - A snapshot of the initial state is taken after initialization to
          allow safe resetting between independent applications.
        """
        self.time_config = timing_configuration
        self.relocate_params = relocation_parameters

        # Initialise the superclass
        super().__init__(self.time_config)       

        # Declare class parameters
        self.rand_seed = self.relocate_params.random_seed
        if self.rand_seed is None:
            self.rand_seed = np.random.randint(0,1000)
        self.max_random_offset = self.relocate_params.max_random_offset
        self.out_greyscale = self.relocate_params.out_greyscale
        self.pad_map = {
            0 : 10,
            1 : 10,
            2 : 20,
            3 : 20
        }
        # Compute random state variables once on init
        self.rng = np.random.default_rng(self.rand_seed)

        # If no user spec, generate random spec
        if self.relocate_params.user_specs is None:
            self.relocate_params.user_specs = self._get_random_relocate_spec()

        # Snapshot of initial state
        self._snapshot_state()
    
    def supports_weight(self):
        """
        Indicate whether the layer supports temporal weighting.

        Returns
        -------
        bool
            ``False``, as landmark relocation operates as a binary on/off
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
            combining both timing and relocation configuration fields.
        """
        # Dump the pydantic models to get dict of full parameter list
        self._layer_parameters = self.time_config.model_dump()
        self._layer_parameters.update(self.relocate_params.model_dump())
        self._layer_parameters["onset_time_msec"] = self.onset_t
        self._layer_parameters["offset_time_msec"] = self.offset_t
        return dict(self._layer_parameters)
    
    def _get_random_relocate_spec(self) -> Dict[int, LandmarkRelocateSpec]:
        """
        Generates a random relocation specification for each landmark key.
        Keys:
            0 - left eye
            1 - right eye
            2 - nose
            3 - mouth
        
        Returns
        -------
        Dict[int, `LandmarkRelocateSpec`]
            A dictionary mapping landmark keys to relocation specifications. Each 
            `LandmarkRelocateSpec` contains a facial anchor, a floating point rotation
            angle, and a tuple of (x,y) normalised floating point offsets in [-1,1].
        """

        specs: Dict[int, LandmarkRelocateSpec] = {}

        anchors = list(FaceAnchor)
        rand_anchors = list(self.rng.choice(anchors, 4, replace=False))

        for key in range(4):
            # select random face anchor
            anchor = rand_anchors[key]

            # random rotation logic
            rotation_angle = float(self.rng.uniform(-180.0, 180.0))

            # random offsets
            offset_x = float(self.rng.uniform(-self.max_random_offset, self.max_random_offset))
            offset_y = float(self.rng.uniform(-self.max_random_offset, self.max_random_offset))

            specs[key] = LandmarkRelocateSpec(
                anchor=anchor,
                rotation_deg=rotation_angle,
                offsets=(offset_x, offset_y)
            )
        
        return specs
    
    @staticmethod
    def _anchor_to_pixel(anchor, face_center, face_width, face_height):
        """
        Convert a normalised ``FaceAnchor`` position to absolute pixel
        coordinates in the frame.

        The anchor's normalised (x, y) value is scaled by face width and
        face height respectively and offset from the face center to produce
        the final pixel position.

        Parameters
        ----------
        anchor : FaceAnchor
            The facial anchor whose normalised position is to be converted.
        face_center : tuple of int
            The (x, y) pixel coordinates of the face center, computed from
            the bounding box of the face oval landmark.
        face_width : int
            The pixel width of the face oval bounding box.
        face_height : int
            The pixel height of the face oval bounding box.

        Returns
        -------
        px : int
            The absolute x pixel coordinate of the anchor position.
        py : int
            The absolute y pixel coordinate of the anchor position.
        """
        ax, ay = anchor.value
        px = int(face_center[0] + ax * face_width)
        py = int(face_center[1] + ay * face_height)
        return px, py
    
    def apply_layer(self, landmarker_coordinates:list[tuple[int,int]], frame:cv.typing.MatLike, dt:float) -> cv.typing.MatLike:
        """
        Apply the landmark relocation manipulation to a single frame.

        Each of the four landmark regions (left eye, right eye, nose, mouth)
        is extracted as a padded crop, replaced in the output frame using
        Navier-Stokes inpainting, which also has a slight Gaussian-blur applied
        to soften artefacts. After all four regions have been
        removed, a weighted blend with a face-mean-toned overlay is applied
        to reduce colour discontinuities. Each landmark crop is then rotated
        according to its relocation specification and seamlessly cloned onto
        the face at the position defined by its anchor and normalised offsets.

        Parameters
        ----------
        landmarker_coordinates : list of tuple of int
            Facial landmark coordinates associated with the current frame.
        frame : MatLike
            Input image frame to which the landmark relocation is applied.
        dt : float
            Current time (in milliseconds).

        Returns
        -------
        MatLike
            The frame with all four landmark regions removed from their
            original positions and recomposited at their specified new
            positions. If ``out_greyscale`` is ``True``, the output is
            returned as a three-channel greyscale image.

        Notes
        -----
        - The inpainting radius of 15 pixels is currently hardcoded and may be
          insufficient for larger landmark regions at high resolutions. A future
          addition is planned to add dynamic radius computation.
        - ``cv.seamlessClone`` may produce artefacts or raise an error if
          the destination center point ``(cx, cy)`` lies within approximately
          half the clone patch width of the frame boundary.
        """
        if dt is None:
            weight = 1.0
        else:
            weight = super().compute_weight(dt, self.supports_weight())

        if weight == 0.0:
            return frame

        # Get the pixel coordinates of various landmark regions
        landmark_nose_wide = create_landmark_path(NOSE_WIDE_IDX)
        le_screen_coords = get_pixel_coordinates_from_landmark(landmarker_coordinates, LANDMARK_LEFT_EYE_REGION)
        re_screen_coords = get_pixel_coordinates_from_landmark(landmarker_coordinates, LANDMARK_RIGHT_EYE_REGION)
        nose_screen_coords = get_pixel_coordinates_from_landmark(landmarker_coordinates, landmark_nose_wide)
        lips_screen_coords = get_pixel_coordinates_from_landmark(landmarker_coordinates, LANDMARK_MOUTH_REGION)
        fo_screen_coords = get_pixel_coordinates_from_landmark(landmarker_coordinates, LANDMARK_FACE_OVAL)

        # Creating boolean masks of each landmark region
        le_mask = mask_from_landmarks(frame, LANDMARK_LEFT_EYE_REGION, landmarker_coordinates)
        re_mask = mask_from_landmarks(frame, LANDMARK_RIGHT_EYE_REGION, landmarker_coordinates)
        nose_mask = mask_from_landmarks(frame, landmark_nose_wide, landmarker_coordinates)
        lip_mask = mask_from_landmarks(frame, LANDMARK_MOUTH_REGION, landmarker_coordinates)
        fo_mask = mask_from_landmarks(frame, LANDMARK_FACE_OVAL, landmarker_coordinates)
        face_skin_mask = mask_from_landmarks(frame, LANDMARK_FACE_SKIN, landmarker_coordinates)

        # Compute face reference frame for anchors
        max_x_fo = max(fo_screen_coords, key=itemgetter(0))[0]
        min_x_fo = min(fo_screen_coords, key=itemgetter(0))[0]
        max_y_fo = max(fo_screen_coords, key=itemgetter(1))[1]
        min_y_fo = min(fo_screen_coords, key=itemgetter(1))[1]

        face_width = max_x_fo - min_x_fo
        face_height = max_y_fo - min_y_fo

        face_center = (
            (min_x_fo + max_x_fo) // 2,
            (min_y_fo + max_y_fo) // 2
        )

        masks = [le_mask, re_mask, nose_mask, lip_mask]
        screen_coords = [le_screen_coords, re_screen_coords, nose_screen_coords, lips_screen_coords]
        landmarks = {
            0 : {},
            1 : {},
            2 : {},
            3 : {}
        }
        output_frame = frame.copy()

        for i, (mask, coords) in enumerate(zip(masks, screen_coords)):
            # --------- Cutting out and storing landmarks ----------
            # min and max coords + padding
            max_x = min(output_frame.shape[1], max(coords, key=itemgetter(0))[0] + self.pad_map[i])
            min_x = max(0, min(coords, key=itemgetter(0))[0] - self.pad_map[i])
            max_y = min(output_frame.shape[0], max(coords, key=itemgetter(1))[1] + self.pad_map[i])
            min_y = max(0, min(coords, key=itemgetter(1))[1] - self.pad_map[i])

            # Crop the frame to the bounds of the landmark
            cropped_lm = output_frame[min_y:max_y, min_x:max_x]
            cropped_mask = mask[min_y:max_y, min_x:max_x]

            # Cut out the current landmark region and store it
            lm = cv.bitwise_and(src1=cropped_lm, src2=cropped_lm, mask=cropped_mask)
            # Compute the localised landmark center coords
            cx = (cropped_lm.shape[1]) // 2
            cy = (cropped_lm.shape[0]) // 2

            landmarks[i].update({'img' : lm, 'mask' : cropped_mask, 'center' : (cx,cy)})

            # ---------- Filling in cut out regions of the original frame ----------
            # fill the original lm region in the output frame with empty pixels 
            output_frame = cv.bitwise_and(src1=output_frame, src2=output_frame, mask=cv.bitwise_not(mask))

            # Fill landmark holes with navier-stokes inpainting;
            # uses nearest-neighbor colour sampling to fill in gaps in an image
            output_frame = cv.inpaint(output_frame, mask, 15, cv.INPAINT_NS)

            # dilate the landmark mask slightly, and blur around inpainted edges
            kernel = np.ones((5,5), np.uint8)
            dilated_mask = cv.dilate(mask, kernel, iterations=1)
            dilated_mask = dilated_mask[..., np.newaxis]

            # blur the output frame around the inpainted landmarks
            face_only = cv.bitwise_and(output_frame, output_frame, mask=fo_mask)
            face_only = cv.GaussianBlur(face_only, (15,15), sigmaX=10)

            output_frame = np.where(dilated_mask == 255, face_only, output_frame)
        
        # Perform a weighted addition between the facial mean tone and the inpainted image
        facial_mean = cv.mean(frame, mask=face_skin_mask)[:3]
        fo_only = np.where(fo_mask[..., np.newaxis] == 255, facial_mean, frame).astype(np.uint8)
        output_frame = cv.addWeighted(output_frame, 0.6, fo_only, 0.4, 0)
        
        # Cloning rotated landmarks into their new positions on the underlying face
        for key in list(landmarks.keys()):
            spec = self.relocate_params.user_specs.get(key)
            lm = landmarks.get(key)
            landmark = lm['img']
            mask = lm['mask']
            local_center = lm['center']
            h,w = landmark.shape[:2]

            # Get pixel coordinates from facial anchor
            cx, cy = self._anchor_to_pixel(spec.anchor, face_center, face_width, face_height)

            # Adding normalised offsets
            cx += int(spec.offsets[0] * face_width)
            cy += int(spec.offsets[1] * face_height)

            rot_mat = cv.getRotationMatrix2D(center=local_center, angle=spec.rotation_deg, scale=1)

            # rotate landmark and mask
            landmark = cv.warpAffine(landmark, rot_mat, (w,h))
            mask = cv.warpAffine(mask, rot_mat, (w,h))
            
            # Clone the landmark onto the original face in its new position
            output_frame = cv.seamlessClone(landmark, output_frame, mask, (cx, cy), cv.NORMAL_CLONE)

        if self.out_greyscale:
            gray = cv.cvtColor(output_frame, cv.COLOR_BGR2GRAY)
            # Convert 2-D gray image back to 3-D colour image while maintaining greyscale visual
            return cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
        else:
            return output_frame
        
def layer_spatial_landmark_relocate(timing_configuration:TimingConfiguration | None = None, landmark_relocate_specs:dict[int, LandmarkRelocateSpec] | None = None,
                                    random_seed:int | None = None, max_random_offset:float = 0.15, out_greyscale:bool = True) -> LayerSpatialLandmarkRelocate:
    """
    Factory function for the landmark relocation spatial manipulation layer.
    `LayerSpatialLandmarkRelocate` extracts the four primary facial feature
    regions (left eye, right eye, nose, and mouth), removes them from their
    original positions using Navier-Stokes inpainting, and recomposites them
    at new positions on the face defined by a relocation specification. Each
    landmark may be assigned an independent anchor point, normalised positional
    offset, and rotation angle.

    A relocation specification may be provided explicitly or generated
    randomly from a configurable maximum offset magnitude. When generated
    randomly, the specification is fixed at layer initialisation and applied
    consistently across all frames of the sequence.

    Parameters
    ----------
    timing_configuration : TimingConfiguration or None, optional
        A pydantic model containing timing configurations controlling onset
        and offset. If ``None``, a default ``TimingConfiguration`` is
        instantiated. The default instantiation assumes onset at 0.0 and
        offset at the video's duration.
    landmark_relocate_specs : dict of {int : LandmarkRelocateSpec} or None, default=None
        A dictionary mapping landmark keys to their relocation specifications.
        Keys are integers in the range [0, 3], corresponding to left eye (0),
        right eye (1), nose (2), and mouth (3). If ``None``, a random
        specification is generated from ``max_random_offset``.
    random_seed : int or None, default=None
        An optional positive integer seed for the random number generator,
        enabling reproducible random relocation specifications. If ``None``,
        a seed is sampled from [0, 1000) at layer initialisation.
    max_random_offset : float, default=0.15
        The maximum absolute normalised offset applied to each landmark
        when generating a random relocation specification. Must lie in the
        range [0.0, 1.0]. Ignored when ``landmark_relocate_spec`` is provided.
    out_greyscale : bool, default=True
        If ``True``, the output frame is converted to greyscale before being
        returned, while preserving the three-channel BGR format required by
        downstream processing.

    Returns
    -------
    LayerSpatialLandmarkRelocate
        An instance of the landmark relocation spatial manipulation layer.

    Raises
    ------
    ValueError
        When provided invalid or out-of-range parameter values.
    """
    # Populate with defaults if None
    time_config = timing_configuration or TimingConfiguration()

    # Validate input parameters
    try:
        params = RelocateParameters(
            random_seed=random_seed,
            user_specs=landmark_relocate_specs,
            max_random_offset=max_random_offset,
            out_greyscale=out_greyscale
        )
    except ValidationError as e:
        raise ValueError(f"Invalid parameters for {LayerSpatialLandmarkRelocate.__name__}: {e}")
    
    return LayerSpatialLandmarkRelocate(time_config, params)

__all__ = ["layer_spatial_landmark_relocate", "RelocateParameters"]