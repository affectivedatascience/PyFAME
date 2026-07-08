from pydantic import BaseModel, field_validator, ValidationInfo, ValidationError, PositiveInt, NonNegativeFloat
from typing import Union, Optional, List, Tuple
from pyfame.landmark.facial_landmarks import *
from pyfame.landmark.get_landmark_coordinates import get_relative_landmark_coordinates
from pyfame.layer.layer import Layer, TimingConfiguration
from pyfame.layer.manipulations.mask import mask_from_landmarks
from pyfame.utils.constants import * 
from pyfame.utils.general_utilities import get_landmark_names, compute_slope, compute_rotation_angle
import cv2 as cv
import numpy as np
from operator import itemgetter

class GridShuffleParameters(BaseModel):
    """
    Configuration model defining the control parameters for applying
    a grid-based spatial shuffle to a frame or image.

    This class inherits from pydantic's `BaseModel` to provide validation
    and default handling of grid shuffle parameters.

    Attributes
    ----------
    random_seed : int or None
        An optional seed for the random number generator, enabling
        reproducible shuffle permutations across runs. If ``None``, the
        generator is seeded non-deterministically.
    shuffle_method : str or int
        The shuffling algorithm to apply. Accepted string values are
        ``"random"``, ``"cyclic shift"`` and ``"none"``. Accepted integer values are
        ``27`` (random), ``28`` (cyclic shift) and ``48`` (none). String inputs are
        normalised to lowercase on validation.
    grid_square_size : int
        The baseline side length in pixels of each grid square at the
        reference face width. Must be a positive integer. The effective
        square size is scaled per-frame to follow changes in face size.
    grid_dimensions : tuple of int
        An override to grid_square_size, infers the size of the individual
        cells to fit into the specified grid dimensions. Must be a tuple 
        of positive integers. The effective square size is scaled per-frame
        to follow changes in face size.
    mask_overlap_threshold : float
        The minimum fraction of a grid square's pixels that must lie
        within the landmark mask for that square to be considered active
        and included in the shuffle. Must lie in the range (0.0, 1.0].
    cyclic_shift_amount : int
        The number of positions by which active grid square indices are
        rotated in the ``"cyclic shift"`` shuffle method. Must be a
        positive integer. Ignored by other shuffle methods.
    landmark_paths : list of list of tuple of int or list of tuple of int
        A list of one or more closed landmark paths representing the
        region in which the grid shuffle will be applied.
    """

    random_seed:Optional[int]
    shuffle_method:Union[int,str]
    grid_square_size:PositiveInt
    grid_dimensions:Optional[tuple[int,int]]
    mask_overlap_threshold:NonNegativeFloat
    cyclic_shift_amount:PositiveInt
    landmark_paths:Union[List[List[Tuple[int,...]]], List[Tuple[int,...]]]

    @field_validator("shuffle_method", mode="before")
    @classmethod
    def check_accepted_value(cls, value, info:ValidationInfo):
        field_name = info.field_name
        shuffle_method_map = {27:"random", 28:"cyclic-shift", 48:"none"}

        if isinstance(value, str):
            value = str.lower(value)
            if value not in {"random", "cyclic shift", "none"}:
                raise ValueError(f"Unrecognized value for parameter {field_name}.")
            return value
        
        elif isinstance(value, int):
            if value not in {27, 28, 48}:
                raise ValueError(f"Unrecognized value for parameter {field_name}.")
            return shuffle_method_map.get(value)

        raise TypeError(f"Invalid type for parameter {field_name}, Must be one of int or str.")
    
    @field_validator("mask_overlap_threshold")
    @classmethod
    def check_normal_range(cls, value, info:ValidationInfo):
        field_name = info.field_name

        if not (0.0 < value <= 1.0):
            raise ValueError(f"{field_name} must be a float in the range (0, 1].")
        
        return value

class LayerSpatialGridShuffle(Layer):
    """
    Manipulation layer that spatially rearranges a landmark-defined facial
    region by partitioning it into a uniform grid and shuffling the grid
    squares according to a configured permutation method.

    The face region is partitioned into a grid of equal-sized squares whose
    side length scales proportionally with face width across frames, ensuring
    consistent spatial granularity regardless of subject distance. Only grid
    squares with sufficient overlap with the landmark mask are included in
    the shuffle; the remaining squares are left unmodified.

    Two shuffle methods are supported: random permutation, which reassigns
    squares to arbitrary positions within the active set; and cyclic shift,
    which rotates the active square positions by a fixed number of steps.

    The shuffled grid is rotated to follow the estimated head roll angle and
    composited back onto the original frame, confined to the landmark mask
    region.

    Grid geometry, active square indices, and the shuffle permutation are
    computed on the first call to ``apply_layer`` and cached for all
    subsequent frames, ensuring temporal consistency of the shuffle pattern
    across the video sequence.

    Parameters
    ----------
    timing_configuration : TimingConfiguration
        Timing configuration controlling onset, offset, and rise/fall
        durations.
    shuffle_parameters : GridShuffleParameters
        Configuration model specifying the shuffle method, grid square size,
        overlap threshold, cyclic shift amount, landmark region(s), and
        optional random seed.

    Attributes
    ----------
    time_config : TimingConfiguration
        Timing configuration used by the layer.
    shuffle_params : GridShuffleParameters
        Grid shuffle-specific configuration parameters.
    rand_seed : int or None
        Seed for the random number generator. If ``None``, permutations
        are non-deterministic across runs.
    shuffle_method : str or int
        The shuffling algorithm to apply (``"random"``, ``"cyclic shift"`` or ``"none"``).
    grid_square_size : int
        Baseline side length in pixels of each grid square at the reference
        face width.
    grid_dimensions : tuple of int
        An override to grid_square_size, infers the size of the individual
        cells to fit into the specified grid dimensions.
    overlap_threshold : float
        Minimum fraction of a grid square that must overlap the landmark
        mask for it to be included in the shuffle.
    shift_amount : int
        Number of positions by which active indices are rotated in the
        ``"cyclic shift"`` method.
    landmark_paths : list of list of tuple of int or list of tuple of int
        Landmark paths defining the region in which the shuffle is applied.
    baseline_face_width : int or None
        Face width at the first processed frame, used as the reference scale
        for per-frame grid square size adjustment. ``None`` until first call.
    baseline_padded_width : int or None
        Padded grid width in pixels at the reference scale. ``None`` until
        first call.
    baseline_padded_height : int or None
        Padded grid height in pixels at the reference scale. ``None`` until
        first call.
    baseline_x_pad : int or None
        Horizontal padding added to align the grid width to a multiple of
        ``grid_square_size``. ``None`` until first call.
    baseline_y_pad : int or None
        Vertical padding added to align the grid height to a multiple of
        ``grid_square_size``. ``None`` until first call.
    baseline_cols : int or None
        Number of grid columns at the reference scale. Fixed after first
        call and reused across all subsequent frames. ``None`` until first
        call.
    baseline_rows : int or None
        Number of grid rows at the reference scale. Fixed after first call
        and reused across all subsequent frames. ``None`` until first call.
    baseline_active_indices : list of int or None
        Indices of grid squares determined to be active at the reference
        frame. Cached after first call. ``None`` until first call.
    baseline_active_permutation : list of int or None
        The shuffle permutation of active grid square indices, computed
        once at the reference frame and reused across all subsequent frames.
        ``None`` until first call.

    Notes
    -----
    - This layer does not support temporal weighting; the shuffle is applied
      as a binary on/off effect governed solely by onset and offset times.
    - Grid geometry, active square selection, and the shuffle permutation
      are computed once on the first call and cached, ensuring a temporally
      stable shuffle pattern. This means changes in face geometry across
      frames are accommodated only in square size, not in grid topology.
    - Head roll is estimated from landmarks 162 and 389, consistent with
      other rotation-aware layers in PyFAME.
    - When ``landmark_paths`` is ``LANDMARK_FACE_OVAL``, the shuffled output
      is additionally masked to the face oval region to prevent shuffled
      content from bleeding outside the face boundary.
    """

    def __init__(self, timing_configuration:TimingConfiguration, shuffle_parameters:GridShuffleParameters):
        """
        Initialize a grid shuffle spatial manipulation layer.

        Parameters
        ----------
        timing_configuration : TimingConfiguration
            Timing configuration controlling when the grid shuffle effect
            is applied.
        shuffle_parameters : GridShuffleParameters
            Parameters defining the shuffle method, grid square size,
            overlap threshold, cyclic shift amount, target landmark
            region(s), and optional random seed.

        Notes
        -----
        - All baseline geometry attributes are initialised to ``None`` and
          populated lazily on the first call to ``apply_layer``.
        - A snapshot of the initial state is taken after initialization to
          allow safe resetting between independent applications.
        """
        self.time_config = timing_configuration
        self.shuffle_params = shuffle_parameters

        # Initialise superclass
        super().__init__(self.time_config)

        # Declare class parameters
        self.rand_seed = self.shuffle_params.random_seed
        self.shuffle_method = self.shuffle_params.shuffle_method
        self.grid_square_size = self.shuffle_params.grid_square_size
        self.grid_dimensions = self.shuffle_params.grid_dimensions
        self.overlap_threshold = self.shuffle_params.mask_overlap_threshold
        self.shift_amount = self.shuffle_params.cyclic_shift_amount
        self.landmark_paths = self.shuffle_params.landmark_paths
        self.baseline_face_width = None
        self.baseline_padded_width = None
        self.baseline_padded_height = None
        self.baseline_x_pad = None
        self.baseline_y_pad = None
        self.baseline_cols = None
        self.baseline_rows = None
        self.baseline_active_indices = None
        self.baseline_active_permutation = None

        # Snapshot of initial state
        self._snapshot_state()

    def supports_weight(self):
        """
        Indicate whether the layer supports temporal weighting.

        Returns
        -------
        bool
            ``False``, as grid shuffle operates as a binary on/off effect
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
            combining both timing and grid shuffle configuration fields.
        """
        # Dump the pydantic models to get dict of full parameter list
        self._layer_parameters = self.time_config.model_dump()
        self._layer_parameters.update(self.shuffle_params.model_dump())
        self._layer_parameters["onset_time_msec"] = self.onset_t
        self._layer_parameters["offset_time_msec"] = self.offset_t
        return dict(self._layer_parameters)
    
    def get_active_square_indices(self, keys:List[Tuple[int,int]], region_mask:cv.typing.MatLike, square_size:int, 
                                  overlap_threshold:float) -> Tuple[List[int], List[int]]:
        """
        Partition grid square indices into active and passive sets based on
        their overlap with the landmark region mask.

        For each grid square, the fraction of its pixels that fall within
        the masked region is computed. Squares meeting or exceeding the
        overlap threshold are classified as active and included in the
        shuffle; the remainder are classified as passive and left in place.

        Parameters
        ----------
        keys : list of tuple of int
            Ordered list of ``(x, y)`` top-left corner coordinates for
            each grid square, in row-major order.
        region_mask : MatLike
            Binary mask of the landmark region of interest, where 255
            denotes pixels inside the region.
        square_size : int
            Side length in pixels of each grid square for the current frame.
        overlap_threshold : float
            Minimum fraction of a grid square's pixels that must lie within
            the masked region for it to be classified as active.

        Returns
        -------
        active_indices : list of int
            Indices into `keys` of grid squares with sufficient mask overlap.
        passive_indices : list of int
            Indices into `keys` of grid squares below the overlap threshold
            or with degenerate bounds.
        """

        active_indices = []
        passive_indices = []

        for i, (x,y) in enumerate(keys):
            # Clamp to frame bounds before slicing
            x1, y1 = x + square_size, y + square_size

            if x1 <= x or y1 <= y:
                passive_indices.append(i)
                continue

            masked_grid_square = region_mask[y:y1, x:x1]
            total_pixels = (x1 - x) * (y1 - y)
            inside_pixels = int(np.count_nonzero(masked_grid_square))

            if inside_pixels / total_pixels >= overlap_threshold:
                active_indices.append(i)
            else:
                passive_indices.append(i)
        
        return active_indices, passive_indices
    
    def shuffle_indices(self, active_indices:List[int]) -> List[int]:
        """
        Generate a shuffled permutation of the active grid square indices
        according to the configured shuffle method.

        For ``"random"`` shuffle, the active indices are randomly permuted
        using a ``numpy.random.Generator`` seeded with ``rand_seed``. For
        ``"cyclic shift"``, the active indices are rotated by
        ``shift_amount`` positions using ``numpy.roll``.

        Parameters
        ----------
        active_indices : list of int
            Ordered list of indices identifying the active grid squares
            to be shuffled.

        Returns
        -------
        list of int
            A permuted list of the same active indices, defining the
            mapping from source to destination positions in the shuffle.
        """
        rng = np.random.default_rng(self.rand_seed)
        permutation = None

        # Fully random shuffle
        if self.shuffle_method == "random":
            permutation = np.array(active_indices)
            rng.shuffle(permutation)
        
        elif self.shuffle_method == "cyclic shift":
            # Rotate the index list by shift_amount positions.
            permutation = np.roll(active_indices, self.shift_amount)
        
        else:
            return active_indices

        return permutation.tolist()
    
    def apply_layer(self, landmarker_coordinates:list[tuple[int,int]], frame:cv.typing.MatLike, dt:float) -> cv.typing.MatLike:
        """
        Apply the grid shuffle spatial manipulation to a single frame.

        The face region is partitioned into a grid of equal-sized squares
        scaled to the current face width. Active squares are identified by
        their overlap with the landmark mask and rearranged according to
        the cached shuffle permutation. The shuffled grid is rotated to
        follow the estimated head roll angle, tightly cropped, and
        composited back onto the original frame centered on the landmark
        region centroid.

        On the first call, grid geometry, active square indices, and the
        shuffle permutation are computed and cached for reuse across all
        subsequent frames. Per-frame scaling ensures that the effective
        grid square size tracks changes in face width while the grid
        topology and shuffle pattern remain fixed.

        Parameters
        ----------
        landmarker_coordinates : list of tuple of int
            Facial landmark coordinates associated with the current frame.
        frame : MatLike
            Input image frame to which the grid shuffle is applied.
        dt : float
            Current time (in milliseconds).

        Returns
        -------
        MatLike
            The frame with the landmark-defined region spatially shuffled,
            and original pixel values preserved outside it.

        Raises
        ------
        RuntimeError
            If the landmark mask, shuffled grid mask, or rotated grid mask
            contains no non-zero pixels, indicating an invalid or empty
            masked region.

        Notes
        -----
        - Grid squares that extend partially outside the frame boundary are
          handled by clamping source and destination slice indices, filling
          out-of-bounds portions with black pixels.
        - The shuffled grid is rotated on a square canvas sized to its
          diagonal to prevent corner clipping, consistent with the approach
          used in other rotation-aware layers in PyFAME.
        - When ``landmark_paths`` is ``LANDMARK_FACE_OVAL``, the final
          output is masked to the face oval boundary to prevent shuffled
          content from bleeding into the background.
        """
        if dt is None:
            weight = 1.0
        else:
            weight = super().compute_weight(dt, self.supports_weight())

        if weight == 0.0:
            return frame

        fo_screen_coords = get_relative_landmark_coordinates(landmarker_coordinates, LANDMARK_FACE_OVAL)
        output_frame = np.zeros((frame.shape[0], frame.shape[1], frame.shape[2]), dtype=np.uint8)

        # Compute centroid of masked region for later grid overlay
        region_mask = mask_from_landmarks(frame, self.landmark_paths, landmarker_coordinates)
        region_pixels = cv.findNonZero(region_mask)

        if region_pixels is None:
            raise RuntimeError("Invalid image mask: no masked region present.")
        rx, ry, rw, rh = cv.boundingRect(region_pixels)
        landmark_cx = rx + rw // 2
        landmark_cy = ry + rh // 2

        # Get x and y bounds of the face oval
        max_x = max(fo_screen_coords, key=itemgetter(0))[0]
        min_x = min(fo_screen_coords, key=itemgetter(0))[0]

        max_y = max(fo_screen_coords, key=itemgetter(1))[1]
        min_y = min(fo_screen_coords, key=itemgetter(1))[1]

        face_height = max_y-min_y
        face_width = max_x-min_x
        face_center_x = round((max_x + min_x)/2)
        face_center_y = round((max_y + min_y)/2)

        if self.baseline_face_width is None:
            self.baseline_face_width = face_width

        # Compute padding and add equally to both sides
        if self.baseline_x_pad is None or self.baseline_y_pad is None:
            if self.grid_dimensions is not None:
                # Dimensions override: derive square size from specified cols/rows
                self.baseline_cols, self.baseline_rows = self.grid_dimensions
                square_size_from_cols = face_width  // self.baseline_cols
                square_size_from_rows = face_height // self.baseline_rows

                # Use the smaller of the two to ensure squares fit in both axes
                derived_square_size = min(square_size_from_cols, square_size_from_rows)
                self.grid_square_size = derived_square_size

                # Recompute padding with the derived square size
                self.baseline_x_pad = self.grid_square_size - (face_width  % self.grid_square_size)
                self.baseline_y_pad = self.grid_square_size - (face_height % self.grid_square_size)
            else:
                self.baseline_x_pad = self.grid_square_size - (face_width % self.grid_square_size)
                self.baseline_y_pad = self.grid_square_size - (face_height % self.grid_square_size)

                if self.baseline_x_pad % 2 !=0:
                    min_x -= int(np.floor(self.baseline_x_pad/2))
                    max_x += int(np.ceil(self.baseline_x_pad/2))
                else:
                    min_x -= int(self.baseline_x_pad/2)
                    max_x += int(self.baseline_x_pad/2)
                
                if self.baseline_y_pad % 2 !=0:
                    min_y -= int(np.floor(self.baseline_y_pad/2))
                    max_y += int(np.ceil(self.baseline_y_pad/2))
                else:
                    min_y -= int(self.baseline_y_pad/2)
                    max_y += int(self.baseline_y_pad/2)

                self.baseline_padded_width = max_x - min_x
                self.baseline_padded_height = max_y - min_y

                # Fix the number of cols throughout processing
                self.baseline_cols = int(self.baseline_padded_width / self.grid_square_size)
                self.baseline_rows = int(self.baseline_padded_height / self.grid_square_size)
        
        scale_factor = face_width / self.baseline_face_width
        square_size = int(np.ceil(scale_factor * self.grid_square_size))
        padded_width = square_size * self.baseline_cols
        padded_height = square_size * self.baseline_rows

        min_x = face_center_x - (padded_width // 2)
        min_y = face_center_y - (padded_height // 2)

        grid_squares = {}

        # Populate the grid_squares dict with segments of the frame
        for row in range(self.baseline_rows):
            for col in range(self.baseline_cols):
                x = min_x + col * square_size
                y = min_y + row * square_size

                # Full-size blank square, will be filled in with in-bounds pixels
                square = np.zeros((square_size, square_size, frame.shape[2]), dtype=np.uint8)

                # Clamp bounds to frame dimensionss
                src_x0 = max(x, 0)
                src_y0 = max(y, 0)
                src_x1 = min(x + square_size, frame.shape[1])
                src_y1 = min(y + square_size, frame.shape[0])

                if src_x1 > src_x0 and src_y1 > src_y0:
                    # Adjusting the square slice if square is outside of frame dims
                    dst_x0 = src_x0 - x
                    dst_y0 = src_y0 - y
                    dst_x1 = dst_x0 + (src_x1 - src_x0)
                    dst_y1 = dst_y0 + (src_y1 - src_y0)

                    square[dst_y0:dst_y1, dst_x0:dst_x1] = frame[src_y0:src_y1, src_x0:src_x1]
                # Grid square id's in the coordinate space
                grid_squares[(x,y)] = square
        
        keys = list(grid_squares.keys())

        # Region based filtering
        if self.baseline_active_permutation is None:
            active_indices, _ = self.get_active_square_indices(keys, region_mask, square_size, self.overlap_threshold)
            self.baseline_active_indices = active_indices

            shuffled_active_indices = self.shuffle_indices(active_indices.copy())
            self.baseline_active_permutation = shuffled_active_indices
        
        shuffled_grid_squares = {}
        # Build shuffled grid mapping
        for key0, key1 in zip(self.baseline_active_indices, self.baseline_active_permutation):
            shuffled_grid_squares.update(
                {
                    keys[key0] : grid_squares.get(keys[key1])
                }
            )

        # Fill the output frame with shuffled grid segments
        for (x,y), square in shuffled_grid_squares.items():

            # Clamp bounds to frame dimensionss
            dest_x0 = max(0,x)
            dest_y0 = max(0,y)
            dest_x1 = min(frame.shape[1], x + square_size)
            dest_y1 = min(frame.shape[0], y + square_size)

            if dest_x1 <= dest_x0 or dest_y1 <= dest_y0:
                # Square is entirely outside the frame
                continue

            # Handling asymetric square slices
            src_x0 = dest_x0 - x
            src_y0 = dest_y0 - y
            src_x1 = src_x0 + (dest_x1 - dest_x0)
            src_y1 = src_y0 + (dest_y1 - dest_y0)

            output_frame[dest_y0:dest_y1, dest_x0:dest_x1] = square[src_y0:src_y1, src_x0:src_x1]

        # Compute rotation angle from face orientation landmarks
        p1 = landmarker_coordinates[162]
        p2 = landmarker_coordinates[389]
        slope = compute_slope(p1, p2)
        rot_angle = compute_rotation_angle(slope_1=slope)

        # Thresholding to create mask of the foreground
        grey = cv.cvtColor(output_frame, cv.COLOR_BGR2GRAY)
        _, thresholded = cv.threshold(grey, 0, 255, cv.THRESH_BINARY)
        # Morphological close to fill in small gaps in the mask
        kernel = np.ones((3,3), np.uint8)
        grid_mask = cv.morphologyEx(thresholded, cv.MORPH_CLOSE, kernel)

        grid_pixels = cv.findNonZero(grid_mask)
        if grid_pixels is None:
            raise RuntimeError("Invalid image mask: no masked region present.")
        
        gx, gy, gw, gh = cv.boundingRect(grid_pixels)
        grid_crop = output_frame[gy:gy+gh, gx:gx+gw]
        grid_mask_crop = grid_mask[gy:gy+gh, gx:gx+gw]

        # Build a square frame large enough to hold the crop at any rotation angle
        diagonal = int(np.ceil(np.sqrt(gw**2 + gh**2)))
        # Pad to even size
        diag_size = diagonal + (diagonal % 2)

        temp_padded_grid = np.zeros((diag_size, diag_size, frame.shape[2]), dtype=np.uint8)
        temp_padded_mask = np.zeros((diag_size, diag_size), dtype=np.uint8)

        # Place the cropped grid and mask in the center of the padded frames
        off_x = (diag_size - gw) // 2
        off_y = (diag_size - gh) // 2
        temp_padded_grid[off_y:off_y+gh, off_x:off_x+gw] = grid_crop
        temp_padded_mask[off_y:off_y+gh, off_x:off_x+gw] = grid_mask_crop

        temp_padded_cx = diag_size / 2
        temp_padded_cy = diag_size / 2
        rot_mat = cv.getRotationMatrix2D((temp_padded_cx, temp_padded_cy), rot_angle, 1)
        rotated_padded_grid = cv.warpAffine(temp_padded_grid, rot_mat, (diag_size, diag_size))
        rotated_padded_mask = cv.warpAffine(temp_padded_mask, rot_mat, (diag_size, diag_size), flags=cv.INTER_NEAREST)

        # Tightly re crop around rotated contents
        rotated_pixels = cv.findNonZero(rotated_padded_mask)
        if rotated_pixels is None:
            raise RuntimeError("Invalid image mask: no masked region present.")

        rx2, ry2, rw2, rh2 = cv.boundingRect(rotated_pixels)
        rotated_grid_crop = rotated_padded_grid[ry2:ry2+rh2, rx2:rx2+rw2]
        rotated_mask_crop = rotated_padded_mask[ry2:ry2+rh2, rx2:rx2+rw2]

        # Overlay the rotated grid over the landmark centroid
        dst_x = landmark_cx - rw2 // 2
        dst_y = landmark_cy - rh2 // 2

        # Clamp destination to frame bounds
        dst_x0 = max(dst_x, 0)
        dst_y0 = max(dst_y, 0)
        dst_x1 = min(dst_x + rw2, frame.shape[1])
        dst_y1 = min(dst_y + rh2, frame.shape[0])

        # Source slice into the cropped grid
        src_x0 = dst_x0 - dst_x
        src_y0 = dst_y0 - dst_y
        src_x1 = src_x0 + (dst_x1 - dst_x0)
        src_y1 = src_y0 + (dst_y1 - dst_y0)

        final_output = frame.copy()
        roi_mask = rotated_mask_crop[src_y0:src_y1, src_x0:src_x1]
        final_output[dst_y0:dst_y1, dst_x0:dst_x1][roi_mask == 255] = \
            rotated_grid_crop[src_y0:src_y1, src_x0:src_x1][roi_mask == 255]

        if get_landmark_names(self.landmark_paths) == "LANDMARK_FACE_OVAL":
            final_output = np.where(region_mask[:,:,np.newaxis] == 255, final_output, frame)

        return final_output
        
def layer_spatial_grid_shuffle(timing_configuration:TimingConfiguration | None = None, random_seed:int|None = None, shuffle_method:int|str = "random", 
                               grid_square_size:int = 40, grid_dimensions:tuple[int,int] | None = None, mask_overlap_threshold:float = 0.5, 
                               cyclic_shift_amount:int = 1, landmark_paths:List[List[Tuple[int,int]]] | List[Tuple[int,int]] = LANDMARK_FACE_OVAL) -> LayerSpatialGridShuffle:
    """
    Factory function for the grid shuffle spatial manipulation layer.
    `LayerSpatialGridShuffle` partitions a landmark-defined facial region
    into a uniform grid and rearranges its squares according to a configured
    permutation method, disrupting the spatial layout of facial features
    while preserving local appearance within each grid square.

    Grid geometry and the shuffle permutation are computed once from the
    first processed frame and remain fixed for the duration of processing,
    ensuring temporal consistency of the shuffle pattern across video
    sequences. The effective grid square size scales proportionally with
    face width per frame to accommodate changes in subject distance.

    Parameters
    ----------
    timing_configuration : TimingConfiguration or None, optional
        A pydantic model containing timing configurations controlling onset
        and offset. If ``None``, a default ``TimingConfiguration`` is
        instantiated. The default instantiation assumes onset at 0.0 and
        offset at the video's duration.
    random_seed : int or None, default=None
        An optional seed for the random number generator, enabling
        reproducible shuffle permutations across runs. If ``None``,
        permutations are non-deterministic.
    shuffle_method : str or int, default="random"
        The shuffling algorithm to apply. Accepted string values are
        ``"random"``, ``"cyclic shift"`` and ``"none"``. Accepted integer values are
        ``27`` (random), ``28`` (cyclic shift) and ``48`` (none).
    grid_square_size : int, default=40
        The baseline side length in pixels of each grid square at the
        reference face width. Must be a positive integer.
    grid_dimensions : tuple of int
        An override to grid_square_size, infers the size of the individual
        cells to fit into the specified grid dimensions. Must be a tuple 
        of positive integers. The effective square size is scaled per-frame
        to follow changes in face size.
    mask_overlap_threshold : float, default=0.5
        The minimum fraction of a grid square's pixels that must lie within
        the landmark mask for it to be included in the shuffle. Must lie in
        the range (0.0, 1.0].
    cyclic_shift_amount : int, default=1
        The number of positions by which active grid square indices are
        rotated in the ``"cyclic shift"`` method. Must be a positive
        integer. Ignored when ``shuffle_method`` is ``"random"``.
    landmark_paths : list of list of tuple of int or list of tuple of int, default=LANDMARK_FACE_OVAL
        A list of one or more closed landmark paths representing the
        region in which the grid shuffle will be applied.

    Returns
    -------
    LayerSpatialGridShuffle
        An instance of the grid shuffle spatial manipulation layer.

    Raises
    ------
    ValueError
        When provided invalid, out-of-range, or unrecognized parameter values.
    """
    # Populate with defaults if None
    time_config = timing_configuration or TimingConfiguration()

    # Validate input params
    try:
        params = GridShuffleParameters(
            random_seed=random_seed, 
            shuffle_method=shuffle_method,
            grid_square_size=grid_square_size,
            grid_dimensions=grid_dimensions,
            mask_overlap_threshold=mask_overlap_threshold,
            cyclic_shift_amount=cyclic_shift_amount,
            landmark_paths=landmark_paths
        )
    except ValidationError as e:
        raise ValueError(f"Invalid parameters for {LayerSpatialGridShuffle.__name__}: {e}")
    
    return LayerSpatialGridShuffle(time_config, params)

__all__ = ["layer_spatial_grid_shuffle", "GridShuffleParameters"]