from pydantic import BaseModel, ValidationError, ValidationInfo, field_validator, PositiveInt
from typing import Optional, Tuple, Dict
from pyfame.landmark.facial_landmarks import *
from pyfame.landmark.get_landmark_coordinates import get_pixel_coordinates_from_landmark
from pyfame.layer.layer import Layer, TimingConfiguration
from pyfame.layer.manipulations.mask import mask_from_landmarks
from pyfame.layer.manipulations.spatial.face_anchors import FaceAnchor
from pyfame.utilities.constants import *
import cv2 as cv
import numpy as np
from operator import itemgetter

class LandmarkRelocateSpec(BaseModel):
    anchor:FaceAnchor
    rotatation_deg:float = 0.0
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
    random_seed:Optional[PositiveInt] = None
    user_spec:Optional[Dict[int, LandmarkRelocateSpec]] = None
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
    def __init__(self, timing_configuration:TimingConfiguration, relocation_parameters:RelocateParameters):
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
        if self.relocate_params.user_spec is None:
            self.relocate_params.user_spec = self._get_random_relocate_spec()

        # Snapshot of initial state
        self._snapshot_state()
    
    def supports_weight(self):
        return False

    def get_layer_parameters(self) -> dict:
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
                rotatation_deg=rotation_angle,
                offsets=(offset_x, offset_y)
            )
        
        return specs
    
    @staticmethod
    def _anchor_to_pixel(anchor, face_center, face_width, face_height):
        ax, ay = anchor.value
        px = int(face_center[0] + ax * face_width)
        py = int(face_center[1] + ay * face_height)
        return px, py
    
    def apply_layer(self, landmarker_coordinates:list[tuple[int,int]], frame:cv.typing.MatLike, dt:float) -> cv.typing.MatLike:
        
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
            spec = self.relocate_params.user_spec.get(key)
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

            rot_mat = cv.getRotationMatrix2D(center=local_center, angle=spec.rotatation_deg, scale=1)

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
        
def layer_spatial_landmark_relocate(timing_configuration:TimingConfiguration | None = None, landmark_relocate_spec:dict[int, LandmarkRelocateSpec] | None = None,
                                    random_seed:int | None = None, max_random_offset:float = 0.15, out_greyscale:bool = True) -> LayerSpatialLandmarkRelocate:
    # Populate with defaults if None
    time_config = timing_configuration or TimingConfiguration()

    # Validate input parameters
    try:
        params = RelocateParameters(
            random_seed=random_seed,
            user_spec=landmark_relocate_spec,
            max_random_offset=max_random_offset,
            out_greyscale=out_greyscale
        )
    except ValidationError as e:
        raise ValueError(f"Invalid parameters for {LayerSpatialLandmarkRelocate.__name__}: {e}")
    
    return LayerSpatialLandmarkRelocate(time_config, params)