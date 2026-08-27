from pyfame.utils.constants import *
from pyfame.landmark.get_landmark_coordinates import get_relative_landmark_coordinates
from pyfame.landmark.facial_landmarks import *
from pyfame.utils.exceptions import *
import cv2 as cv
import numpy as np

def mask_from_landmarks(frame:cv.typing.MatLike, landmark_paths:list[list[tuple[int,int]]] | list[tuple[int,int]], landmarker_coordinates) -> cv.typing.MatLike:
    """ Given a landmark path, create a binary image mask based
    on the input image or frame.

    Parameters
    ----------
    frame : MatLike
        A static image or decoded video frame used to create the binary image mask.
    landmark_paths : array_like
        A predefined single landmark path, or a list of landmark paths specifying
        the masked region in the frame.
    landmarker_coordinates : array_like
        A list of screen pixel coordinates returned by the mediapipe 
        FaceLandmarker task.

    Returns
    -------
    mask : MatLike
        A binary image with everything masked out except for the regions
        enclosed by landmark_paths.
    """
    masked_frame = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

    # Extracting all of the relevant landmark pixel coordinates
    lc_screen_coords = get_relative_landmark_coordinates(landmarker_coordinates, LANDMARK_LEFT_CHEEK)
    rc_screen_coords = get_relative_landmark_coordinates(landmarker_coordinates, LANDMARK_RIGHT_CHEEK)
    chin_screen_coords = get_relative_landmark_coordinates(landmarker_coordinates, LANDMARK_CHIN)
    nose_wide_screen_coords = get_relative_landmark_coordinates(landmarker_coordinates, create_landmark_path(NOSE_WIDE_IDX))
    ler_screen_coords = get_relative_landmark_coordinates(landmarker_coordinates, LANDMARK_LEFT_EYE_REGION)
    rer_screen_coords = get_relative_landmark_coordinates(landmarker_coordinates, LANDMARK_RIGHT_EYE_REGION)
    le_screen_coords = get_relative_landmark_coordinates(landmarker_coordinates, LANDMARK_LEFT_EYE)
    re_screen_coords = get_relative_landmark_coordinates(landmarker_coordinates, LANDMARK_RIGHT_EYE)
    li_screen_coords = get_relative_landmark_coordinates(landmarker_coordinates, LANDMARK_LEFT_IRIS)
    ri_screen_coords = get_relative_landmark_coordinates(landmarker_coordinates, LANDMARK_RIGHT_IRIS)
    leb_screen_coords = get_relative_landmark_coordinates(landmarker_coordinates, LANDMARK_LEFT_EYEBROW)
    reb_screen_coords = get_relative_landmark_coordinates(landmarker_coordinates, LANDMARK_RIGHT_EYEBROW)
    mouth_screen_coords = get_relative_landmark_coordinates(landmarker_coordinates, LANDMARK_MOUTH)
    lip_o_screen_coords = get_relative_landmark_coordinates(landmarker_coordinates, LANDMARK_LIPS_OUTER_CONTOUR)
    lip_i_screen_coords = get_relative_landmark_coordinates(landmarker_coordinates, LANDMARK_LIPS_INNER_CONTOUR)
    fo_screen_coords = get_relative_landmark_coordinates(landmarker_coordinates, LANDMARK_FACE_OVAL)

    if isinstance(landmark_paths[0], list):
        for path in landmark_paths:
            match path:
                # Both Cheeks
                case [(0,)]:
                    lc_screen_coords = np.array(lc_screen_coords, dtype=np.int32)
                    lc_screen_coords.reshape((-1, 1, 2))

                    lc_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    lc_mask = cv.fillPoly(img=lc_mask, pts=[lc_screen_coords], color=(255,255,255))
                    lc_mask = lc_mask.astype(bool)

                    rc_screen_coords = np.array(rc_screen_coords, dtype=np.int32)
                    rc_screen_coords.reshape((-1, 1, 2))

                    rc_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    rc_mask = cv.fillPoly(img=rc_mask, pts=[rc_screen_coords], color=(255,255,255))
                    rc_mask = rc_mask.astype(bool)

                    masked_frame[lc_mask] = 255
                    masked_frame[rc_mask] = 255
                
                # Left Cheek Only
                case [(1,)]:
                    # cv2.fillPoly requires a specific shape and int32 values for the points
                    lc_screen_coords = np.array(lc_screen_coords, dtype=np.int32)
                    lc_screen_coords.reshape((-1, 1, 2))

                    lc_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    lc_mask = cv.fillPoly(img=lc_mask, pts=[lc_screen_coords], color=(255,255,255))
                    lc_mask = lc_mask.astype(bool)

                    masked_frame[lc_mask] = 255
                
                # Right Cheek Only
                case [(2,)]:
                    # cv2.fillPoly requires a specific shape and int32 values for the points
                    rc_screen_coords = np.array(rc_screen_coords, dtype=np.int32)
                    rc_screen_coords.reshape((-1, 1, 2))

                    rc_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    rc_mask = cv.fillPoly(img=rc_mask, pts=[rc_screen_coords], color=(255,255,255))
                    rc_mask = rc_mask.astype(bool)

                    masked_frame[rc_mask] = 255

                # Cheeks and Nose
                case [(3,)]: 
                    lc_screen_coords = np.array(lc_screen_coords, dtype=np.int32)
                    lc_screen_coords.reshape((-1, 1, 2))

                    lc_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    lc_mask = cv.fillPoly(img=lc_mask, pts=[lc_screen_coords], color=(255,255,255))
                    lc_mask = lc_mask.astype(bool)

                    rc_screen_coords = np.array(rc_screen_coords, dtype=np.int32)
                    rc_screen_coords.reshape((-1, 1, 2))

                    rc_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    rc_mask = cv.fillPoly(img=rc_mask, pts=[rc_screen_coords], color=(255,255,255))
                    rc_mask = rc_mask.astype(bool)

                    nose_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    nose_mask = cv.fillConvexPoly(nose_mask, np.array(nose_wide_screen_coords), 1)
                    nose_mask = nose_mask.astype(bool)

                    masked_frame[lc_mask] = 255
                    masked_frame[rc_mask] = 255
                    masked_frame[nose_mask] = 255
                
                # Both eye regions
                case [(4,)]:
                    # Creating boolean masks for the facial landmarks 
                    ler_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                    ler_mask = cv.fillConvexPoly(ler_mask, np.array(ler_screen_coords), 1)
                    ler_mask = ler_mask.astype(bool)

                    rer_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                    rer_mask = cv.fillConvexPoly(rer_mask, np.array(rer_screen_coords), 1)
                    rer_mask = rer_mask.astype(bool)

                    masked_frame[ler_mask] = 255
                    masked_frame[rer_mask] = 255

                # Face Skin
                case [(5,)]:
                    # Creating boolean masks for the facial landmarks 
                    le_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    le_mask = cv.fillConvexPoly(le_mask, np.array(le_screen_coords), 1)
                    le_mask = le_mask.astype(bool)

                    re_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    re_mask = cv.fillConvexPoly(re_mask, np.array(re_screen_coords), 1)
                    re_mask = re_mask.astype(bool)

                    leb_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    leb_mask = cv.fillConvexPoly(leb_mask, np.array(leb_screen_coords), 1)
                    leb_mask = leb_mask.astype(bool)

                    reb_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    reb_mask = cv.fillConvexPoly(reb_mask, np.array(reb_screen_coords), 1)
                    reb_mask = reb_mask.astype(bool)

                    lip_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                    lip_mask = cv.fillConvexPoly(lip_mask, np.array(lip_o_screen_coords), 1)
                    lip_mask = lip_mask.astype(bool)

                    oval_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                    oval_mask = cv.fillConvexPoly(oval_mask, np.array(fo_screen_coords), 1)
                    oval_mask = oval_mask.astype(bool)

                    # Masking the face oval
                    masked_frame[oval_mask] = 255
                    masked_frame[le_mask] = 0
                    masked_frame[re_mask] = 0
                    masked_frame[leb_mask] = 0
                    masked_frame[reb_mask] = 0
                    masked_frame[lip_mask] = 0
                
                # Chin
                case [(6,)]:    
                    chin_screen_coords = np.array(chin_screen_coords, dtype=np.int32)
                    chin_screen_coords.reshape((-1, 1, 2))
                    
                    chin_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    chin_mask = cv.fillPoly(img=chin_mask, pts=[chin_screen_coords], color=(255,255,255))
                    chin_mask = chin_mask.astype(bool)

                    masked_frame[chin_mask] = 255
                
                # Both eyes (sclera)
                case [(8,)]:
                    le_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    le_mask = cv.fillConvexPoly(le_mask, np.array(le_screen_coords), 1)
                    le_mask = le_mask.astype(bool)

                    re_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    re_mask = cv.fillConvexPoly(re_mask, np.array(re_screen_coords), 1)
                    re_mask = re_mask.astype(bool)

                    masked_frame[le_mask] = 255
                    masked_frame[re_mask] = 255
                
                # Both irises
                case [(9,)]:
                    li_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    li_mask = cv.fillConvexPoly(li_mask, np.array(li_screen_coords), 1)
                    li_mask = li_mask.astype(bool)

                    ri_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    ri_mask = cv.fillConvexPoly(ri_mask, np.array(ri_screen_coords), 1)
                    ri_mask = ri_mask.astype(bool)

                    masked_frame[li_mask] = 255
                    masked_frame[ri_mask] = 255

                case [(10,)]:
                    leb_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    leb_mask = cv.fillConvexPoly(leb_mask, np.array(leb_screen_coords), 1)
                    leb_mask = leb_mask.astype(bool)

                    reb_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    reb_mask = cv.fillConvexPoly(reb_mask, np.array(reb_screen_coords), 1)
                    reb_mask = reb_mask.astype(bool)

                    masked_frame[leb_mask] = 255
                    masked_frame[reb_mask] = 255

                case [(11,)]:
                    lip_o_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    lip_o_mask = cv.fillConvexPoly(lip_o_mask, np.array(lip_o_screen_coords), 1)
                    lip_o_mask = lip_o_mask.astype(bool)

                    lip_i_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    lip_i_mask = cv.fillConvexPoly(lip_i_mask, np.array(lip_i_screen_coords), 1)
                    lip_i_mask = lip_i_mask.astype(bool)

                    masked_frame[lip_o_mask] = 255
                    masked_frame[lip_i_mask] = 0

                case _:
                    cur_landmark_coords = get_relative_landmark_coordinates(landmarker_coordinates, path)

                    # Creating boolean masks for the facial landmarks 
                    bool_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                    bool_mask = cv.fillConvexPoly(bool_mask, np.array(cur_landmark_coords), 1)
                    bool_mask = bool_mask.astype(bool)

                    masked_frame[bool_mask] = 255
    else:
        match landmark_paths:
            # Both Cheeks
            case [(0,)]:
                lc_screen_coords = np.array(lc_screen_coords, dtype=np.int32)
                lc_screen_coords.reshape((-1, 1, 2))

                lc_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                lc_mask = cv.fillPoly(img=lc_mask, pts=[lc_screen_coords], color=(255,255,255))
                lc_mask = lc_mask.astype(bool)

                rc_screen_coords = np.array(rc_screen_coords, dtype=np.int32)
                rc_screen_coords.reshape((-1, 1, 2))

                rc_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                rc_mask = cv.fillPoly(img=rc_mask, pts=[rc_screen_coords], color=(255,255,255))
                rc_mask = rc_mask.astype(bool)

                masked_frame[lc_mask] = 255
                masked_frame[rc_mask] = 255
            
            # Left Cheek Only
            case [(1,)]:
                # cv2.fillPoly requires a specific shape and int32 values for the points
                lc_screen_coords = np.array(lc_screen_coords, dtype=np.int32)
                lc_screen_coords.reshape((-1, 1, 2))

                lc_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                lc_mask = cv.fillPoly(img=lc_mask, pts=[lc_screen_coords], color=(255,255,255))
                lc_mask = lc_mask.astype(bool)

                masked_frame[lc_mask] = 255
            
            # Right Cheek Only
            case [(2,)]:
                # cv2.fillPoly requires a specific shape and int32 values for the points
                rc_screen_coords = np.array(rc_screen_coords, dtype=np.int32)
                rc_screen_coords.reshape((-1, 1, 2))

                rc_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                rc_mask = cv.fillPoly(img=rc_mask, pts=[rc_screen_coords], color=(255,255,255))
                rc_mask = rc_mask.astype(bool)

                masked_frame[rc_mask] = 255

            # Cheeks and Nose
            case [(3,)]: 
                lc_screen_coords = np.array(lc_screen_coords, dtype=np.int32)
                lc_screen_coords.reshape((-1, 1, 2))

                lc_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                lc_mask = cv.fillPoly(img=lc_mask, pts=[lc_screen_coords], color=(255,255,255))
                lc_mask = lc_mask.astype(bool)

                rc_screen_coords = np.array(rc_screen_coords, dtype=np.int32)
                rc_screen_coords.reshape((-1, 1, 2))

                rc_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                rc_mask = cv.fillPoly(img=rc_mask, pts=[rc_screen_coords], color=(255,255,255))
                rc_mask = rc_mask.astype(bool)

                nose_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                nose_mask = cv.fillConvexPoly(nose_mask, np.array(nose_wide_screen_coords), 1)
                nose_mask = nose_mask.astype(bool)

                masked_frame[lc_mask] = 255
                masked_frame[rc_mask] = 255
                masked_frame[nose_mask] = 255
            
            # Both eyes
            case [(4,)]:
                # Creating boolean masks for the facial landmarks 
                ler_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                ler_mask = cv.fillConvexPoly(ler_mask, np.array(ler_screen_coords), 1)
                ler_mask = ler_mask.astype(bool)

                rer_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                rer_mask = cv.fillConvexPoly(rer_mask, np.array(rer_screen_coords), 1)
                rer_mask = rer_mask.astype(bool)

                masked_frame[ler_mask] = 255
                masked_frame[rer_mask] = 255

            # Face Skin
            case [(5,)]:
                # Creating boolean masks for the facial landmarks 
                le_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                le_mask = cv.fillConvexPoly(le_mask, np.array(le_screen_coords), 1)
                le_mask = le_mask.astype(bool)

                re_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                re_mask = cv.fillConvexPoly(re_mask, np.array(re_screen_coords), 1)
                re_mask = re_mask.astype(bool)

                leb_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                leb_mask = cv.fillConvexPoly(leb_mask, np.array(leb_screen_coords), 1)
                leb_mask = leb_mask.astype(bool)

                reb_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                reb_mask = cv.fillConvexPoly(reb_mask, np.array(reb_screen_coords), 1)
                reb_mask = reb_mask.astype(bool)

                lip_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                lip_mask = cv.fillConvexPoly(lip_mask, np.array(lip_o_screen_coords), 1)
                lip_mask = lip_mask.astype(bool)

                oval_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                oval_mask = cv.fillConvexPoly(oval_mask, np.array(fo_screen_coords), 1)
                oval_mask = oval_mask.astype(bool)

                # Masking the face oval
                masked_frame[oval_mask] = 255
                masked_frame[le_mask] = 0
                masked_frame[re_mask] = 0
                masked_frame[leb_mask] = 0
                masked_frame[reb_mask] = 0
                masked_frame[lip_mask] = 0
            
            # Chin
            case [(6,)]:    
                chin_screen_coords = np.array(chin_screen_coords, dtype=np.int32)
                chin_screen_coords.reshape((-1, 1, 2))
                
                chin_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                chin_mask = cv.fillPoly(img=chin_mask, pts=[chin_screen_coords], color=(255,255,255))
                chin_mask = chin_mask.astype(bool)

                masked_frame[chin_mask] = 255

            # Both eyes (sclera)
            case [(8,)]:
                le_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                le_mask = cv.fillConvexPoly(le_mask, np.array(le_screen_coords), 1)
                le_mask = le_mask.astype(bool)

                re_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                re_mask = cv.fillConvexPoly(re_mask, np.array(re_screen_coords), 1)
                re_mask = re_mask.astype(bool)

                masked_frame[le_mask] = 255
                masked_frame[re_mask] = 255
            
            # Both irises
            case [(9,)]:
                li_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                li_mask = cv.fillConvexPoly(li_mask, np.array(li_screen_coords), 1)
                li_mask = li_mask.astype(bool)

                ri_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                ri_mask = cv.fillConvexPoly(ri_mask, np.array(ri_screen_coords), 1)
                ri_mask = ri_mask.astype(bool)

                masked_frame[li_mask] = 255
                masked_frame[ri_mask] = 255

            # Both Eyebrows
            case [(10,)]:
                leb_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                leb_mask = cv.fillConvexPoly(leb_mask, np.array(leb_screen_coords), 1)
                leb_mask = leb_mask.astype(bool)

                reb_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                reb_mask = cv.fillConvexPoly(reb_mask, np.array(reb_screen_coords), 1)
                reb_mask = reb_mask.astype(bool)

                masked_frame[leb_mask] = 255
                masked_frame[reb_mask] = 255

            # Lips inner and outer contour
            case [(11,)]:
                lip_o_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                lip_o_mask = cv.fillConvexPoly(lip_o_mask, np.array(lip_o_screen_coords), 1)
                lip_o_mask = lip_o_mask.astype(bool)

                lip_i_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                lip_i_mask = cv.fillConvexPoly(lip_i_mask, np.array(lip_i_screen_coords), 1)
                lip_i_mask = lip_i_mask.astype(bool)

                masked_frame[lip_o_mask] = 255
                masked_frame[lip_i_mask] = 0

            case _:
                cur_landmark_coords = get_relative_landmark_coordinates(landmarker_coordinates, landmark_paths)

                # Creating boolean masks for the facial landmarks 
                bool_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                bool_mask = cv.fillConvexPoly(bool_mask, np.array(cur_landmark_coords), 1)
                bool_mask = bool_mask.astype(bool)

                masked_frame[bool_mask] = 255

    return masked_frame.astype(np.uint8)

__all__ = [mask_from_landmarks]