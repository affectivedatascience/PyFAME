from enum import Enum

class FaceAnchor(Enum):
    """
    Facial location anchors for LayerSpatialLandmarkRelocate.
    Anchors are relative to the subject's face, not the user-
    perceived location.
    """
    SUBJECT_UPPER_RIGHT = (-0.25, -0.25)
    SUBJECT_UPPER_CENTER = (0.0, -0.25)
    SUBJECT_UPPER_LEFT = (0.25, -0.25)

    SUBJECT_CENTER_RIGHT = (-0.25, 0.0)
    SUBJECT_CENTER = (0.0, 0.0)
    SUBJECT_CENTER_LEFT = (0.25, 0.0)
    
    SUBJECT_LOWER_CENTER = (0.0, 0.25)