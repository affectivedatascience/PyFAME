# File Access

Provides utilities for reading, writing, locating, and managing PyFAME data files.

::: pyfame.file_access.file_access_paths
    options:
      members:
        - make_paths
        - load_user_data
        - load_sample_data

---

::: pyfame.file_access.file_access_video_capture
    options:
      members:
        - get_video_capture

---

::: pyfame.file_access.file_access_video_writer
    options: 
      members:
        - get_video_writer

## File Conversions

::: pyfame.file_access.conversion.apply_conversion_remux_to_mp4

---

::: pyfame.file_access.conversion.apply_conversion_image_to_video
    options:
      members:
        - standardise_image_dimensions
        - apply_conversion_image_to_video