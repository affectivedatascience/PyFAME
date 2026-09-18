# File Access

Provides utilities for reading, writing, locating, and managing PyFAME data files.

::: pyfame.file_access.file_access_paths
    options:
      members:
        - make_paths

!!! note
    On your first time running PyFAME locally, make sure to run `make_paths()` once first on its own! This will not return any paths and will instead create and populate the `data/` folder in your working directory. 

    If you plan to manipulate your own custom images or videos you will then populate the `data/raw` subfolder with your files, otherwise the next time you run `make_paths()` run it with the parameter `load_sample_data = true` to instead populate `data/raw` with several provided sample videos.

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