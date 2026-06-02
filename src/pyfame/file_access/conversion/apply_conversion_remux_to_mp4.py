from pyfame.file_access.file_access_directories import create_output_directory
from pyfame.utils.exceptions import FileReadError
import os
import pandas as pd
import av

def apply_conversion_remux_to_mp4(file_paths:pd.DataFrame) -> None:
    """ Given an input directory containing one or more video files, remuxes all video files from their current
    container to mp4 video.

    Parameters
    ----------

    file_paths: DataFrame
        A 2-column dataframe consisting of absolute and relative file paths (relative to the working directory root).
    
    Raises
    ------

    FileReadError:
        If expected directory structure is missing, or input paths are invalid.
    
    Returns
    -------

    None
    
    """

    # Extracting the i/o paths from the file_paths dataframe
    # Extracting the i/o paths from the file_paths dataframe
    if len(file_paths["Absolute Path"]) == 0:
        raise FileReadError(message="File_paths dataframe is empty, please inspect your working folder to ensure you have populated the raw/ directory.")
    
    absolute_paths = file_paths["Absolute Path"]

    norm_path = os.path.normpath(absolute_paths.iloc[0])
    norm_cwd = os.path.normpath(os.getcwd())
    rel_dir_path, *_ = os.path.split(os.path.relpath(norm_path, norm_cwd))
    parts = rel_dir_path.split(os.sep)
    root_directory = None

    if parts is not None:
        root_directory = parts[0]
    
    if root_directory is None:
        root_directory = "data"
    
    test_path = os.path.join(norm_cwd, root_directory)

    if not os.path.isdir(test_path):
        raise FileReadError(message=f"Unable to locate the input {root_directory} directory. Please call make_output_paths() to set up the correct directory structure.")
    if not os.path.isdir(os.path.join(test_path, "raw")):
        raise FileReadError(message=f"Unable to locate the 'raw' subdirectory under root directory '{root_directory}'. Please call make_output_paths() to set up the correct directory structure.")
    if not os.path.isdir(os.path.join(test_path, "processed")):
        raise FileReadError(message=f"Unable to locate the 'processed' subdirectory under root directory '{root_directory}'. Please call make_output_paths() to set up the correct directory structure.")
    
    output_directory = create_output_directory(os.path.join(test_path, "processed"), "remuxed")

    for file in absolute_paths:
    
        filename, _ = os.path.splitext(os.path.basename(file))
        input_vid = av.open(file)
        output_vid = av.open(os.path.join(output_directory, f"{filename}.mp4"), 'w')

        # Initialise stream objects
        in_stream = input_vid.streams.video[0]
        out_stream = output_vid.add_stream_from_template(in_stream)

        for packet in input_vid.demux(in_stream):
            # Skip flushing packets
            if packet.dts is None:
                continue

            packet.stream = out_stream
            output_vid.mux(packet)
        
        input_vid.close()
        output_vid.close()

__all__ = ["apply_conversion_remux_to_mp4"]