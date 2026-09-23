import os
import pandas as pd
from pyfame.file_access.checks import *
from pathlib import Path
from importlib import resources
import shutil
import warnings

def make_paths(root_folder:str = None):
    """ If the top level "data/" folder is not yet set up, this function will
    create the "data/" folder along with the subfolders "raw", "processed/", "logs/",
    "conversion/" and "analysis". 

    Users will populate the "raw/" subfolder with their image/video data, and
    after manipulation results will be written to "processed/". Similarly the
    other subfolders will contain manipulation logs, file conversion results 
    and analysis outputs respectively.

    Parameters
    ----------
    root_folder : str
        An optional existing directory where all of the projects
        data will be stored. By default this is None and this method 
        will create a new "data/" folder at the project root.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        given invalid parameter typings.

    OSError
        given invalid file paths, or nonexistent directory names.
    
    """
    # The standard folder names for Pyfame input and output files. 
    subdir_names = ["raw", "processed", "logs", "conversion", "analysis"]

    if root_folder is not None:
        # If a root_path is provided, check that it is a valid path that exists in the CWD.
        check_type(root_folder, [str])
        check_valid_path(root_folder)
        check_is_dir(root_folder)
    else:
        # If no root_path is provided, the data/ folder will become the root for all of Pyfame's i/o.
        root_folder = os.path.join(os.getcwd(), "data")
        # Check if the directory path already exists; if not then create it. 
        if not os.path.isdir(root_folder):
            os.mkdir(root_folder)

    # For each directory name in dir_names, check if it already exists. 
    # If not, create the directory at the specified path.
    for dir in subdir_names:
        path = os.path.join(root_folder, dir)
        if not os.path.exists(path):
            os.mkdir(path)
        else:
            print(f"Directory {dir} already exists within {root_folder}.")

def load_user_data(data_folder_name:str = "data", include_folders:list[str] = ["raw"]) -> pd.DataFrame:
    """ Given the name of the projects data folder, and a list specifying 
    which subfolders to walk through, returns a DataFrame of relative and
    absolute file paths to all files contained in the specified subfolders.

    Parameters
    ----------
    data_folder_name : str
        The name of the root data folder, defaults to "data".
    
    include_folders : list[str]
        A list of subfolder names to include in the returned
        file paths DataFrame. Can include one or more of "raw", 
        "processed", "logs", "conversion" or "analysis". Defaults 
        to just the "raw" subfolder, returning paths to the users 
        unmodified images and videos.
    
    Returns
    -------
    DataFrame
        A 2-column dataframe containing absolute and relative 
        file paths.
    
    Raises
    ------
    OSError
        Given invalid folder names.
    """
    if not os.path.exists(Path(os.getcwd(), data_folder_name)):
        raise OSError(f"Folder {data_folder_name} does not exist or cannot be found in the cwd.")
    data_path = Path(os.getcwd(), data_folder_name)

    for dir in include_folders:
        if not os.path.exists(data_path / dir):
            raise OSError(f"Folder {data_path / dir} does not exist or cannot be found.")

    full_file_paths = []
    rel_file_paths = []

    for path, dirs, files in os.walk(data_path, topdown=True):
        # Include only selected subdirectories, excluding PyFAME's sample data
        dirs[:] = [d for d in dirs if d in include_folders and d != "samples"]

        for file in files:
            full_path = os.path.join(path, file)
            rel_path = os.path.relpath(full_path, data_path)
            rel_path = os.path.join(os.path.basename(data_path), rel_path)
            
            full_file_paths.append(full_path)
            rel_file_paths.append(rel_path)

    df1 = pd.DataFrame({
        "Absolute Path":full_file_paths,
        "Relative Path":rel_file_paths,
    })

    if df1.empty:
        warnings.warn(
            "load_user_data(): Could not find any files at the specified location, " \
            "returning an empty DataFrame.",
            UserWarning
        )

    return df1

def load_sample_data(data_folder_name:str = "data") -> pd.DataFrame:
    """ Given the name of the projects data folder, populates "raw/samples/"
    with sample data and returns a DataFrame of relative and
    absolute file paths to each of the sample files.

    Parameters
    ----------
    data_folder_name : str
        The name of the root data folder, defaults to "data".
    
    Returns
    -------
    DataFrame
        A 2-column dataframe containing absolute and relative 
        file paths.
    
    Raises
    ------
    OSError
        Given invalid folder names.
    """
    cwd = Path.cwd()
    data_path = cwd / data_folder_name

    if not data_path.exists():
        raise OSError(f"Folder {data_folder_name} does not exist or cannot be found in the cwd.")

    dest_path = data_path / "raw" / "samples"
    dest_path.mkdir(exist_ok=True, parents=True)

    sample_dirs = [
        resources.files("pyfame").joinpath("data", "sample", "01"),
        resources.files("pyfame").joinpath("data", "sample", "02")
    ]

    full_file_paths = []
    rel_file_paths = []

    for sample_dir in sample_dirs:
        for res in sample_dir.iterdir():
            if res.name.endswith(".mp4"):
                output_path = dest_path / res.name

                # Ensure files are only written if they dont yet exist
                if not output_path.exists():
                    with res.open("rb") as src:
                        with output_path.open("wb") as dst:
                            shutil.copyfileobj(src, dst)

                full_file_paths.append(output_path)
                rel_file_paths.append(output_path.relative_to(cwd))

    df1 = pd.DataFrame({
        "Absolute Path":full_file_paths,
        "Relative Path":rel_file_paths,
    })

    return df1

def get_directory_walk(input_directory:str) -> pd.DataFrame:
        full_file_paths = []
        rel_file_paths = []

        for path, dirs, files in os.walk(input_directory, topdown=True):
            for file in files:
                full_path = os.path.join(path, file)
                rel_path = os.path.relpath(full_path, input_directory)
                rel_path = os.path.join(os.path.basename(input_directory), rel_path)
                
                full_file_paths.append(full_path)
                rel_file_paths.append(rel_path)

        df1 = pd.DataFrame({
            "Absolute Path":full_file_paths,
            "Relative Path":rel_file_paths,
        })

        return df1

def get_sub_directories_relative_to_path(file_path:str, anchor_directory:str) -> Path:
    path = Path(file_path).resolve()

    try:
        parts = path.parts
        anchor_idx = parts.index(anchor_directory)
        relative_parts = parts[anchor_idx + 1:-1]
        return Path(*relative_parts)
    except ValueError:
        raise ValueError(f"Anchor directory '{anchor_directory}' not found in path: {file_path}")

def get_landmarker_task_path() -> str:
    with resources.as_file(resources.files("pyfame.models.mediapipe") / "face_landmarker.task") as path:
        return str(path)
    
__all__ = ["make_paths", "load_user_data", "load_sample_data", "get_directory_walk", "get_sub_directories_relative_to_path", "get_landmarker_task_path"]