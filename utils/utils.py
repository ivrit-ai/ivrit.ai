import glob
import os
import pathlib


def find_files(root_directories, skip, extensions):
    ret_files = []

    for directory in root_directories:
        for dir_name, subdirs, _ in os.walk(directory, followlinks=True):
            # Check if the directory name exists in the skip list
            if set(skip).intersection(set(subdirs)):
                # Remove the directories to skip from the subdirectories list
                subdirs[:] = [d for d in subdirs if d not in skip]

            # Select all relevant files
            for ext in extensions:
                for fname in glob.glob(f"{dir_name}/*{ext}"):
                    ret_files.append(fname)

    return ret_files


def parse_source_and_episode_from_filename(filename: str) -> tuple[str, str]:
    """Get source and episode from file path as they appear in the Ivrit.ai dataset filesystem structure
    Episode is the file name without the extension
    Source is the parent folder name, if no parent folder, return None.

    Args:
        filename (str): _description_

    Returns:
        tuple[str, str]: (source, episode) derived from file name and path
    """
    path = pathlib.Path(filename)
    source_name = path.parent.name if path.parent else None
    episode_name = path.stem
    return source_name, episode_name
