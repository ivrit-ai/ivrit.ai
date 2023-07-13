import os
import pathlib

def find_files(root_directories, skip, extensions):
    ret_files = []

    for directory in root_directories:
        for dir_name, subdirs, files in os.walk(directory):
            # Check if the directory name exists in the skip list
            if set(skip).intersection(set(subdirs)):
                # Remove the directories to skip from the subdirectories list
                subdirs[:] = [d for d in subdirs if d not in skip]

            # Print the mp3 file names
            for fname in files:
                path = pathlib.Path(fname)
                if path.suffix in extensions:
                    ret_files.append(os.path.join(dir_name, fname))

    return ret_files
