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
                for fname in glob.glob(f'{dir_name}/*{ext}'):
                    ret_files.append(fname)

    return ret_files
