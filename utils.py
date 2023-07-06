import os

def find_mp3_files(root_directories, skip):
    mp3s = []

    for directory in root_directories:
        for dir_name, subdirs, files in os.walk(directory):
            # Check if the directory name exists in the skip list
            if set(skip).intersection(set(subdirs)):
                # Remove the directories to skip from the subdirectories list
                subdirs[:] = [d for d in subdirs if d not in skip]

            # Print the mp3 file names
            for fname in files:
                if fname.endswith('.mp3'):
                    mp3s.append(os.path.join(dir_name, fname))

    return mp3s
