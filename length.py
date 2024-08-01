#!/usr/bin/python3

import argparse
import pathlib
import mutagen.mp3
import pydub

from utils import utils

def calculate_total_length(args):
    audio_files = utils.find_files(args.root_dir, args.skip_dir, ['.mp3', '.m4a'])

    print(f'Processing {len(audio_files)} files...')

    total_length = 0

    for idx, f in enumerate(audio_files):
        if idx % 10000 == 0:
            print(f'Done processing {idx} files.')

        total_length += mutagen.mp3.MP3(f).info.length

    print(f'Done processing all files. Length in hours: {total_length/60/60}.')

if __name__ == '__main__':
    # Define an argument parser
    parser = argparse.ArgumentParser(description='Search for audio files in a directory excluding specified subdirectories')

    # Add the arguments
    parser.add_argument('--root-dir', action='append', required=True,
                        help='Root directory to start search from. Can be passed multiple times.')
    parser.add_argument('--skip-dir', action='append', required=False, default=[],
                        help='Directories to skip. Can be passed multiple times.')

    # Parse the arguments
    args = parser.parse_args()

    calculate_total_length(args)

