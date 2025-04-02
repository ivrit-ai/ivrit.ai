#!/usr/bin/python3

import argparse
import pathlib
import av

from utils import utils


def calculate_total_length(args):
    audio_files = utils.find_files(args.root_dir, args.skip_dir, [".mp3", ".m4a"])

    print(f"Processing {len(audio_files)} files...")

    total_length = 0
    subtitled_length = 0
    subtitled_count = 0

    for idx, f in enumerate(audio_files):
        if idx % 100 == 0:
            print(f"Done processing {idx} files.")

        f = pathlib.Path(f)

        # Get duration using PyAV
        fav = av.open(str(f))
        duration = float(fav.duration) / av.time_base
        total_length += duration

        # Check for subtitle file in both possible locations
        subtitle_path1 = f.with_suffix(f.suffix + '.iw.vtt')  # file.m4a.iw.vtt
        subtitle_path2 = f.with_suffix('.iw.vtt')  # file.iw.vtt
        if subtitle_path1.exists() or subtitle_path2.exists():
            subtitled_length += duration
            subtitled_count += 1

    print(f"\nResults:")
    print(f"Total audio files: {len(audio_files)}")
    print(f"Files with subtitles: {subtitled_count}")
    print(f"Total duration: {total_length/60/60:.2f} hours")
    print(f"Duration with subtitles: {subtitled_length/60/60:.2f} hours")
    print(f"Percentage with subtitles: {(subtitled_count/len(audio_files))*100:.1f}%")


if __name__ == "__main__":
    # Define an argument parser
    parser = argparse.ArgumentParser(
        description="Calculate total duration of audio files and files with subtitles"
    )

    # Add the arguments
    parser.add_argument(
        "--root-dir",
        action="append",
        required=True,
        help="Root directory to start search from. Can be passed multiple times.",
    )
    parser.add_argument(
        "--skip-dir",
        action="append",
        required=False,
        default=[],
        help="Directories to skip. Can be passed multiple times.",
    )

    # Parse the arguments
    args = parser.parse_args()

    calculate_total_length(args)
