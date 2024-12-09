#!/usr/bin/python3

import argparse

from utils import utils
from vad.frame_vad_infer import generate_frame_vad_predictions


def preprocess_predict_vad(args):
    audio_files = utils.find_files(args.root_dir, args.skip_dir, [".mp3", ".m4a"])

    # We invoke a single worker - Nemo uses batching to use the GPU
    # while generating frame-level vaf predictions.
    # A pre-splitting step to limit max batch audio file duration
    # can be parallelized if it seems to be a bottleneck.
    nemo_frame_vad_config = {
        "force_reprocess": args.force_reprocess,
        "nemo_vad_presplit_duration": args.nemo_presplit_max_duration,
        "nemo_vad_presplit_workers": args.nemo_presplit_workers,
    }
    generate_frame_vad_predictions(audio_files, args.target_dir, nemo_frame_vad_config)


if __name__ == "__main__":
    # Define an argument parser
    parser = argparse.ArgumentParser(
        description="Search for audio files in a directory excluding specified subdirectories"
    )

    # Add the arguments
    parser.add_argument("processing_type", choices=["vad", "transcribe", "segment"])
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
    parser.add_argument(
        "--target-dir",
        type=str,
        required=True,
        help="The directory where splitted audios will be stored.",
    )
    parser.add_argument(
        "--force-reprocess",
        required=False,
        action="store_true",
        help="Force processing even if already done",
    )
    parser.add_argument(
        "--nemo-presplit-workers",
        type=int,
        required=False,
        default=1,
        help="NeMo Frame VAD: How many CPU workers to use for pre-splitting long audio files",
    )
    parser.add_argument(
        "--nemo-presplit-max-duration",
        type=int,
        required=False,
        default=400,
        help="NeMo Frame VAD: Max duration of audio files before splitting to multiple files as part of the pre processing step",
    )

    # Parse the arguments
    args = parser.parse_args()

    if args.processing_type == "vad":
        pass
        preprocess_predict_vad(args)
    else:
        raise NotImplementedError(f"Processing type {args.processing_type} is not implemented yet.")
