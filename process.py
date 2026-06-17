#!/usr/bin/python3

import argparse

from utils import utils
from vad.frame_vad_infer import generate_frame_vad_predictions, exclude_already_predicted
from transcribe.stable_ts import transcribe, exclude_already_transcribed


def process_vad_on_files(audio_files: list, args):
    # Assumed that all audio_files here require processing.
    # Any prefiltering or size limits should be before calling this method
    
    if len(audio_files) == 0:
        print("No audio files to predict vad on.")
        return

    # We invoke a single worker - Nemo uses batching to use the GPU
    # while generating frame-level vaf predictions.
    # A pre-splitting step to limit max batch audio file duration
    # can be parallelized if it seems to be a bottleneck.
    nemo_frame_vad_config = {
        "force_reprocess": args.force_reprocess,
        "nemo_vad_pretranscode_workers": args.nemo_pretranscode_workers,
        "nemo_vad_presplit_duration": args.nemo_presplit_max_duration,
        "nemo_vad_presplit_workers": args.nemo_presplit_workers,
    }

    # Throwing all input files (could be 1000s...) on Nemo could work...
    # but to make recovery easier, and conserve temp storage (intermediate audio transcoding is stored in WAV format)
    # we will chunk it to (large) number thus to balance the cost of loading the vad model every time
    chunk_size = args.nemo_files_chunk_size
    for audio_files_chunk in [audio_files[i : i + chunk_size] for i in range(0, len(audio_files), chunk_size)]:
        print(f"Processing vad for a {len(audio_files_chunk)} files chunk")
        generate_frame_vad_predictions(audio_files_chunk, args.target_dir, nemo_frame_vad_config)


def process_vad(args):
    audio_files = utils.find_files(args.root_dir, args.skip_dir, [".mp3", ".m4a"])
    audio_files = audio_files if args.force_reprocess else exclude_already_predicted(audio_files, args.target_dir)

    if args.max_files_to_process is not None:
        audio_files = audio_files[: args.max_files_to_process]

    process_vad_on_files(audio_files, args)


def process_transcribe(args):
    audio_files = utils.find_files(args.root_dir, args.skip_dir, [".mp3", ".m4a"])
    audio_files = audio_files if args.force_reprocess else exclude_already_transcribed(audio_files, args.target_dir)

    if len(audio_files) == 0:
        print("No audio files to transcribe.")
        return

    if args.max_files_to_process is not None:
        audio_files = audio_files[: args.max_files_to_process]

    # require that all processed files have frame level vads
    audio_files_to_vad = (
        audio_files if args.force_reprocess else exclude_already_predicted(audio_files, args.target_dir)
    )
    process_vad_on_files(audio_files_to_vad, args)

    config = {
        "force_reprocess": args.force_reprocess,
        "consider_speech_clips": True,
        "vad_input_dir": None,
        "generate_timestamp_tokens": True,
        "get_word_level_timestamp": True,
        "use_stable_ts": True,
        "whisper_model_name": args.transcribe_model,
        "language": "he",
        "compute_type": "float16",
    }

    transcribe(
        audio_files,
        args.target_dir,
        config,
    )


if __name__ == "__main__":
    # Define an argument parser
    parser = argparse.ArgumentParser(
        description="Process normalized source audio files into crowd-transcribe bound dataset"
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
        "--max-files-to-process",
        type=int,
        required=False,
        help="Max amount of audio files to process in this run",
    )
    parser.add_argument(
        "--force-reprocess",
        required=False,
        action="store_true",
        help="Force processing even if already done",
    )
    parser.add_argument(
        "--nemo-files-chunk-size",
        type=int,
        required=False,
        default=200,
        help="NeMo Frame VAD: How many files, at most to send to the VAD model. All chunk is transcribed, splitted and processed serially.",
    )
    parser.add_argument(
        "--nemo-pretranscode-workers",
        type=int,
        required=False,
        default=1,
        help="NeMo Frame VAD: How many CPU threads to use for pre-transcoding input audio (each uses ffmpeg sub process)",
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
    parser.add_argument(
        "--transcribe-model",
        type=str,
        required=False,
        default="ivrit-ai/faster-whisper-v2-d4",
        help="The whisper model to use (faster-whisper format)",
    )

    # Parse the arguments
    args = parser.parse_args()

    if args.processing_type == "vad":
        pass
        process_vad(args)
    elif args.processing_type == "transcribe":
        process_transcribe(args)
    else:
        raise NotImplementedError(f"Processing type {args.processing_type} is not implemented yet.")
