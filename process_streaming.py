#!/usr/bin/python3

import argparse
import json
import os
import pathlib
import subprocess

import torch
from torch.multiprocessing import Pool
import torch.multiprocessing.spawn

import utils

# Using more than 1 thread appears to actually make VAD slower.
# Using a single thread, and forking to run multiple processes.
#
# Yair, July 2023
torch.set_num_threads(1)

SAMPLING_RATE = 16000


def parse_audio_info(audio_info):
    if audio_info is not None and "streams" in audio_info:
        for stream in audio_info["streams"]:
            if stream["codec_type"] == "audio":
                sample_rate = int(stream["sample_rate"])
                channels = int(stream["channels"])
                duration = float(stream["duration"])
                return {"sample_rate": sample_rate, "channels": channels, "duration": duration}
    return None


def get_audio_properties(input_file):
    # Run ffprobe to get audio properties in JSON format
    cmd = ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-select_streams", "a", input_file]
    result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    output = result.stdout

    # Parse the JSON output
    info = None
    try:
        raw_info = json.loads(output)
        info = parse_audio_info(raw_info)
    except:
        print("Warning - Unable to probe input audio source properties...")
        pass

    return info


def should_transcode(audio_info):
    return (
        audio_info is None  # Cannot detect audio info
        or audio_info["channels"] > 1  # Not mono
        or audio_info["sample_rate"] != SAMPLING_RATE  # Wrong sampling rate
    )


def transcode_audio(input_file, output_file):
    cmd = ["ffmpeg", "-y", "-i", input_file, "-c:a", "libmp3lame", "-q:a", "2", "-ac", "1", "-ar", "16000", output_file]
    subprocess.run(cmd)


def copy_audio(input_file, output_file):
    cmd = ["ffmpeg", "-y", "-i", input_file, "-c", "copy", output_file]
    subprocess.run(cmd)


def prepare_audio_for_processing(source_filename, output_filename):
    audio_info = get_audio_properties(source_filename)

    if not output_filename.is_file():
        if should_transcode(audio_info):
            print("Transcoding audio to expected processing format")
            transcode_audio(source_filename, output_filename)
        else:
            print("Copying audio for processing")
            copy_audio(source_filename, output_filename)

    return audio_info


def bulk_vad(args):
    if args.jobs:
        parallel_processes = args.jobs

    audio_files = utils.find_files(args.root_dir, args.skip_dir, [".mp3", ".m4a"])
    processing_groups = [audio_files[i::parallel_processes] for i in range(parallel_processes)]

    vad_process_config = {
        "target_dir": args.target_dir,
        "force_reprocess": args.force_reprocess,
        "segment_audio": args.process_audio_in_segments,
        "audio_segment_length_s": args.audio_segment_length_s,
        "min_speech_duration_ms": args.min_speech_duration_ms,
        "max_speech_duration_s": args.max_speech_duration_s,
        "min_silence_duration_ms": args.min_silence_duration_ms,
        "speech_pad_ms": args.speech_pad_ms,
        "speech_threshold": args.speech_threshold,
    }

    with Pool(processes=len(processing_groups)) as pool:
        pool.starmap(
            invoke_processing_group,
            [(i, processing_groups[i], vad_process_config) for i in range(len(processing_groups))],
        )


def invoke_processing_group(this_group_id: int, this_group: list, config):
    model, torch_utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad")
    bulk_vad_single_process(config, this_group_id, this_group, model, torch_utils)


def bulk_vad_single_process(config, group_idx, audio_files, model, torch_utils):
    (get_speech_timestamps, _, read_audio, _, _) = torch_utils

    for idx, audio_file in enumerate(audio_files):
        print(f"Processing group {group_idx}, file #{idx}: {audio_file}")

        audio_path = pathlib.Path(audio_file)
        source = audio_path.parent.name
        episode = audio_path.stem
        target_dir = pathlib.Path(config["target_dir"]) / source / episode

        # Check if file was already processed successfully.
        try:
            desc_filename = target_dir / "desc.json"
            if desc_filename.is_file() and not config["force_reprocess"]:
                print("Already processed. Skipping...")
                json.load(open(desc_filename, "r"))
                continue
        except:
            pass

        pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)

        audio_file_in_processing_format = target_dir / "full.mp3"
        audio_info = prepare_audio_for_processing(audio_file, audio_file_in_processing_format)
        if audio_info is not None:
            print(f"Audio input duration: {audio_info['duration']} secs.")
            audio_file = str(audio_file_in_processing_format)
        else:
            # this should not happen unless ffpeobe somehow fails to identify
            # the input audio source properties
            print("Warning - Skipping audio format preperation - Working directly with the source")

        data = read_audio(audio_file, sampling_rate=SAMPLING_RATE)

        speech_timestamps = get_speech_timestamps(
            data,
            model,
            sampling_rate=SAMPLING_RATE,
            min_speech_duration_ms=config["min_speech_duration_ms"],
            max_speech_duration_s=config["max_speech_duration_s"],
            min_silence_duration_ms=config["min_silence_duration_ms"],
            speech_pad_ms=config["speech_pad_ms"],
            threshold=config["speech_threshold"],
        )

        canonical_splits = [
            (split["start"] / SAMPLING_RATE, split["end"] / SAMPLING_RATE) for split in speech_timestamps
        ]

        store_splits(audio_file, canonical_splits, target_dir)

        desc = {"source": source, "episode": episode, "splits": canonical_splits}
        desc_filename = os.path.join(target_dir, "desc.json")

        json.dump(desc, open(desc_filename, "w"), indent=2)


def store_splits(filename, splits, target_dir):
    print("Storing audio split files...")
    escaped_filename = filename.replace('"', '\\"')
    ffmpeg_cmd_base = f'ffmpeg -y -i "{escaped_filename}"'
    ffmpeg_cmd = ffmpeg_cmd_base

    elems = 0

    for idx, split in enumerate(splits):
        (start, end) = split

        escaped_target_dir = target_dir.as_posix().replace('"', '\\"')
        ffmpeg_cmd += f' -vn -c copy -ss {start} -to {end} "{escaped_target_dir}/{idx}.mp3"'

        elems += 1

        if elems == 200:
            os.system(ffmpeg_cmd)
            ffmpeg_cmd = ffmpeg_cmd_base
            elems = 0

    if elems > 0:
        os.system(ffmpeg_cmd)


if __name__ == "__main__":
    # Define an argument parser
    parser = argparse.ArgumentParser(
        description="Search for audio files in a directory excluding specified subdirectories"
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
    parser.add_argument(
        "--target-dir",
        type=str,
        required=True,
        help="The directory where splitted audios will be stored.",
    )
    parser.add_argument("--jobs", type=int, required=False, default=3, help="Allow N jobs at once.")
    parser.add_argument(
        "--force-reprocess",
        required=False,
        action="store_true",
        help="Force processing even if already done",
    )
    parser.add_argument(
        "--min_speech_duration_ms",
        type=int,
        required=False,
        default=2000,
        help="VAD param: min_speech_duration_ms",
    )
    parser.add_argument(
        "--max_speech_duration_s",
        type=int,
        required=False,
        default=29,
        help="VAD param: max_speech_duration_s",
    )
    parser.add_argument(
        "--min_silence_duration_ms",
        type=int,
        required=False,
        default=300,
        help="VAD param: min_silence_duration_ms",
    )
    parser.add_argument(
        "--speech_pad_ms",
        type=int,
        required=False,
        default=30,
        help="VAD param: speech_pad_ms",
    )
    parser.add_argument(
        "--speech_threshold",
        type=float,
        required=False,
        default=0.5,
        help="VAD param: threshold (for speech detection)",
    )

    # Parse the arguments
    args = parser.parse_args()

    bulk_vad(args)
