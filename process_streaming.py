#!/usr/bin/python3

import argparse
import json
import os
import pathlib

import torch
import torchaudio
from torch.multiprocessing import Pool
import torch.multiprocessing.spawn
import torch.multiprocessing.spawn
from tqdm import tqdm

import utils

# Using more than 1 thread appears to actually make VAD slower.
# Using a single thread, and forking to run multiple processes.
#
# Yair, July 2023
torch.set_num_threads(1)

SAMPLING_RATE = 16000


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


# How close a speech needs to be to the end of a segment to be considered
# as potentially overlapping with the next segment.
audio_overlap_detection_threshold_seconds = 10


def bulk_vad_single_process(config, group_idx, audio_files, model, torch_utils):
    (get_speech_timestamps, _, _, _, _) = torch_utils

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

        audio_file_info = torchaudio.info(audio_file)
        total_audio_frames = audio_file_info.num_frames
        native_sample_rate = audio_file_info.sample_rate
        audio_file_duration = total_audio_frames // native_sample_rate
        print(f"Audio file total duration: {audio_file_duration} seconds")

        process_in_segments = config["segment_audio"]
        segment_length_seconds = config["audio_segment_length_s"] if process_in_segments else audio_file_duration
        segment_length_frames = segment_length_seconds * native_sample_rate

        next_segment_start_frame = 0
        next_segment_end_frame = next_segment_start_frame + segment_length_frames

        all_speech_timestamps = []
        while next_segment_start_frame < total_audio_frames:
            time_at_segment_start = next_segment_start_frame / native_sample_rate
            time_at_segment_end = next_segment_end_frame / native_sample_rate
            print(
                f"Processing segment starting at {time_at_segment_start} seconds ending at {time_at_segment_end} seconds"
            )

            # Load the data for the processed segment
            frames_to_load_for_segment = next_segment_end_frame - next_segment_start_frame
            wav, sr = torchaudio.load(
                audio_file,
                frame_offset=next_segment_start_frame,
                num_frames=frames_to_load_for_segment,
            )

            if wav.size(0) > 1:
                wav = wav.mean(dim=0, keepdim=True)

            if sr != SAMPLING_RATE:
                try:
                    transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLING_RATE)
                except Exception as e:
                    print(e)
                    raise e
                wav = transform(wav.clone().detach())
                sr = SAMPLING_RATE

            speech_timestamps = get_speech_timestamps(
                wav,
                model,
                sampling_rate=SAMPLING_RATE,
                min_speech_duration_ms=config["min_speech_duration_ms"],
                max_speech_duration_s=config["max_speech_duration_s"],
                min_silence_duration_ms=config["min_silence_duration_ms"],
                speech_pad_ms=config["speech_pad_ms"],
                threshold=config["speech_threshold"],
                return_seconds=True,
            )

            # rebase timestamps to absolute start of audio file
            speech_timestamps = [
                {
                    "start": s["start"] + time_at_segment_start,
                    "end": s["end"] + time_at_segment_start,
                }
                for s in speech_timestamps
            ]

            # If this segment had speech parts within it - the last segment
            # might overflow onto the next segment.
            # we will load the next segment with overlap to this one to make sure we capture
            # it's entirety
            next_segment_start_frame += frames_to_load_for_segment
            is_last_segment = next_segment_end_frame >= total_audio_frames
            # We set the next end frame now - so if next start needs to move back
            # to perform an overlap - we will still reach forward enough to see new data
            next_segment_end_frame = min(next_segment_start_frame + segment_length_frames, total_audio_frames)

            if len(speech_timestamps) > 0 and not is_last_segment:
                last_found_speech = speech_timestamps[-1]
                last_found_speech_end_seconds = last_found_speech["end"]
                last_found_speech_end_time_to_segment_end = time_at_segment_end - last_found_speech_end_seconds
                if last_found_speech_end_time_to_segment_end < audio_overlap_detection_threshold_seconds:
                    last_found_speech_start_seconds = last_found_speech["start"]
                    last_found_speech_start_time_to_segment_end = time_at_segment_end - last_found_speech_start_seconds
                    frames_to_overlap = last_found_speech_start_time_to_segment_end * native_sample_rate
                    next_segment_start_frame -= frames_to_overlap
                    # remove the last speeach timestamp from the list
                    # since it will be included in the next segment
                    speech_timestamps = speech_timestamps[:-1]

            all_speech_timestamps.extend(speech_timestamps)

        pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)

        canonical_splits = [(split["start"], split["end"]) for split in all_speech_timestamps]

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
        "--process-audio-in-segments",
        required=False,
        action="store_true",
        help="Process audio files in segments and not in full",
    )
    parser.add_argument(
        "--audio_segment_length_s",
        type=int,
        required=False,
        default=1800,
        help="Length of audio segments to split by (reduce memory overhead)",
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
