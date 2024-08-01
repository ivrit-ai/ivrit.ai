#!/usr/bin/python3

import argparse
import json
import os
import pathlib
import sys

import torch
import torchaudio

from utils import utils

# Using more than 1 thread appears to actually make VAD slower.
# Using a single thread, and forking to run multiple processes.
#
# Yair, July 2023
torch.set_num_threads(1)

SAMPLING_RATE = 16000

def bulk_vad(args):
    if args.jobs:
        NUM_PROCESSES = args.jobs

    model, torch_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')

    audio_files = utils.find_files(args.root_dir, args.skip_dir, ['.mp3', '.m4a'])
    groups = [audio_files[i::NUM_PROCESSES] for i in range(NUM_PROCESSES)] 

    children_pids = []
    for group_idx, group in enumerate(groups):
        pid = os.fork()

        if pid:
            children_pids.append(pid)
        else:
            bulk_vad_single_process(group_idx, group, model, torch_utils)
            sys.exit(0)

    for pid in children_pids:
        os.waitpid(pid, 0)

def bulk_vad_single_process(group_idx, audio_files, model, torch_utils):
    (get_speech_timestamps, _, read_audio, _, _) = torch_utils

    for idx, audio_file in enumerate(audio_files):
        print(f'Processing group {group_idx}, file #{idx}: {audio_file}')

        audio_path = pathlib.Path(audio_file)
        source = audio_path.parent.name
        episode = audio_path.stem
        target_dir = pathlib.Path(args.target_dir) / source / episode

        # Check if file was already processed successfully.
        try:
            desc_filename = target_dir / 'desc.json'
            if desc_filename.is_file():
                json.load(open(desc_filename, 'r'))
                continue
        except:
            pass 

        data = read_audio(audio_file, sampling_rate=SAMPLING_RATE)

        speech_timestamps = get_speech_timestamps(data, model, sampling_rate=SAMPLING_RATE, min_speech_duration_ms=2000, max_speech_duration_s=29, min_silence_duration_ms=300)

        canonical_splits = [(split['start'] / SAMPLING_RATE, split['end'] / SAMPLING_RATE) for split in speech_timestamps]

        pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)

        store_splits(audio_file, canonical_splits, target_dir, source, episode)

        desc = { 'source' : source, 'episode' : episode, 'splits' : canonical_splits}
        desc_filename = os.path.join(target_dir, 'desc.json')

        json.dump(desc, open(desc_filename, 'w'), indent=2)

def store_splits(filename, splits, target_dir, source, episode):
    escaped_filename = filename.replace('"', '\\"')
    ffmpeg_cmd_base = f'ffmpeg -i "{escaped_filename}"'
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



if __name__ == '__main__':
    # Define an argument parser
    parser = argparse.ArgumentParser(description='Search for audio files in a directory excluding specified subdirectories')

    # Add the arguments
    parser.add_argument('--root-dir', action='append', required=True,
                        help='Root directory to start search from. Can be passed multiple times.')
    parser.add_argument('--skip-dir', action='append', required=False, default=[],
                        help='Directories to skip. Can be passed multiple times.')
    parser.add_argument('--target-dir', type=str, required=True,
                        help='The directory where splitted audios will be stored.')
    parser.add_argument('--jobs', type=int, required=False, default=3,
                        help='Allow N jobs at once.')

    # Parse the arguments
    args = parser.parse_args()

    bulk_vad(args)
