#!/usr/bin/python3

import argparse
import json
import os
import pathlib

import pydub
import torch

import utils

# Using more than 1 thread appears to actually make VAD slower.
# Using a single threads, and will run the script in parallel.
#
# Yair, July 2023
torch.set_num_threads(1)

SAMPLING_RATE = 16000

def bulk_vad(args):
    model, torch_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
    (get_speech_timestamps, _, read_audio, _, _) = torch_utils

    mp3s = utils.find_mp3_files(args.root_dir, args.skip_dir)

    for idx, mp3 in enumerate(mp3s):
        print(f'Processing {idx+1}: {mp3}...')

        mp3_path = pathlib.Path(mp3)
        source = mp3_path.parent.name
        episode = mp3_path.stem
        target_dir = pathlib.Path(args.target_dir) / source / episode

        # Check if file was already processed successfully.
        try:
            desc_filename = target_dir / 'desc.json'
            if desc_filename.is_file():
                json.load(open(desc_filename, 'r'))
                print(f'Skipping {mp3}...')
                continue
        except:
            pass 

        data = read_audio(mp3, sampling_rate=SAMPLING_RATE)
        speech_timestamps = get_speech_timestamps(data, model, sampling_rate=SAMPLING_RATE)

        canonical_splits = [(split['start'] / SAMPLING_RATE, split['end'] / SAMPLING_RATE) for split in speech_timestamps]

        store_splits(mp3, canonical_splits, target_dir, source, episode)

def store_splits(filename, splits, target_dir, source, episode):
    mp3 = pydub.AudioSegment.from_mp3(filename)

    pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)
 
    for idx, split in enumerate(splits):
        (start, end) = split
        start = int(start * 1000)
        end = int(end * 1000) 

        target_split = os.path.join(target_dir, f'{idx}.mp3')
        print(f'Generating {target_split}...')

        mp3[start:end].export(target_split, format='mp3')

    desc = { 'source' : source, 'episode' : episode, 'splits' : splits}
    desc_filename = os.path.join(target_dir, 'desc.json')

    json.dump(desc, open(desc_filename, 'w'), indent=2)


if __name__ == '__main__':
    # Define an argument parser
    parser = argparse.ArgumentParser(description='Search for mp3 files in a directory excluding specified subdirectories')

    # Add the arguments
    parser.add_argument('--root-dir', action='append', required=True,
                        help='Root directory to start search from. Can be passed multiple times.')
    parser.add_argument('--skip-dir', action='append', required=False, default=[],
                        help='Directories to skip. Can be passed multiple times.')
    parser.add_argument('--target-dir', type=str, required=True,
                        help='The directory where splitted audios will be stored.')

    # Parse the arguments
    args = parser.parse_args()

    bulk_vad(args)
