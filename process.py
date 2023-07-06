#!/usr/bin/python3

import argparse
import os
import utils

import torch

def bulk_vad(args):
    model, torch_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
    (get_speech_timestamps, _, read_audio, _, _) = torch_utils

    mp3s = utils.find_mp3_files(args.root_dir, args.skip_dir)

    for idx, mp3 in enumerate(mp3s):
        print(f'Processing {idx+1}: {mp3}...')
        data = read_audio(mp3, sampling_rate=16000)
        speech_timestamps = get_speech_timestamps(data, model, sampling_rate=16000)

if __name__ == '__main__':
    # Define an argument parser
    parser = argparse.ArgumentParser(description='Search for mp3 files in a directory excluding specified subdirectories')

    # Add the arguments
    parser.add_argument('--root-dir', action='append', required=True,
                        help='Root directory to start search from. Can be passed multiple times.')
    parser.add_argument('--skip-dir', action='append', required=False, default=[],
                        help='Directories to skip. Can be passed multiple times.')
    parser.add_argument('--dataset', type=str, required=True,
                        help='The dataset name to upload.')
    parser.add_argument('--hf-token', type=str, required=True,
                        help='The HuggingFace token.')

    # Parse the arguments
    args = parser.parse_args()

    bulk_vad(args)
