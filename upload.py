#!/usr/bin/python3

import argparse
import os

import datasets
import utils


def create_dataset(mp3s):
    episodes = []
    sources = []
    uuids = []
    attrs = []

    for mp3 in mp3s:
        episode = os.path.splitext(os.path.basename(mp3))[0]
        source = os.path.basename(os.path.dirname(mp3))
        uuid = f'{source}/{episode}'

        episodes.append(episode)
        sources.append(source)
        uuids.append(uuid)
        attrs.append({'license' : 'v1'})

    ds = datasets.Dataset.from_dict({
        'audio' : mp3s,
        'episode' : episodes,
        'source' : sources,
        'uuid' : uuids,
        'attrs' : attrs
    })
    
    ds = ds.cast_column('audio', datasets.Audio())

    return ds

def initialize_dataset(args):
    # Iterate over each root directory
    mp3s = utils.find_mp3_files(args.root_dir, args.skip_dir)

    ds = create_dataset(mp3s)

    ds.push_to_hub(repo_id=args.dataset, token=args.hf_token)

if __name__ == '__main__':
    # Define an argument parser
    parser = argparse.ArgumentParser(description='Create a dataset and upload to Huggingface.')

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

    initialize_dataset(args)
