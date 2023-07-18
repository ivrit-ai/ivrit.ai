#!/usr/bin/python3

import argparse
import json
import pathlib

import datasets
import utils

import mutagen.mp3

def create_dataset(mp3s, is_splits):
    total_duration = 0

    episodes = []
    sources = []
    uuids = []
    attrs = []

    hashed_desc = None
    hashed_uuid = None

    for idx, mp3 in enumerate(mp3s):
        print(idx)

        mp3 = pathlib.Path(mp3) 

        if is_splits:
            episode = mp3.parent.name
            source = mp3.parent.parent.name
            segment = int(mp3.stem)
            uuid = f'{source}/{episode}/{segment}'

            if not hashed_uuid == uuid:
                hashed_desc = json.load(open(mp3.parent / 'desc.json'))
                hashed_uuid = uuid

            start = hashed_desc['splits'][segment][0]
            end = hashed_desc['splits'][segment][1]
            duration = end - start
        else:
            episode = mp3.stem
            source = mp3.parent.name
            uuid = f'{source}/{episode}'
            duration = mutagen.mp3.MP3(mp3).info.length

        total_duration += duration

        attr = {'license' : 'v1', 'duration' : duration} 
        if is_splits:
            attr['segment'] = segment
            attr['start'] = start
            attr['end'] = end 

        episodes.append(episode)
        sources.append(source)
        uuids.append(uuid)
        attrs.append(attr)

    print(f'Total dataset size is {total_duration:.2f} seconds.')

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
    mp3s = utils.find_files(args.root_dir, args.skip_dir, ['.mp3'])

    ds = create_dataset(mp3s, args.splits)

    # For now, uploading a dataset of >200GB tends to fail, so we generate a local DB
    # and upload it using git lfs.
    #
    # Yair, July 2023

    #ds.push_to_hub(repo_id=args.dataset, token=args.hf_token)
    ds.save_to_disk(args.dataset)

if __name__ == '__main__':
    # Define an argument parser
    parser = argparse.ArgumentParser(description='Create a dataset and upload to Huggingface.')

    # Add the arguments
    parser.add_argument('--root-dir', action='append', required=True,
                        help='Root directory to start search from. Can be passed multiple times.')
    parser.add_argument('--skip-dir', action='append', required=False, default=[],
                        help='Directories to skip. Can be passed multiple times.')
    parser.add_argument('--dataset', type=str, required=True, help='The dataset name to upload.')
    parser.add_argument('--hf-token', type=str, required=False, help='The HuggingFace token.')
    parser.add_argument('--splits', action='store_true', help='This is a splits dataset.')

    # Parse the arguments
    args = parser.parse_args()

    initialize_dataset(args)
