#!/usr/bin/python3

import argparse
import json
import pathlib

import datasets
import utils

import mutagen.mp3

def create_transcripts_dataset(args):
    files = utils.find_files(args.root_dir, args.skip_dir, ['.json'])

    episodes = []
    sources = []
    uuids = []
    texts = []
    attrs = []

    #cached_desc = None
    #cached_uuid = None

    for idx, file in enumerate(files):
        print(f'{idx}/{len(files)}')
        transcript_desc = json.load(open(file))

        file = pathlib.Path(file) 

        episode = file.parent.name
        source = file.parent.parent.name
        segment = int(file.stem)
        uuid = f'{source}/{episode}/{segment}'
        text = transcript_desc['text']

        #if not cached_uuid == uuid:
        #    cached_desc = json.load(open(file.parent / 'desc.json'))
        #    cached_uuid = uuid

        #start = cached_desc['splits'][segment][0]
        #end = cached_desc['splits'][segment][1]
        #duration = end - start

        #attr = { 'segment' : segment, 'start' : start, 'end' : end, 'quality' : 1 }
        attr = { 'segment' : segment, 'quality' : 1 }

        episodes.append(episode)
        sources.append(source) 
        uuids.append(uuid)
        texts.append(text)
        attrs.append(attr)

    ds = datasets.Dataset.from_dict({
        'text' : texts,
        'episode' : episodes,
        'source' : sources,
        'uuid' : uuids,
        'attrs' : attrs
    })

    return ds


def create_audio_dataset(args):
    files = utils.find_files(args.root_dir, args.skip_dir, ['.mp3', '.mp4'])

    total_duration = 0

    episodes = []
    sources = []
    uuids = []
    attrs = []

    cached_desc = None
    cached_uuid = None

    for idx, file in enumerate(files):
        print(f'{idx}/{len(files)}')

        file = pathlib.Path(file) 

        if args.splits:
            episode = file.parent.name
            source = file.parent.parent.name
            segment = int(file.stem)
            uuid = f'{source}/{episode}/{segment}'

            if not cached_uuid == uuid:
                cached_desc = json.load(open(file.parent / 'desc.json'))
                cached_uuid = uuid

            start = cached_desc['splits'][segment][0]
            end = cached_desc['splits'][segment][1]
            duration = end - start
        else:
            episode = file.stem
            source = file.parent.name
            uuid = f'{source}/{episode}'
            duration = mutagen.mp3.MP3(file).info.length

        total_duration += duration

        attr = {'license' : 'v1', 'duration' : duration} 
        if args.splits:
            attr['segment'] = segment
            attr['start'] = start
            attr['end'] = end 

        episodes.append(episode)
        sources.append(source)
        uuids.append(uuid)
        attrs.append(attr)

    print(f'Total dataset size is {total_duration:.2f} seconds.')

    ds = datasets.Dataset.from_dict({
        'audio' : files,
        'episode' : episodes,
        'source' : sources,
        'uuid' : uuids,
        'attrs' : attrs
    })
    
    ds = ds.cast_column('audio', datasets.Audio())

    return ds

def initialize_dataset(args):
    # Iterate over each root directory
    if args.transcripts:
        ds = create_transcripts_dataset(args)
    else:
        ds = create_audio_dataset(args)


    # For now, uploading a dataset of >200GB tends to fail, so we generate a local DB
    # and upload it using git lfs.
    #
    # Yair, July 2023

    #ds.push_to_hub(repo_id=args.dataset, token=args.hf_token)
    ds.save_to_disk(args.dataset, num_shards=2000)

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
    parser.add_argument('--transcripts', action='store_true', help='This is a transcripts dataset.')

    # Parse the arguments
    args = parser.parse_args()

    initialize_dataset(args)
