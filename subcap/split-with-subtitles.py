#!/usr/bin/python3

import argparse
import json
import os
import pathlib
import sys
import tempfile

import datasets
import pydub
import subcap
import utils

MAX_SEGMENT_DURATION = 29.9

def split_sources(root_dir, target_dir):
    sources = utils.find_files([root_dir], [], ['.mp3', '.mp4', '.wmv'])

    subtitled_duration = 0
    skipped_duration = 0

    for idx, s in enumerate(sources):
        #print(f'{idx}/{len(sources)}: {s}')
        media_file = pathlib.Path(s)        
        subcap_file = pathlib.Path(f'{media_file.parent / media_file.stem}_subCap.txt')

        #duration = pydub.AudioSegment.from_file(media_file).duration_seconds
        duration = 0

        if not subcap_file.exists():
            print(f'Error: subcap file does not exist; media_file={media_file} subcap_file={subcap_file}')
            skipped_duration += duration

            continue

        subtitled_duration += duration

        dest_dir = pathlib.Path(target_dir) / (media_file.parent / media_file.stem).relative_to(root_dir)

        split_media_with_subcaps(media_file, subcap_file, dest_dir) 
 

    print(f'Subtitled: {subtitled_duration}')
    print(f'Skipped: {skipped_duration}')

def split_media_with_subcaps(media_file, subcap_file, dest_dir):
    desc_file = dest_dir / 'desc.json'
    if desc_file.exists():
        return

    dest_dir.mkdir(parents=True, exist_ok=True)

    mf = pydub.AudioSegment.from_file(media_file)
    subcaps = subcap.parse_subcap(subcap_file)

    if len(subcaps) == 0:
        return

    merged_subcaps = []

    curr_start_seg = subcaps[0]
    curr_end_seg = subcaps[0]
    texts = [curr_start_seg['text']]

    for segment in subcaps[1:]:
        if segment['end_time'] - curr_start_seg['start_time'] < MAX_SEGMENT_DURATION:
            curr_end_seg = segment
            texts.append(segment['text'])

            continue

        merged_subcap = { 'start_time' : curr_start_seg['start_time'],
                          'end_time' : curr_end_seg['end_time'],
                          'text' : ' '.join(texts) }
        merged_subcaps.append(merged_subcap)

        curr_start_seg = segment
        curr_end_seg = segment
        texts = [curr_start_seg['text']] 


    merged_subcap = { 'start_time' : curr_start_seg['start_time'],
                      'end_time' : curr_end_seg['end_time'],
                      'text' : ' '.join(texts) }
    merged_subcaps.append(merged_subcap)

    for idx, segment in enumerate(merged_subcaps):
        mf[int(segment['start_time']*1000):int(segment['end_time']*1000)].export(f'{dest_dir}/{idx}.mp3', format='mp3', bitrate='16k')

    json.dump(merged_subcaps, open(desc_file, 'w'))


def generate_dataset(ds_name, splits_dir):
    descs = utils.find_files([splits_dir], [], ['.json'])

    uuids = []
    mp3_files = []
    start_times = []
    end_times = [] 
    sentences = []

    for d in descs:
        print(f'Processing {d}...')
        desc = json.load(open(d, 'r'))

        desc_dir = pathlib.Path(d).parent
        relative_desc_dir = desc_dir.relative_to(splits_dir)
        for idx, seg in enumerate(desc):
            uuids.append(str(relative_desc_dir / f'{idx}'))
            mp3_files.append(str(desc_dir / f'{idx}.mp3'))
            start_times.append(seg['start_time'])
            end_times.append(seg['end_time'])
            sentences.append(seg['text'])

    ds = datasets.Dataset.from_dict({
        'uuid' : uuids,
        'audio' : mp3_files,
        'sentence' : sentences,
        'start' : start_times,
        'end' : end_times
    })

    ds = ds.cast_column('audio', datasets.Audio())

    ds.save_to_disk(ds_name)

if __name__ == '__main__':
    # Define an argument parser
    parser = argparse.ArgumentParser(description='Generate Huggingface dataset from video/audio data with subtitles.')

    # Add the arguments
    parser.add_argument('--root-dir', type=str, required=True,
                        help='Root directory to start search from.')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name.')
    parser.add_argument('--target-dir', type=str, required=True,
                        help='Target directory.')

    # Parse the arguments
    args = parser.parse_args()

    split_sources(args.root_dir, args.target_dir)

    generate_dataset(args.dataset, args.target_dir)
