#!/usr/bin/python3

import argparse
import json
import os
import pathlib
import sys
import tempfile

import datasets
import pydub
from utils import utils
import webvtt

MAX_SEGMENT_DURATION = 29.9

def vtt_time_to_seconds(timestamp):
    return (timestamp.hours * 3600 + timestamp.minutes * 60 + timestamp.seconds + timestamp.milliseconds * 0.001)

def split_sources(root_dir, target_dir):
    sources = utils.find_files([root_dir], [], ['.vtt'])

    subtitled_duration = 0
    skipped_duration = 0

    for idx, s in enumerate(sources):
        #print(f'{idx}/{len(sources)}: {s}')
        vtt_file = pathlib.Path(s) 
        media_file = pathlib.Path(f'{vtt_file.parent / vtt_file.stem}.mp3')

        #duration = pydub.AudioSegment.from_file(media_file).duration_seconds
        duration = 0

        if not media_file.exists():
            print(f'Error: media file does not exist; media_file={media_file} vtt_file={vtt_file}')
            skipped_duration += duration

            continue

        subtitled_duration += duration

        dest_dir = pathlib.Path(target_dir) / (media_file.parent / media_file.stem).relative_to(root_dir)

        split_media_with_subcaps(media_file, vtt_file, dest_dir) 
 

    print(f'Subtitled: {subtitled_duration}')
    print(f'Skipped: {skipped_duration}')

def split_media_with_subcaps(media_file, vtt_file, dest_dir):
    desc_file = dest_dir / 'desc.json'
    if desc_file.exists():
        return

    print(f'Processing {vtt_file}...')

    dest_dir.mkdir(parents=True, exist_ok=True)

    mf = pydub.AudioSegment.from_file(media_file)
    caps = webvtt.read_buffer(open(vtt_file))

    if len(caps) < 3:
        return

    caps = caps[0:len(caps) - 1]

    merged_caps = []

    curr_start_seg = caps[0]
    curr_end_seg = caps[0]
    texts = [curr_start_seg.text.replace('\n', ' ')]

    for segment in caps[1:]:
        start_time = vtt_time_to_seconds(curr_start_seg.start_time)
        end_time = vtt_time_to_seconds(segment.end_time)
        if end_time - start_time < MAX_SEGMENT_DURATION:
            curr_end_seg = segment
            texts.append(segment.text.replace('\n', ' '))

            continue

        merged_cap = { 'start_time' : start_time,
                       'end_time' : vtt_time_to_seconds(curr_end_seg.end_time),
                       'text' : ' '.join(texts) }
        merged_caps.append(merged_cap)

        curr_start_seg = segment
        curr_end_seg = segment
        texts = [curr_start_seg.text.replace('\n', ' ')] 

    merged_cap = { 'start_time' : start_time,
                   'end_time' : vtt_time_to_seconds(curr_end_seg.end_time),
                   'text' : ' '.join(texts) }
    merged_caps.append(merged_cap)

    for idx, segment in enumerate(merged_caps):
        mf[int(segment['start_time']*1000):int(segment['end_time']*1000)].export(f'{dest_dir}/{idx}.mp3', format='mp3', bitrate='16k')

    json.dump(merged_caps, open(desc_file, 'w'))


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
