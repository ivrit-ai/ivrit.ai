#!/usr/bin/python3

import argparse
import json
import pathlib

import utils

import whisper

model = whisper.load_model('small')

def transcribe(args):
    # Iterate over each root directory
    mp3s = utils.find_files(args.root_dir, args.skip_dir, ['.mp3'])
    num_files = len(mp3s)

    for idx, mp3 in enumerate(mp3s):
        print(f'Evaluating #{idx+1}/{num_files}: {mp3}')
        transcribe_single(mp3, args)

def transcribe_single(mp3, args):
    mp3_fn = pathlib.Path(mp3)

    source = mp3_fn.parent.parent.name
    episode = mp3_fn.parent.name
    target_dir = pathlib.Path(args.target_dir) / source / episode
    pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)

    json_fn = target_dir / f'{mp3_fn.stem}.json'

    try:
        desc = json.load(open(json_fn, 'r'))
        print('Already transcribed, skipping.')
        return
    except:
        pass
    
    transcript = model.transcribe(str(mp3_fn), language='he')

    json.dump({'source' : source, 'episode' : episode, 'segment' : mp3_fn.stem, 'text' : transcript['text'], 'transcript_source' : 'asr', 'raw' : transcript}, open(json_fn, 'w'))

    print('Done transcribing.')
    


if __name__ == '__main__':
    # Define an argument parser
    parser = argparse.ArgumentParser(description='Transcribe a set of audio snippets generated using process.py files.')

    # Add the arguments
    parser.add_argument('--root-dir', action='append', required=True,
                        help='Root directory to start search from. Can be passed multiple times.')
    parser.add_argument('--skip-dir', action='append', required=False, default=[],
                        help='Directories to skip. Can be passed multiple times.')
    parser.add_argument('--target-dir', type=str, required=True,
                                    help='The directory where splitted audios will be stored.')

    # Parse the arguments
    args = parser.parse_args()

    transcribe(args)

