#!/usr/bin/python3

import argparse
import json
import pathlib

import utils

import datasets
import torch
from transformers import pipeline

ckpt = "openai/whisper-small"
lang = "he"
device = 0 if torch.cuda.is_available() else "cpu"

pipe = pipeline(
    task="automatic-speech-recognition",
    model=ckpt,
    chunk_length_s=60,
    device=device,
)

pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(language=lang, task="transcribe")

def iterate_data(dataset):
    for i, item in enumerate(dataset):
        yield item["audio"]

BATCH_SIZE = 64


def transcribe(args):
    # Iterate over each root directory
    mp3s = [mp3 for mp3 in utils.find_files(args.root_dir, args.skip_dir, ['.mp3']) if not_processed(mp3)]

    num_files = len(mp3s)

    audio_dataset = datasets.Dataset.from_dict({"audio": mp3s}).cast_column("audio", datasets.Audio())

    for idx, out in enumerate(pipe(iterate_data(audio_dataset), batch_size=BATCH_SIZE)):
        mp3 = audio_dataset[idx]['audio']['path']
        print(f'Evaluating #{idx+1}/{num_files}: {mp3}')
        transcribe_single(mp3, args, out['text'])

def not_processed(mp3):
    mp3_fn = pathlib.Path(mp3)

    source = mp3_fn.parent.parent.name
    episode = mp3_fn.parent.name
    target_dir = pathlib.Path(args.target_dir) / source / episode
    pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)

    json_fn = target_dir / f'{mp3_fn.stem}.json'

    try:
        desc = json.load(open(json_fn, 'r'))
        return False
    except:
        pass

    return True

def transcribe_single(mp3, args, text):
    mp3_fn = pathlib.Path(mp3)

    source = mp3_fn.parent.parent.name
    episode = mp3_fn.parent.name
    target_dir = pathlib.Path(args.target_dir) / source / episode
    pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)

    json_fn = target_dir / f'{mp3_fn.stem}.json'

    transcript = text

    json.dump({'source' : source, 'episode' : episode, 'segment' : mp3_fn.stem, 'text' : transcript, 'transcript_source' : 'asr', 'raw' : transcript}, open(json_fn, 'w'))

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

