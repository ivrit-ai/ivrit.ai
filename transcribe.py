#!/usr/bin/python3

import argparse
import base64
import json
import pathlib
import sys

import asyncio
import aiohttp

import utils

NUM_ELEMENTS_PER_BATCH = 50

async def fetch(session, url, data):
    headers = { 'Content-Type' : 'application/json' }
    async with session.post(url, data=json.dumps(data), headers=headers) as response:
        if not response.ok:
            print(response)

        return await response.json()

def transcribe(args):
    # Iterate over each root directory
    descs = utils.find_files(args.root_dir, args.skip_dir, ['.json'])

    for idx, _desc in enumerate(descs):
        print(f'Transcribing episode {idx}/{len(descs)}, {_desc}.')
        desc = pathlib.Path(_desc)
        asyncio.run(transcribe_single(desc, args))

url = 'http://0.0.0.0:4500/execute' 

async def transcribe_single(desc, args):
    source = desc.parent.parent.name
    episode = desc.parent.name
    target_dir = pathlib.Path(args.target_dir) / source / episode
    pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)

    json_fn = target_dir / f'transcripts.json'

    try:
        desc = json.load(open(json_fn, 'r'))
        print('Already transcribed, skipping.')
        return
    except:
        pass

    j = json.load(open(desc))

    num_segments = len(j['splits'])
    print(f'Total segments: {num_segments}.') 
  
    results = []

    for seg_base in range(0, num_segments, NUM_ELEMENTS_PER_BATCH):
        async with aiohttp.ClientSession() as session:
            tasks = []
            for seg_idx in range(seg_base, min(seg_base + NUM_ELEMENTS_PER_BATCH, num_segments)):
                seg_path = desc.parent / f'{seg_idx}.mp3'
                if not seg_path.exists():
                    print(f'Unable to find segment {seg_idx}. Exiting.')
                    sys.exit(-1)

                mp3_data = open(seg_path, 'rb').read()

                payload = {
                    'type': 'audio_processing',
                    'data': base64.b64encode(mp3_data).decode('utf-8'),
                    'token': '' 
                }

                tasks.append(fetch(session, url, payload))

            responses = await asyncio.gather(*tasks)
            for r in responses:
                results.append(r['result'])

    json.dump({'source' : source, 'episode' : episode, 'transcripts' : results}, open(json_fn, 'w'))

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

