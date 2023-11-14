#!/usr/bin/python3

import argparse
import base64
import json
import pathlib

import asyncio
import aiohttp

import utils

async def fetch(session, url, data):
    headers = { 'Content-Type' : 'application/json' }
    async with session.post(url, data=json.dumps(data), headers=headers) as response:
        return await response.json()

def transcribe(args):
    # Iterate over each root directory
    descs = utils.find_files(args.root_dir, args.skip_dir, ['.json'])

    for idx, _desc in enumerate(descs):
        print(f'Transcribing episode {idx}, {_desc}.')
        desc = pathlib.Path(_desc)
        asyncio.run(transcribe_single(desc, args))

url = 'http://127.0.0.1:5000/execute' 

async def transcribe_single(desc, args):
    j = json.load(open(desc))

    num_segments = len(j['splits'])
    print(f'Total segments: {num_segments}.') 
  
    blobs = []

    async with aiohttp.ClientSession() as session:
        tasks = []
        for seg_idx in range(num_segments):
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
            print(r['result']) 

        sys.exit(-1)

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

