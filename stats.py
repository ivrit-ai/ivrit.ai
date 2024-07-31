#!/usr/bin/python3

import argparse
import json
import pathlib

from import utils

def collect_statistics(args):
    # Iterate over each root directory
    descs = utils.find_files(args.root_dir, args.skip_dir, ['.json'])

    stats = {}

    for desc_fn in descs:
        print(desc_fn)
        desc_fn = pathlib.Path(desc_fn)
        desc = json.load(open(desc_fn))

        source = desc_fn.parent.parent.name
        episode = desc_fn.parent.name 

        if not source in stats:
            stats[source] = { 'duration' : 0, 'durations' : [], 'episodes' : 0, 'segments' : 0 }

        stats[source]['duration'] += desc['splits'][-1][1]
        stats[source]['episodes'] += 1

        for split in desc['splits']:
            duration = split[1] - split[0]
            stats[source]['durations'].append(split[1] - split[0])
            stats[source]['segments'] += 1

    json.dump(stats, open('stats.json', 'w'), indent=2)

if __name__ == '__main__':
    # Define an argument parser
    parser = argparse.ArgumentParser(description='Create a dataset and upload to Huggingface.')

    # Add the arguments
    parser.add_argument('--root-dir', action='append', required=True,
                        help='Root directory to start search from. Can be passed multiple times.')
    parser.add_argument('--skip-dir', action='append', required=False, default=[],
                        help='Directories to skip. Can be passed multiple times.')
    parser.add_argument('--splits', action='store_true', help='This is a splits dataset.')

    # Parse the arguments
    args = parser.parse_args()

    collect_statistics(args)
