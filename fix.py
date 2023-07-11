#!/usr/bin/python3

import argparse
import pydub

if __name__ == '__main__':
    # Define an argument parser
    parser = argparse.ArgumentParser(description='Re-encode an mp3 file.')

    # Add the arguments
    parser.add_argument('--file', type=str, required=True,
                        help='File to reencode.')

    args = parser.parse_args()

    pydub.AudioSegment.from_mp3(args.file).export(args.file, format='mp3')
    print(f'Done fixing {args.file}.')

