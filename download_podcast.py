#!/usr/bin/python3

import argparse
import feedparser
import requests
from datetime import datetime
import pydub

import os
import pathlib
import re
import shlex
import sys

def download_podcast(feed, target_dir):
    target_dir = pathlib.Path(target_dir)
    feed_target = target_dir / 'feed.xml'

    if pathlib.Path(feed).exists():
        feed_target = feed
    else: 
        download_file(feed, feed_target)
    
    feed = feedparser.parse(feed_target)

    for entry in reversed(feed.entries):
        title = entry.get('title', '')
        date = entry.get('published', '')
        
        download_link, download_len, file_type = extract_download_link(entry.links)
        print(f'Raw download link: {download_link}')
        
        try:
            date_obj = datetime.strptime(date, '%a, %d %b %Y %H:%M:%S %Z')
        except:
            date_obj = datetime.strptime(date, '%a, %d %b %Y %H:%M:%S %z')

        formatted_date = date_obj.strftime('%Y.%m.%d')

        fn = f'{formatted_date} {title}'
        fn = re.sub(r'[\\/*?:"<>|]', '', fn)
        fn = trim_filename(fn)
        fn = target_dir / f'{fn}.{file_type}'
    
        fn_mp3 = f'{formatted_date} {title}'
        fn_mp3 = re.sub(r'[\\/*?:"<>|]', '', fn_mp3)
        fn_mp3 = trim_filename(fn_mp3)
        fn_mp3 = target_dir / f'{fn_mp3}.mp3'
    
        print(f'Downloading {fn}...')

        if not os.path.exists(fn) and not os.path.exists(fn_mp3):
            download_file(download_link, fn)

            if file_type != 'mp3':
                pydub.AudioSegment.from_file(fn).export(fn_mp3, format='mp3')
                os.unlink(fn)
   
    print('Done.')

def download_file(url, target):
    tmp_target = f'{target}.tmp'

    status = 0

    NUM_RETRIES = 3
    for i in range(NUM_RETRIES):
        print(f'Download attempt #{i+1}...')
        status = os.system(f'wget {url} -O {shlex.quote(tmp_target)}')
        if status == 0:
            break
 
    if status != 0:
        sys.exit(-1)
 
    if os.stat(tmp_target).st_size != 0:
        os.rename(tmp_target, target)
 
    
def extract_download_link(links):
    download_link = None

    for link in links:
        if link.rel != 'enclosure':
            continue

        if download_link:
            print('Multiple download links detected. Exiting.')
            sys.exit(1)

        download_link = link.href
        download_len = int(link.length)
                
        if link.type == 'audio/mpeg':
            file_type = 'mp3'
        elif link.type == 'audio/x-m4a':
            file_type = 'm4a'
        else:
            print('Unknown download type detected. Exiting.')
            sys.exit(2)
            
        if not download_link:
            print(links)
            print('No download link detected. Exiting.')
            sys.exit(2)

    return download_link, download_len, file_type

def trim_filename(fn):
    MAX_FILENAME_LEN = 240
    while len(fn.encode('utf-8')) > MAX_FILENAME_LEN:
        fn = fn[0:-1]

    return fn

if __name__ == '__main__':
    # Define an argument parser
    parser = argparse.ArgumentParser(description='''Download a podcast RSS file and episodes.''')

    # Add the arguments
    parser.add_argument('--feed', type=str, required=True,
                        help='RSS feed.')
    parser.add_argument('--target-dir', type=str, required=True,
                        help='The directory where episodes will be stored.')

    # Parse the arguments
    args = parser.parse_args()

    download_podcast(args.feed, args.target_dir)
