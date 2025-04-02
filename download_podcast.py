#!/usr/bin/python3

import argparse
import feedparser
import http.client
import requests
from datetime import datetime
import yt_dlp

import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
from typing import Set, Tuple
from tqdm import tqdm


def download_podcasts(feeds: list, target_dir: str):
    target_dir = pathlib.Path(target_dir)
    downloaded_entries: Set[Tuple[str, str]] = set()  # (title, date) tuples to track duplicates
    
    # Download and save all feed files with sequential numbering
    feed_contents = []
    for i, feed in enumerate(tqdm(feeds, desc="Downloading feeds", unit="feed"), 1):
        feed_target = target_dir / f"feed_{i}.xml"
        
        if pathlib.Path(feed).exists():
            shutil.copy(feed, feed_target)
        else:
            download_file(feed, feed_target)
            
        parsed_feed = feedparser.parse(feed_target)
        feed_contents.append(parsed_feed)

    # Process all feeds
    total_entries = sum(len(feed.entries) for feed in feed_contents)
    with tqdm(total=total_entries, desc="Processing entries", unit="entry") as pbar:
        for feed in feed_contents:
            for entry in reversed(feed.entries):
                title = entry.get("title", "")
                date = entry.get("published", "")
                
                # Create a unique identifier for this entry
                entry_id = (title, date)
                if entry_id in downloaded_entries:
                    pbar.write(f"Skipping duplicate entry: {title}")
                    pbar.update(1)
                    continue
                    
                downloaded_entries.add(entry_id)

                download_link, download_len, file_type = extract_download_link(entry.links)
                pbar.write(f"Raw download link: {download_link}")

                try:
                    date_obj = datetime.strptime(date, "%a, %d %b %Y %H:%M:%S %Z")
                except:
                    date_obj = datetime.strptime(date, "%a, %d %b %Y %H:%M:%S %z")

                formatted_date = date_obj.strftime("%Y.%m.%d")

                raw_fn = f"{formatted_date} {title}"
                raw_fn = re.sub(r'[\\/*?:"<>|]', "", raw_fn)
                raw_fn = re.sub(r"[\u2013\u2014]", "-", raw_fn)
                raw_fn = trim_filename(raw_fn)

                fn = target_dir / f"{raw_fn}.{file_type}"
                fn_m4a = target_dir / f"{raw_fn}.m4a"
                fn_done = target_dir / f"{raw_fn}.done"

                if not fn_done.exists():
                    pbar.write(f"Downloading {fn}...")
                    pbar.write(f"link: {download_link}")
                    try:
                        download_file(download_link, fn)
                    except yt_dlp.utils.DownloadError:
                        pbar.write(f"Skipping {download_link} as yt_dlp is unable to access it.")
                        pbar.update(1)
                        continue

                    if file_type != "m4a":
                        # This code used pydub in the past.
                        # pydub fails conversion for some cases, while ffmpeg plays nice for a wider range of inputs.
                        os.system(f"ffmpeg -loglevel error -hide_banner -stats -i {shlex.quote(str(fn))} -c:a aac -q:a 0 {shlex.quote(str(fn_mp3))}")
                        os.unlink(fn)

                    fn_done.touch()

                pbar.update(1)


def is_youtube_url(url):
    return url.startswith("https://www.youtube.com")


def download_file(url, target):
    status = 0

    NUM_RETRIES = 3
    for i in range(NUM_RETRIES):
        if is_youtube_url(url):
            status = download_youtube_video(url, target)
        else:
            status = os.system(f"wget {shlex.quote(url)} -O {shlex.quote(str(target))}")

        if not target.exists():
            continue

        if status == 0:
            break

    if status != 0:
        sys.exit(-1)

    if target.exists() and target.stat().st_size == 0:
        target.unlink()


def download_youtube_video(url, target):
    # Options for youtube_dl: fetch URL and metadata without downloading the actual video
    ydl_opts = {
        "format": "bestaudio/best",  # Get the best audio or video file
        "quiet": True,  # Do not print messages to stdout
        "simulate": False,  # Do not download the video files
        "geturl": True,  # Print final URL to stdout
        "writesubtitles": True,  # Download subtitles
        "subtitleslangs": ["iw"],  # Download Hebrew subtitles
        "subtitlesformat": "vtt/srt/best",  # Prefer VTT, then SRT, then best available
        "outtmpl": str(target),
    }

    # Fetch the URL of the video/audio using youtube_dl
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)


    return 0


def extract_download_link(links):
    download_link = None
    download_len = None
    file_type = None

    for link in links:
        if link.rel != "enclosure":
            continue

        if download_link:
            print("Multiple download links detected. Exiting.")
            sys.exit(1)

        download_link = link.href
        if not download_link:
            print(links)
            print("No download link (link.href) detected. Exiting.")
            sys.exit(2)

        download_len = int(link.length)

        if link.type == "audio/mpeg":
            file_type = "mp3"
        elif link.type == "audio/x-m4a":
            file_type = "m4a"
        elif link.type == "video/mp4":
            file_type = "m4a"
        else:
            print("Unknown download type detected. Exiting.")
            sys.exit(2)

    if not download_link or download_len == None or not file_type:
        print(links)
        print("No download links found. Exiting.")
        sys.exit(2)

    return download_link, download_len, file_type


def trim_filename(fn):
    MAX_FILENAME_LEN = 240
    while len(fn.encode("utf-8")) > MAX_FILENAME_LEN:
        fn = fn[0:-1]

    return fn


if __name__ == "__main__":
    # Define an argument parser
    parser = argparse.ArgumentParser(description="""Download podcast RSS files and episodes.""")

    # Add the arguments
    parser.add_argument("--feeds", type=str, nargs="+", required=True, help="List of RSS feeds.")
    parser.add_argument("--target-dir", type=str, required=True, help="The directory where episodes will be stored.")

    # Parse the arguments
    args = parser.parse_args()

    download_podcasts(args.feeds, args.target_dir)
