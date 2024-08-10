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
import subprocess
import sys


def download_podcast(feed, target_dir):
    target_dir = pathlib.Path(target_dir)
    feed_target = target_dir / "feed.xml"

    if pathlib.Path(feed).exists():
        feed_target = feed
    else:
        download_file(feed, feed_target)

    feed = feedparser.parse(feed_target)

    for entry in reversed(feed.entries):
        title = entry.get("title", "")
        date = entry.get("published", "")

        download_link, download_len, file_type = extract_download_link(entry.links)
        print(f"Raw download link: {download_link}")

        try:
            date_obj = datetime.strptime(date, "%a, %d %b %Y %H:%M:%S %Z")
        except:
            date_obj = datetime.strptime(date, "%a, %d %b %Y %H:%M:%S %z")

        formatted_date = date_obj.strftime("%Y.%m.%d")

        raw_fn = f"{formatted_date} {title}"
        raw_fn = re.sub(r'[\\/*?:"<>|]', "", raw_fn)
        raw_fn = trim_filename(raw_fn)

        fn = target_dir / f"{raw_fn}.{file_type}"
        fn_mp3 = target_dir / f"{raw_fn}.mp3"
        fn_done = target_dir / f"{raw_fn}.done"

        if not fn_done.exists():
            print(f"Downloading {fn}...")

            try:
                download_file(download_link, fn)
            except yt_dlp.utils.DownloadError:
                print(f"Skipping {download_link} as yt_dlp is unable to access it.")
                continue

            if file_type != "mp3":
                # This code used pydub in the past.
                # pydub fails conversion for some cases, while ffmpeg plays nice for a wider range of inputs.
                os.system(f"ffmpeg -i {shlex.quote(str(fn))} {shlex.quote(str(fn_mp3))}")
                os.unlink(fn)

            fn_done.touch()

    print("Done.")


def is_youtube_url(url):
    return url.startswith("https://www.youtube.com")


def download_file(url, target):
    status = 0

    NUM_RETRIES = 3
    for i in range(NUM_RETRIES):
        print(f"Download attempt #{i+1}...")

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
        "simulate": True,  # Do not download the video files
        "geturl": True,  # Print final URL to stdout
        "listsubs": True,  # Get list of subtitles
    }

    # Fetch the URL of the video/audio using youtube_dl
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=False)
        video_url = info_dict.get("url", None)

        subtitles = info_dict.get("subtitles", {}).get("iw", {})

        pd = pathlib.Path(target)
        pd.parent.mkdir(parents=True, exist_ok=True)

        if not subtitles == {}:
            print(f'Has hebrew subtitles; duration={info_dict["duration"]}')
            vtt_url = None

            for st in info_dict["subtitles"]["iw"]:
                if st["ext"] != "vtt":
                    continue

                vtt_url = st["url"]
                vtt_fn = target.parent / f"{target.stem}.vtt"

            if not vtt_url:
                print(f"No VTT found. Subtitles: {subtitles}")
                sys.exit(-1)

            os.system(f"""wget '{vtt_url}' -O {shlex.quote(str(vtt_fn))}""")

    if video_url:
        cmd = f"""aria2c -x 16 -s 16 -k 1M -q '{video_url}' -d / -o {shlex.quote(str(target))}"""
        os.system(cmd)

        return 0
    else:
        return -1


def extract_download_link(links):
    download_link = None

    for link in links:
        if link.rel != "enclosure":
            continue

        if download_link:
            print("Multiple download links detected. Exiting.")
            sys.exit(1)

        download_link = link.href
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

        if not download_link:
            print(links)
            print("No download link detected. Exiting.")
            sys.exit(2)

    return download_link, download_len, file_type


def trim_filename(fn):
    MAX_FILENAME_LEN = 240
    while len(fn.encode("utf-8")) > MAX_FILENAME_LEN:
        fn = fn[0:-1]

    return fn


if __name__ == "__main__":
    # Define an argument parser
    parser = argparse.ArgumentParser(description="""Download a podcast RSS file and episodes.""")

    # Add the arguments
    parser.add_argument("--feed", type=str, required=True, help="RSS feed.")
    parser.add_argument("--target-dir", type=str, required=True, help="The directory where episodes will be stored.")

    # Parse the arguments
    args = parser.parse_args()

    download_podcast(args.feed, args.target_dir)
