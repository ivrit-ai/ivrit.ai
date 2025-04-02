#!/usr/bin/python3

from yt_dlp import YoutubeDL
from feedgen.feed import FeedGenerator
from datetime import datetime, timezone
import xml.dom.minidom
import argparse
from tqdm import tqdm
import os

def fetch_channel_info(channel_url, limit=None):
    """Fetch basic channel information and video URLs."""
    ydl_opts = {
        'extract_flat': True,  # Only fetch basic metadata
        'quiet': True,
        'ignoreerrors': True,
        'playlist_items': f'1-{limit}' if limit else None,
    }

    def collect_entries(entries):
        """Recursively collect entries from playlists."""
        video_urls = []
        for entry in entries:
            if not entry:
                continue
            if entry.get('_type') == 'playlist' and entry.get('entries'):
                # Recursively collect entries from nested playlists
                video_urls.extend(collect_entries(entry['entries']))
            elif entry.get('url'):
                video_urls.append({
                    'url': entry['url'],
                    'id': entry.get('id', '')
                })
        return video_urls

    with YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(channel_url, download=False)
            if '_type' in info:
                if info['_type'] == 'playlist':
                    channel_info = {
                        'title': info.get('title', 'YouTube Channel Feed'),
                        'description': info.get('description', 'Full feed of all videos from the channel'),
                        'id': info.get('id', 'unknown'),
                    }
                    # Extract video URLs and IDs from all playlists
                    video_urls = collect_entries(info.get('entries', []))
                    return channel_info, video_urls
                else:
                    return {
                        'title': 'YouTube Channel Feed',
                        'description': 'Full feed of all videos from the channel',
                        'id': 'single_video'
                    }, [{'url': info['url'], 'id': info.get('id', '')}]
        except Exception as e:
            print(f"❌ Error fetching channel info: {e}")
            return None, []

def fetch_video_details(video_url):
    """Fetch detailed information for a single video."""
    ydl_opts = {
        'extract_flat': False,  # Fetch full metadata
        'quiet': True,
        'ignoreerrors': True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(video_url, download=False)
            if info:
                return {
                    'id': info.get('id', ''),
                    'title': info.get('title', 'No title'),
                    'description': info.get('description', ''),
                    'upload_date': info.get('upload_date', ''),
                    'duration': info.get('duration', 0),
                }
            return None
        except Exception as e:
            print(f"❌ Error fetching video details for {video_url}: {e}")
            return None

def generate_rss(channel_info, videos, channel_url):
    fg = FeedGenerator()
    fg.title(channel_info['title'])
    fg.link(href=channel_url)
    fg.description(channel_info['description'])

    # Process videos with progress bar
    for video in tqdm(videos, desc=f"Processing {channel_info['title']}", unit="video"):
        if not video:
            continue

        fe = fg.add_entry()
        video_id = video.get("id")
        title = video.get("title", "No title")
        description = video.get("description", "")
        upload_date = video.get("upload_date")
        pub_date = datetime.strptime(upload_date, "%Y%m%d").replace(tzinfo=timezone.utc) if upload_date else datetime.now(timezone.utc)

        fe.title(title)
        fe.link(href=f'https://www.youtube.com/watch?v={video_id}')
        fe.guid(video_id)
        fe.description(description)
        fe.pubDate(pub_date)
        
        # Add enclosure for the video
        fe.enclosure(
            url=f'https://www.youtube.com/watch?v={video_id}',
            length=video.get('duration', 0) * 1024 * 1024,  # Approximate size in bytes
            type='video/mp4'
        )

    # Get raw XML
    raw_xml = fg.rss_str(pretty=False)

    # Pretty-print the XML
    pretty_xml = xml.dom.minidom.parseString(raw_xml).toprettyxml(indent="  ")

    # Generate filename from channel title and ID
    output_file = 'rss.xml' 

    # Write to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(pretty_xml)

    print(f"✅ RSS feed written to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RSS feed from YouTube channel")
    parser.add_argument("channel_id", help="YouTube channel ID")
    parser.add_argument("--limit", type=int, help="Limit the number of episodes to process")
    args = parser.parse_args()

    channel_url = f"https://www.youtube.com/channel/{args.channel_id}"
    print("🔍 Fetching channel information...")
    channel_info, video_urls = fetch_channel_info(channel_url, args.limit)
    
    if channel_info:
        print(f"📼 Found {len(video_urls)} videos. Fetching details...")
        # Fetch detailed information for each video
        videos = []
        for video_url in tqdm(video_urls, desc="Fetching video details", unit="video"):
            if video_url and video_url.get('url'):
                video_details = fetch_video_details(video_url['url'])
                if video_details:
                    videos.append(video_details)
        
        print(f"✅ Successfully fetched details for {len(videos)} videos. Generating RSS...")
        generate_rss(channel_info, videos, channel_url)
    else:
        print("❌ Failed to fetch channel information")

