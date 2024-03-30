#!/usr/bin/python3

import argparse
from datetime import datetime
import dotenv
import os

from googleapiclient.discovery import build
from feedgen.feed import FeedGenerator
import html



dotenv.load_dotenv()

def get_all_videos_by_year(api_key, channel_id, year):
    youtube = build('youtube', 'v3', developerKey=api_key)
    all_videos = []
    page_token = None

    while True:
        request = youtube.search().list(
            part='snippet',
            channelId=channel_id,
            maxResults=50,
            pageToken=page_token,
            publishedAfter = datetime(year, 1, 1).isoformat() + 'Z',
            publishedBefore = datetime(year + 1, 1, 1).isoformat() + 'Z',
            type='video',
            order='date'
        )
        response = request.execute()

        for item in response.get('items', []):
            details = youtube.videos().list(part='snippet,contentDetails', id=item['id']['videoId']).execute()
            all_videos.append(details['items'][0])

        page_token = response.get('nextPageToken', None)
        if page_token is None:
            break

    return all_videos

def get_all_videos(api_key, channel_id, start_year, end_year):
    all_videos = []

    for year in range(end_year, start_year - 1, -1):
        print(f'Fetching videos for {year}...')
        yearly_videos = get_all_videos_by_year(api_key, channel_id, year)
        print(f'Done fetching {len(yearly_videos)} videos.')
        all_videos += yearly_videos

    return all_videos

def generate_rss_feed(channel_id, videos):
    fg = FeedGenerator()
    fg.id(f'https://www.youtube.com/channel/{channel_id}')
    fg.title(f'YouTube Channel Videos for {channel_id}')
    fg.link(href=f'https://www.youtube.com/channel/{channel_id}', rel='alternate')
    fg.description(f'Latest videos from the YouTube channel {channel_id}')
    fg.language('en')

    for video in videos:
        video_details = video['snippet']
        video_url = f'https://www.youtube.com/watch?v={video["id"]}'

        fe = fg.add_entry()
        fe.id(video['id'])
        fe.title(html.unescape(video_details['title']))
        fe.link(href=video_url)
        fe.published(video_details['publishedAt'])
        fe.description(html.unescape(video_details['description']))
        if 'tags' in video_details:
            for tag in video_details['tags']:
                fe.category(term=tag)

        fe.enclosure(video_url, 0, 'video/mp4')

    return fg.rss_str(pretty=True)

# Set up argument parsing
parser = argparse.ArgumentParser(description='Generate an RSS feed from a YouTube channel.')
parser.add_argument('--channel_id', type=str, required=True, help='YouTube Channel ID')
parser.add_argument('--output_file', type=str, required=True, help='Output file name (e.g., feed.xml)')
args = parser.parse_args()

videos = get_all_videos(os.environ['YOUTUBE_API_KEY'], args.channel_id, 2006, 2024)
rss_feed = generate_rss_feed(args.channel_id, videos)

# Write the feed to the specified output file
with open(args.output_file, 'w', encoding='utf-8') as file:
    if isinstance(rss_feed, bytes):
        rss_feed = rss_feed.decode('utf-8')
    file.write(rss_feed)
