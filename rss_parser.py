#!/usr/bin/python3

import argparse
import backoff
import feedparser
from feedparser.util import FeedParserDict
import os
import re
import json
import dotenv
from openai import OpenAI

dotenv.load_dotenv()


def extract_episode_info_from_rss(feed: FeedParserDict) -> list:
    """
    Extracts episode information from the parsed RSS feed.

    :param feed: Parsed RSS feed content.
    :return: List of dictionaries with episode information.
    """
    episodes = []
    for entry in feed.entries:
        episodes.append(
            {
                "title": entry.get("title", "None"),
                "number": entry.get("episode", "None"),
                "duration": entry.get("duration", "None"),
                "description": entry.get("summary", "None"),
                "id": entry.get("id", "None"),
                "published": entry.get("published_parsed", "None"),
            }
        )
    return episodes


@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def extract_podcast_data_from_content(episode, model_name: str = "gpt-4-0125-preview") -> str:
    """
    Sends a request to the OpenAI API to analyze podcast episode data and extract specific information.

    :param podcast_text: The text of the podcast episode for analysis.
    :param podcast_context: Additional context information for the API request.
    :param model_name: The name of the OpenAI model to be used. Default is 'gpt-4-1106-preview'.
    :return: The API response as a JSON string.
    """

    prompt = f"""
Analyze the following episode:
<TITLE>{episode['title']}</TITLE>
<DESCRIPTION>{episode['description']}</DESCRIPTION>
"""

    prompt += """
Extract all speaking participant names from the title ONLY; only use the description in case there's no participant name in the title.
Other fields can be extracted from both the title and description fields.
Format the response with 'PARTICIPANT', and 'FIELD' tags.
If you can figure out who the hosts are, include them in the 'participants' section.
All fields should match academic fields of study. For example: 

<PARTICIPANT>Alice</PARTICIPANT>
<PARTICIPANT>Bob</PARTICIPANT>
<FIELD>Mathematics</FIELD>
<FIELD>Political Science</FIELD>

        Based on the episode data, fill in the appropriate information in the format abovee. Participant names and topics must both be in Hebrew only.
        """

    api_key = os.getenv("OPENAI_API_KEY")

    with OpenAI(api_key=api_key) as client:
        response = client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": prompt}])

    return response.choices[0].message.content


def extract_information_from_response(response_text: str) -> dict:
    """
    Extracts information from the OpenAI API response_text.

    :param response_text: The response_text from the OpenAI API.
    :return: Extracted information as a dictionary.
    """

    pattern = r"<PARTICIPANT>(.*?)</PARTICIPANT>"
    participants = re.findall(pattern, response_text, re.DOTALL)

    pattern = r"<FIELD>(.*?)</FIELD>"
    fields = re.findall(pattern, response_text, re.DOTALL)

    return {"participants": participants, "fields": fields}


def parse_rss_file_to_json(file_path: str) -> str:
    """
    Parses the RSS feed from a given file path and enriches it with extra information.

    :param file_path: The path to the RSS feed file.
    :return: JSON string with enriched feed content.
    """
    parsed_feed = feedparser.parse(file_path)
    episodes = extract_episode_info_from_rss(parsed_feed)

    for idx, episode in enumerate(episodes):
        print(f"Episode #{idx+1}/{len(episodes)}: {episode['title']}")

        content_data = extract_podcast_data_from_content(episode, model_name="gpt-4-1106-preview")
        extracted_info = extract_information_from_response(content_data)

        episode.update(extracted_info)

    return episodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract per-episode speakers and fields of discussion from RSS feeds."
    )

    # Add the arguments
    parser.add_argument("--feed", type=str, required=True, help="RSS feed.")

    # Parse the arguments
    args = parser.parse_args()

    enriched_episodes = parse_rss_file_to_json(args.feed)
    json.dump(enriched_episodes, open("feed_metadata.json", "w"))
