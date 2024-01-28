import feedparser
from feedparser.util import FeedParserDict
import os
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
        episodes.append({
            'title': entry.get('itunes_title', 'None'),
            'number': entry.get('itunes_episode', 'None'),
            'duration': entry.get('itunes_duration', 'None'),
            'description': entry.get('summary', 'None'),
            'id': entry.get('id', 'None'),
            'published': entry.get('published_parsed', 'None')
        })
    return episodes


def extract_podcast_data_from_content(podcast_text: str, podcast_context: dict = None,
                                      model_name: str = "gpt-4-1106-preview") -> str:
    """
    Sends a request to the OpenAI API to analyze podcast episode data and extract specific information.

    :param podcast_text: The text of the podcast episode for analysis.
    :param podcast_context: Additional context information for the API request.
    :param model_name: The name of the OpenAI model to be used. Default is 'gpt-4-1106-preview'.
    :return: The API response as a JSON string.
    """
    api_key = os.getenv('OPENAI_API_KEY')
    client = OpenAI(api_key=api_key)

    prompt = f"""
        Analyze the following podcast episode data: {podcast_text}
        """
    if podcast_context:
        prompt += f"Context: {json.dumps(podcast_context)}\n"
    prompt += """
        Extract the participant names from the title, their genders (male, female, unknown), 
        and the main topics. Format the response as a JSON object with the keys 
        'participants', 'genders', and 'topics'. For example: 
        {
          "participants": ["Alice", "Bob"],
          "genders": ["female", "male"],
          "topics": ["technology", "science"]
        }
        Based on the episode data, fill in the appropriate information in the JSON structure. Return Only json and not something else.
        """

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def extract_information_from_response(response_text: str) -> dict:
    """
    Extracts information from the OpenAI API response_text.

    :param response_text: The response_text from the OpenAI API.
    :return: Extracted information as a dictionary.
    """
    # Remove the extra characters
    json_str = response_text.replace('```json\n', '').replace('\n```', '').strip()
    return json.loads(json_str)


def parse_rss_file_to_json(file_path: str) -> str:
    """
    Parses the RSS feed from a given file path and enriches it with extra information.

    :param file_path: The path to the RSS feed file.
    :return: JSON string with enriched feed content.
    """
    parsed_feed = feedparser.parse(file_path)
    episodes = extract_episode_info_from_rss(parsed_feed)

    for episode in episodes:
        podcast_text = f"{episode['title']} {episode['description']}"
        content_data = extract_podcast_data_from_content(podcast_text)
        extracted_info = extract_information_from_response(content_data)
        episode.update(extracted_info)

    return json.dumps(episodes, indent=4)


if __name__ == "__main__":
    file_path = 'path_to_your_rss_file.xml'
    enriched_episodes_json = parse_rss_file_to_json(file_path)
    print(enriched_episodes_json)
