import feedparser
import pandas as pd
from datetime import datetime
from feedparser.util import FeedParserDict

def parse_rss_file(file_path: str) -> FeedParserDict:
    """
    Parses the RSS feed from a given file path.

    :param file_path: The path to the RSS feed file.
    :return: Parsed feed content.
    """
    return feedparser.parse(file_path)


def extract_episode_info(feed: FeedParserDict) -> pd.DataFrame:
    """
    Extracts episode information from the parsed RSS feed.

    :param feed: Parsed RSS feed content.
    :return: DataFrame with episode information.
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
    return pd.DataFrame(episodes)


# Example usage
if __name__ == "__main__":
    feed = parse_rss_file('feed.xml')
    df_episodes = extract_episode_info(feed)

    # Get the current time
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create filename with current time
    filename = f'episodes_{current_time}.csv'

    # Export to CSV
    df_episodes.to_csv(filename, index=False)

    print(f"DataFrame exported to {filename}")
