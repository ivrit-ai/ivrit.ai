import pandas as pd
from api_integration import send_request_to_openai, extract_information_from_response
from datetime import datetime


def aggregate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates the data by adding new columns based on API responses.

    :param df: DataFrame containing the podcast episodes information.
    :return: DataFrame enriched with additional information from the API.
    """
    # Create empty lists to store the new data
    participants = []
    genders = []
    topics = []

    for _, row in df.iterrows():
        text = f"the title is: {row['title']} and the description is: {row['description']}"
        response = send_request_to_openai(text)
        extracted_info = extract_information_from_response(response)

        # Append new information to the respective lists
        participants.append(extracted_info.get('participants', 'None'))
        genders.append(extracted_info.get('genders', 'None'))
        topics.append(extracted_info.get('topics', 'None'))

    # Add the new data as columns to the DataFrame
    df['participants'] = participants
    df['genders'] = genders
    df['topics'] = topics

    return df


# Example usage
if __name__ == "__main__":
    # Load the dataset
    filename = f'episodes_20240116_162532.csv'
    df_episodes = pd.read_csv(filename)

    # Sample 10 random rows
    df_sample = df_episodes.sample(n=30)

    # Aggregate data for the sampled rows
    aggregated_df = aggregate_data(df_sample)

    # Create a new filename for the enriched data
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    enriched_filename = filename.replace('.csv', f'_enriched_{current_time}.csv')

    # Export the enriched DataFrame to CSV
    aggregated_df.to_csv(enriched_filename, index=False)

    print(f"DataFrame exported to {enriched_filename}")
