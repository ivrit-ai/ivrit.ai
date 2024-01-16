from openai import OpenAI

import json
import os


def send_request_to_openai(text: str, context_info: dict = None) -> str:
    """
    Sends a request to the OpenAI API with the given text.

    :param text: The text for which information is to be extracted.
    :param context_info: Additional context information for the API request.
    :return: The API response_text as a JSON string.
    """
    api_key = "YOUR-OPENAI-API-KEY"  # replace it with the real key
    client = OpenAI(api_key=api_key)

    prompt = f"Analyze the following podcast episode data: {text}\n"
    if context_info:
        prompt += f"Context: {json.dumps(context_info)}\n"
    prompt += (
        "Extract the participant names from the title, their genders (male, female, unknown), "
        "and the main topics. Format the response as a JSON object with the keys "
        "'participants', 'genders', and 'topics'. For example: \n"
        "{\n"
        "  \"participants\": [\"Alice\", \"Bob\"],\n"
        "  \"genders\": [\"female\", \"male\"],\n"
        "  \"topics\": [\"technology\", \"science\"]\n"
        "}\n"
        "Based on the episode data, fill in the appropriate information in the JSON structure. Return Only json and not something else."
    )

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "user", "content": prompt}, ],
    )
    print(response)
    return response.choices[0].message.content


def extract_information_from_response(response_text: str) -> dict:
    """
    Extracts information from the OpenAI API response_text.

    :param response_text: The response_text from the OpenAI API.
    :return: Extracted information as a dictionary.
    """
    # Remove the extra characters
    json_str = response_text.replace('```json\n', '').replace('\n```', '').strip()

    # Parse the string as JSON
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return {}


# Example usage
if __name__ == "__main__":
    text = """
    גיא נתיב הוא במאי ויוצר קולנוע ישראלי שסרטו ״גולדה״ מוקרן בימים אלה בישראל וברחבי העולם. גיא הוא גם הבמאי הישראלי השני שזכה אי פעם באוסקר (על סרטו הקצר ״סקין״). על איך זה מרגיש לזכות, איך מביאים סרט קצר לזכות באוסקרים ומה הפרוייקט שגיא כרגע עובד עליו.
 
על מה דיברנו:
 
אוסקרים, סטינג, טרודי, הורות, כתיבת תסריט, The Last of Us, אפולו 11, גולדה, שחקנים ישראלים ואמריקאים, קרן הקולנוע הישראלי ועוד המון דברים.
 
קישורים והמלצות:

גיא נתיב - בטוויטר, באינסטגרםהסרט זוכה האוסקר ״סקין״ ביוטיוב
לינק להורדה ישירה
הרשמה ב-iTunes הרשמו ל-RSS שלנו
"""
    context_info = {
        "hosts": ["ראם שרמן", "דורון ניר"]
    }
    response = send_request_to_openai(text, context_info)
    print(response)
    info = extract_information_from_response(response)
    print(info)
