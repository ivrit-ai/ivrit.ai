import re

from hebrew import Hebrew


def remove_bracketed_text(text: str) -> str:
    return re.sub(r"\s*[\(\[\{<].*?[\)\]\}>]", "", text)


def remove_brackets_keep_content(text: str) -> str:
    # Keep one non-word character if brackets had non-word chars around them
    return re.sub(
        r"(\W*)[\(\[\{<](.*?)[\)\]\}>](\W*)",
        lambda m: (m.group(1) or " ") + m.group(2).strip() + (m.group(3) or " "),
        text,
    )


def remove_niqqud(text: str) -> str:
    return Hebrew(text).no_niqqud().string


superfluous_chars_to_remove = "\u061C"  # Arabic letter mark
superfluous_chars_to_remove += "\u200B\u200C\u200D"  # Zero-width space, non-joiner, joiner
superfluous_chars_to_remove += "\u200E\u200F"  # LTR and RTL marks
superfluous_chars_to_remove += "\u202A\u202B\u202C\u202D\u202E"  # LTR/RTL embedding, pop, override
superfluous_chars_to_remove += "\u2066\u2067\u2068\u2069"  # Isolate controls
superfluous_chars_to_remove += "\uFEFF"  # Zero-width no-break space
superfluous_hebrew_unicode_symbols_translator = str.maketrans({ord(c): None for c in superfluous_chars_to_remove})


def remove_superfluous_hebrew_unicode_symbols(text: str) -> str:
    return text.translate(superfluous_hebrew_unicode_symbols_translator)


def remove_colons(text: str) -> str:
    return re.sub(r":", "", text)


def cleanup_text_for_whisper_alignment(text: str) -> str:
    text = remove_brackets_keep_content(text)
    text = remove_niqqud(text)
    text = remove_superfluous_hebrew_unicode_symbols(text)
    text = remove_colons(text)
    return text
