import os
import re
from typing import Union

from stable_whisper.result import WhisperResult


def vtt_to_whisper_result(source: Union[str, bytes]) -> WhisperResult:
    """
    Parse a VTT file (provided as a file path or as file contents in a buffer)
    and return a WhisperResult–like dict with segments.

    Each segment in the returned result is a dict with:
      - "start": float, start time in seconds.
      - "end": float, end time in seconds.
      - "text": str, the caption text (may span multiple lines).

    Any non-cue metadata (e.g., WEBVTT header, NOTE sections) is ignored.

    Parameters
    ----------
    source : Union[str, bytes]
        Either a file path to a VTT file or the VTT file contents as a string (or bytes).

    Returns
    -------
    Dict[str, Any]
        A dictionary with at least the key "segments" (a list of segment dicts) and a key
        "text" which is the full transcription (concatenation of each segment’s text).
    """
    # If the provided source is a valid file path, read its contents.
    if isinstance(source, (str, bytes)) and isinstance(source, str) and os.path.exists(source):
        with open(source, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        # If source is bytes, decode; otherwise assume it's already a string.
        content = source.decode("utf-8") if isinstance(source, bytes) else source

    # Remove BOM if present.
    content = content.lstrip("\ufeff")

    # Split content into lines.
    lines = content.splitlines()

    segments = []
    i = 0
    n = len(lines)

    # Regular expression to match a cue's timestamp line.
    # Matches: HH:MM:SS.mmm --> HH:MM:SS.mmm
    time_pattern = re.compile(r"(?P<start>\d{1,2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(?P<end>\d{1,2}:\d{2}:\d{2}\.\d{3})")

    def parse_timestamp(ts: str) -> float:
        """Convert a VTT timestamp string (HH:MM:SS.mmm) to seconds (float)."""
        h, m, s = ts.split(":")
        return float(h) * 3600 + float(m) * 60 + float(s)

    while i < n:
        line = lines[i].strip()
        # Skip empty lines and header/metadata lines.
        if not line or line.upper().startswith("WEBVTT") or line.upper().startswith("NOTE"):
            i += 1
            continue

        # A cue block may optionally start with an identifier. Look for the timestamp line.
        if "-->" not in line:
            i += 1
            if i >= n:
                break
            line = lines[i].strip()
            if "-->" not in line:
                # Not a valid cue block; continue to next line.
                i += 1
                continue

        # Now we expect a line with the timestamp.
        time_line = line
        match = time_pattern.search(time_line)
        if not match:
            i += 1
            continue

        start_ts = match.group("start")
        end_ts = match.group("end")
        start_sec = parse_timestamp(start_ts)
        end_sec = parse_timestamp(end_ts)

        # Gather all following lines until a blank line (which separates cues).
        text_lines = []
        i += 1
        while i < n and lines[i].strip():
            text_lines.append(lines[i])
            i += 1
        text = " " + " ".join(text_lines).strip()

        segments.append({"start": start_sec, "end": end_sec, "text": text})
        # Skip the blank line separating cues.
        i += 1

    # Build complete text from segments by concatenating their texts.
    full_text = "".join(seg["text"] for seg in segments)

    # Return a WhisperResult-like dict.
    return WhisperResult({"segments": segments, "text": full_text})
