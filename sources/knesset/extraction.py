import csv
import pathlib
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from time import gmtime, strftime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import linregress
from bs4 import BeautifulSoup, NavigableString, PageElement
from webvtt import Caption, WebVTT

from sources.knesset.cleanup import cleanup_time_index


@dataclass
class TimedSegment:
    """
    A dataclass representing a timed segment of text with start and end positions.
    """

    start_loc: int
    end_loc: int
    timestamp: float


def extract_title_from_xml(protocol_xml_path: pathlib.Path) -> Optional[str]:
    """
    Extract the title from the protocol XML file.

    Args:
        protocol_xml_path: Path to the protocol XML file

    Returns:
        str: The title text or None if not found
    """
    try:
        tree = ET.parse(protocol_xml_path)
        root = tree.getroot()
        title_element = root.find("sTitle")
        if title_element is not None and title_element.text:
            return title_element.text
        return None
    except Exception as e:
        print(f"Error extracting title from XML: {e}")
        return None


def extract_time_pointer_array_from_xml(protocol_xml_path: pathlib.Path) -> Optional[np.ndarray]:
    """
    Extract the time pointer array from the protocol XML file.

    Args:
        protocol_xml_path: Path to the protocol XML file

    Returns:
        np.ndarray: The time pointer array or None if not found
    """
    try:
        tree = ET.parse(protocol_xml_path)
        root = tree.getroot()
        js_arrays_element = root.find("sJSArrays")

        if js_arrays_element is not None and js_arrays_element.text:
            js_code = js_arrays_element.text

            # Extract all bk[x]=y; statements using regex
            # Those represent a "segment id" (array index) to time stamp (value in seconds)
            bk_assignments = re.findall(r"bk\[(\d+)\]=(\d+);", js_code)

            if not bk_assignments:
                return None

            # Find the maximum index to determine array size
            max_index = max(int(idx) for idx, _ in bk_assignments)

            # Create a numpy array with appropriate size (add 1 because indices are 0-based)
            time_pointer_array = np.zeros(max_index + 1, dtype=int)

            # Fill the array with the extracted values
            for idx, value in bk_assignments:
                time_pointer_array[int(idx)] = int(value)

            cleaned_up_time_pointer_array = cleanup_time_index(time_pointer_array)
            return cleaned_up_time_pointer_array
    except Exception as e:
        print(f"Error extracting bk array from XML: {e}")
        return None


def extract_transcription_from_html(html_path: pathlib.Path) -> Dict:
    """
    Extract transcription data from an HTML file.

    Args:
        html_path: Path to the HTML file

    Returns:
        Dict: A dictionary containing the transcription data
    """
    # This is a stub function that will be implemented later
    # For now, we'll just return an empty dictionary with the file path
    return {"file_path": str(html_path), "transcription": "Placeholder transcription data"}


# Plenum Custom HTML transcripts structure
plenum_html_text_idx_extractor_pattern = re.compile("txt_(\d+)")


class TimeMarker:
    def __init__(self, start=True, seconds=0, id=None) -> None:
        self.start = start
        self.seconds = seconds
        self.id = id

    # Support serialziing as string
    def __repr__(self) -> str:
        return f"[[{'S' if self.start else 'E'}{self.seconds:g}]]"


def gather_html_transcripts(html_paths: List[pathlib.Path]) -> str:
    """
    Gather HTML transcripts from multiple files.

    Args:
        html_paths: List of paths to HTML files

    Returns:
        str: Concatenated HTML content wrapped in a root div
    """

    # Extract segment numbers from file names and sort by "from segment #"
    def extract_from_segment(path):
        # File name format: "protocol_<plenum id>_<from segment #>_<to segment #>_<from ts>_<to ts>_bulk.html"
        parts = path.name.split("_")
        if len(parts) >= 3:
            try:
                return int(parts[4])  # <from time secs> is at index 4
            except ValueError:
                pass
        return 0  # Default value if extraction fails

    # Sort paths by "from segment #"
    sorted_paths = sorted(html_paths, key=extract_from_segment)

    # Read and concatenate all HTML contents
    html_transcript_snippet = ""
    for path in sorted_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                html_transcript_snippet += f.read()
        except Exception as e:
            print(f"Error reading HTML file {path}: {e}")

    # Wrap all content in a root div
    return f'<div id="root">{html_transcript_snippet}</div>'


def extract_transcript_parts_from_elements(root_element: PageElement, context: dict):
    if isinstance(root_element, NavigableString):
        if (
            root_element.get_text() == "\n"
        ):  # new line is a space when HTML is parsed - not an actual new line which is an explicit <br> tag
            return [" "]

        return [root_element.get_text()]
    elif root_element.name == "span":
        extracted_list = [
            extracted
            for child in root_element.children
            for extracted in extract_transcript_parts_from_elements(child, context)
        ]

        # get the id - find the time context
        if "id" in root_element.attrs:
            text_idx_matches = plenum_html_text_idx_extractor_pattern.findall(root_element["id"])
            text_ts_idx = None

            if len(text_idx_matches) > 0:
                text_ts_idx = int(text_idx_matches[0])

            if text_ts_idx is not None:
                prev_text_ts: pd.Timedelta = context["current_text_ts"]
                time_at_idx: pd.Timedelta = pd.to_timedelta(context["time_pointers_arr_np"][text_ts_idx], "s")

                if not context["first_non_zero_ts_seen"] and time_at_idx.total_seconds() > 0:
                    context["first_non_zero_ts_seen"] = True

                # time 0 is an artifact appearing at the beginning of the transcript
                # When timestamps probably overflow from the previous recording.
                # they are cleaned and this we get a 0 here.
                if not context["first_non_zero_ts_seen"]:
                    # Not seen any real TS yet - we will fake a ts for those segments
                    # which is close to the first real ts we will encounter.
                    first_ts_index = np.argwhere(context["time_pointers_arr_np"]).flatten()[0]
                    time_at_idx = pd.to_timedelta(context["time_pointers_arr_np"][first_ts_index] - 3, "s")

                if time_at_idx.total_seconds() > 0 and (  # Zero TS should not be used - it's garbage
                    prev_text_ts is None or context["current_text_ts"] < time_at_idx
                ):
                    context["current_text_ts"] = time_at_idx
                    extracted_list = [
                        TimeMarker(True, time_at_idx.total_seconds()),
                        *extracted_list,
                    ]

                # if we moved to a new TS - add an end marker before the begin marker
                if prev_text_ts is not None and prev_text_ts.total_seconds() > 0 and prev_text_ts < time_at_idx:
                    extracted_list = [
                        TimeMarker(False, prev_text_ts.total_seconds()),
                        *extracted_list,
                    ]
                    context["latest_closed_ts_marker"] = prev_text_ts

        return extracted_list

    elif root_element.name == "font":
        # Some classes are not part of the transcript itself but rather metadata we currently ignore
        if "class" in root_element.attrs and root_element["class"][0] in [
            "SUBJECT",
            "SPEAKER",
            "SPEAKER_INTERRUPTION",
            "SPEAKER_CONTINUE",
            "YOR",
            "VOTING_TOP",
            "VOTING_MID",
            "VOTING_BOTTOM",
        ]:
            return []

        return [
            extracted
            for child in root_element.children
            for extracted in extract_transcript_parts_from_elements(child, context)
        ]
    elif root_element.name == "div":
        return [
            extracted
            for child in root_element.children
            for extracted in extract_transcript_parts_from_elements(child, context)
        ]
    elif root_element.name == "br":
        return ["\n"]
    else:
        raise ValueError(f"What should I do with type: {type(root_element)} and str: {str(root_element)}")


def parse_plenum_transcript(content, time_pointers_arr_np) -> Tuple[str, List[TimedSegment]]:
    """
    Parse the plenum transcript HTML content.

    Args:
        content: HTML content of the transcript
        time_pointers_arr_np: Array of time pointers

    Returns:
        Tuple containing the transcript text and a list of TimedSegment objects
    """
    soup = BeautifulSoup(content, "html.parser")

    page_text_parts = []
    extraction_context = {
        "time_pointers_arr_np": time_pointers_arr_np,
        "first_non_zero_ts_seen": False,
        "current_text_ts": None,
        "latest_closed_ts_marker": 0,
    }
    root_element = soup.find("div", {"id": "root"})
    page_text_parts.extend(extract_transcript_parts_from_elements(root_element, extraction_context))

    # Ensure last marker is properly closed
    if extraction_context["latest_closed_ts_marker"] < extraction_context["current_text_ts"]:
        page_text_parts.append(TimeMarker(False, extraction_context["current_text_ts"].total_seconds()))

    # Separate text parts from to text and time index
    text_timestamp_index = {}
    text_only_parts = []
    text_len_so_far = 0

    first_marker_seen = False
    for part in page_text_parts:
        if isinstance(part, TimeMarker):
            first_marker_seen = True
            if part.start:
                text_timestamp_index[part.seconds] = {"text_start_idx": text_len_so_far}
            else:
                text_timestamp_index[part.seconds]["text_end_idx"] = text_len_so_far
        elif first_marker_seen:  # Skips any header text prior to first time marker
            # deduplicate whitespaces excluding new lines
            part = re.sub(r"[^\S\n]+", " ", part)

            text_part_len = len(part)
            text_len_so_far += text_part_len
            text_only_parts.append(part)

    text_timestamps = []
    text_timestamp_start_locs = []
    text_timestamp_end_locs = []
    for ts, start_end_pair in text_timestamp_index.items():
        text_timestamps.append(ts)
        text_timestamp_start_locs.append(start_end_pair["text_start_idx"])
        text_timestamp_end_locs.append(start_end_pair["text_end_idx"])

    # Create a list of TimedSegment objects
    timestamp_segments = [
        TimedSegment(start_loc=start, end_loc=end, timestamp=ts)
        for start, end, ts in zip(text_timestamp_start_locs, text_timestamp_end_locs, text_timestamps)
    ]

    # Sort by timestamp
    timestamp_segments.sort(key=lambda segment: segment.timestamp)

    all_processed_text = "".join(text_only_parts)

    return all_processed_text, timestamp_segments


# Common formatting usage in the transcripts which should be included in the captions
translate_formatting_to_replace_with_space = {
    # en-dash characters
    0x2013: " ",
    0x2014: " ",
}


def normalize_text_as_caption_text(text: str) -> str:
    # remove square bracketed text
    text = re.sub(r"\[.*?\]", "", text)
    # Remove formatting
    text = text.translate(translate_formatting_to_replace_with_space)
    # Deduplicate whitespaces
    text = re.sub(r"\s+", " ", text)
    return text


def create_caption(text: str, start: float, end: float) -> Caption:
    return Caption(
        # Format to hh:mm:ss.zzz
        start=strftime("%H:%M:%S.000", gmtime(start)),
        end=strftime("%H:%M:%S.000", gmtime(end)),
        text=text,
    )


max_first_caption_inclusion_lookback_window_length = 120


def save_timestamp_index_to_csv(timestamp_segments: List[TimedSegment], output_path: pathlib.Path) -> None:
    """
    Save a list of TimedSegment objects to a CSV file.

    Args:
        timestamp_segments: List of TimedSegment objects
        output_path: Path to save the CSV file
    """
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["start_loc", "end_loc", "timestamp"])  # Header
        for segment in timestamp_segments:
            writer.writerow([segment.start_loc, segment.end_loc, segment.timestamp])


max_slice_gap_to_merge_seconds = 2
fallback_avg_word_duration = 0.42
max_duration_to_consider_in_avg_word_duration = 10
max_expected_to_actual_duration_sanity_ratio = 3
max_expected_duration_to_merge_small_gap_slices = 15


def expected_duration_of_text(text: str, avg_word_duration: float) -> float:
    """
    Estimate the duration of text based on word count and average word duration.

    Args:
        text: The input text to analyze
        avg_word_duration: Average duration of a word in seconds

    Returns:
        Estimated duration of the text in seconds
    """
    # Count words by splitting on whitespace
    words = text.split()
    word_count = len(words)

    # Calculate expected duration
    avg_word_duration = avg_word_duration or fallback_avg_word_duration
    duration = word_count * avg_word_duration

    return duration


def update_word_duration_stats(
    text: str, actual_duration: float, avg_word_duration: float, alpha: float = 0.98
) -> float:
    """
    Update word duration statistics using exponential moving average.

    Args:
        text: The input text
        avg_word_duration: Current average word duration in seconds
        alpha: EMA weight factor (default: 0.98)

    Returns:
        Updated average word duration in seconds
    """
    # Use fallback value if avg_word_duration is None or 0
    if avg_word_duration is None or avg_word_duration == 0:
        avg_word_duration = fallback_avg_word_duration  # Using 0.4 as provided

    # Count words
    words = text.split()
    word_count = len(words)

    if word_count == 0:
        return avg_word_duration  # No update if no words

    # Calculate new word duration rate
    new_word_duration = actual_duration / word_count

    # Apply exponential moving average
    updated_avg_word_duration = alpha * avg_word_duration + (1 - alpha) * new_word_duration

    return updated_avg_word_duration


def create_recording_transcript_vtt(text: str, time_segments: List[TimedSegment]) -> WebVTT:
    vtt = WebVTT()
    # Ensure segments are sorted by timestamp
    time_segments = sorted(time_segments, key=lambda segment: segment.timestamp)

    merging_next = False
    merge_start_loc = None
    tracked_avg_word_duration = None

    tmp_track = []

    for i, segment in enumerate(time_segments):
        is_not_last = i < len(time_segments) - 1
        # Get the next timestamp (for caption end time)
        next_ts = time_segments[i + 1].timestamp if is_not_last else segment.timestamp

        # merge consecutive slices sharing the same TS
        if next_ts == segment.timestamp:
            merge_start_loc = merge_start_loc or segment.start_loc
            merging_next = True
        else:
            slice_start_loc = segment.start_loc
            if merging_next:
                merging_next = False
                slice_start_loc = merge_start_loc
                merge_start_loc = None

            text_range_slice = slice(int(slice_start_loc), int(segment.end_loc))
            extracted_text_for_slice = normalize_text_as_caption_text(text[text_range_slice])

            # If the slice ends with a word character - this may be a marker mid-word.
            # check if the adjacent slice starts with a word character - if so - merge them.
            if re.match(r"\w", extracted_text_for_slice[-1]) and is_not_last:
                next_slice_start_loc = time_segments[i + 1].start_loc
                next_slice_end_loc = time_segments[i + 1].end_loc
                # if the first character is a word character - merge the slices
                next_slice_text = normalize_text_as_caption_text(
                    text[int(next_slice_start_loc) : int(next_slice_end_loc)]
                )
                if re.match(r"\w", next_slice_text[0]):
                    merging_next = True
                    merge_start_loc = slice_start_loc

            ends_with_punctuation = re.search(r"[,.!?:]$", extracted_text_for_slice[-10:].strip())

            expected_duration_for_this_segment = expected_duration_of_text(
                extracted_text_for_slice, tracked_avg_word_duration
            )

            # If not merging already - consider too short slices (by time) to also be joined.
            # Only do that if this is not already a too-long slice.
            # measure the time span from this to the next segment. (we merge forward)
            # but only if it does not end with punctuation - which will be a good place to slice anyway.
            allow_small_gap_merge = expected_duration_for_this_segment < max_expected_duration_to_merge_small_gap_slices
            if allow_small_gap_merge and not ends_with_punctuation and not merging_next and is_not_last:
                expected_end_at = segment.timestamp + expected_duration_for_this_segment
                if next_ts - expected_end_at <= max_slice_gap_to_merge_seconds:
                    merging_next = True
                    merge_start_loc = slice_start_loc

            # Only if we are not awaiting a future merge - add the caption
            if not merging_next:
                duration_for_this_segment = next_ts - segment.timestamp
                if duration_for_this_segment <= max_duration_to_consider_in_avg_word_duration:
                    tracked_avg_word_duration = update_word_duration_stats(
                        extracted_text_for_slice, duration_for_this_segment, tracked_avg_word_duration
                    )
                else:
                    # For sanity - if the segment is too long to make sense - use the expected duration
                    if (
                        duration_for_this_segment
                        > max_expected_to_actual_duration_sanity_ratio * expected_duration_for_this_segment
                    ):
                        duration_for_this_segment = expected_duration_for_this_segment

                vtt.captions.append(
                    create_caption(
                        extracted_text_for_slice, segment.timestamp, segment.timestamp + duration_for_this_segment
                    )
                )

            tmp_track.append(tracked_avg_word_duration)

    return vtt


def process_transcripts(
    protocol_xml_path: pathlib.Path,
    protocol_html_paths: List[pathlib.Path],
    output_dir: pathlib.Path,
    plenum_id: str,
    force_reprocess: bool = False,
) -> bool:
    """
    Process the transcript files for a plenum.

    Args:
        protocol_xml_path: Path to the protocol XML file
        protocol_html_paths: List of paths to protocol HTML files
        output_dir: Output directory where processed files will be saved
        plenum_id: ID of the plenum
        force_reprocess: Whether to force reprocessing even if files already exist

    Returns:
        bool: True if processing was successful, False otherwise
    """
    try:
        plenum_output_dir = output_dir / plenum_id
        plenum_output_dir.mkdir(parents=True, exist_ok=True)

        if not force_reprocess and (plenum_output_dir / "transcript.txt").exists():
            print(f"Skipping plenum {plenum_id} - transcript already exists")
            return True

        # Extract title from XML
        title = extract_title_from_xml(protocol_xml_path)
        if title is None:
            print(f"Failed to extract title from XML for plenum {plenum_id}")

        # Extract bk array from XML (this is the time_pointers_arr_np)
        bk_array = extract_time_pointer_array_from_xml(protocol_xml_path)
        if bk_array is None:
            print(f"Failed to extract bk array from XML for plenum {plenum_id}")
            return False

        # Gather HTML transcripts
        html_transcript = gather_html_transcripts(protocol_html_paths)

        # Parse the plenum transcript
        transcript_text, timestamp_segments = parse_plenum_transcript(html_transcript, bk_array)

        # Save the transcript text
        transcript_text_path = plenum_output_dir / "transcript.txt"
        with open(transcript_text_path, "w", encoding="utf-8") as f:
            f.write(transcript_text)

        # Save the timestamp index
        timestamp_index_path = plenum_output_dir / "transcript.timeindex.csv"
        save_timestamp_index_to_csv(timestamp_segments, timestamp_index_path)

        print("Creating a VTT file of the transcript...")
        vtt = create_recording_transcript_vtt(transcript_text, timestamp_segments)
        vtt.save(plenum_output_dir / f"transcript.vtt", add_bom=True)

        print(f"Successfully processed {len(protocol_html_paths)} HTML files for plenum {plenum_id}")
        print(f"Transcript saved to {transcript_text_path}")
        print(f"Timestamp index saved to {timestamp_index_path}")

        return True
    except Exception as e:
        print(f"Error processing transcripts for plenum {plenum_id}: {e}")
        return False
