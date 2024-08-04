#!/usr/bin/python3

import argparse
import json
import pathlib
import re
import subprocess
from time import gmtime, strftime
from urllib.parse import urlparse, parse_qs

from bs4 import BeautifulSoup, PageElement, NavigableString
from docx import Document
from docx.text.paragraph import Paragraph
from docx.oxml.parser import register_element_cls
from docx.oxml.simpletypes import ST_DecimalNumber, ST_String
from docx.oxml.text.run import CT_R
from docx.oxml.text.parfmt import CT_PPr
from docx.oxml.xmlchemy import BaseOxmlElement, RequiredAttribute
import numpy as np
import pandas as pd
import pyodata
from pyodata.v2.service import GetEntitySetFilter as esf
import requests
from webvtt import WebVTT, Caption

RECORDING_TYPES = ["plenum", "committee"]

# Custom API for plenum data discovery
base_kenesset_url = "https://online.knesset.gov.il"
base_knesset_app_url = f"{base_kenesset_url}/app"
plenum_player_site_base_url = f"{base_kenesset_url}/app/#/player/peplayer.aspx"
knesset_protocol_ajax_api_base_url = f"{base_kenesset_url}/api/Protocol"
knesset_protocol_video_path_api_url = f"{knesset_protocol_ajax_api_base_url}/GetVideoPath"
knesset_protocol_html_ts_map_api_url = f"{knesset_protocol_ajax_api_base_url}/WriteBKArrayData"
knesset_protocol_page_count_api_url = f"{knesset_protocol_ajax_api_base_url}/GetBlunkCount"
knesset_protocol_page_content_api_url = f"{knesset_protocol_ajax_api_base_url}/GetProtocolBulk"

# Custom Scraping target for committee data discovery
base_knesset_committee_url = "https://main.knesset.gov.il/"
committee_broadcast_player_page_base_url = (
    f"{base_knesset_committee_url}/Activity/committees/Finance/Pages/CommitteeTVarchive.aspx"
)

# Parliament Odata service data discovery
parliament_service_odata_manifest_url = "http://knesset.gov.il/Odata/ParliamentInfo.svc/"
committee_protocol_document_type_id = 23

# EType Time Stampped Word-Doc structure
etype_time_bookmark_split_pattern = "_ETM_Q1_"
etype_time_bookmark_name_matcher = re.compile(r"^_ETM_Q1_")
etype_bookmark_name_matcher = re.compile(r"^ET_")
etype_missed_bookmark_text_pattern = re.compile(r"<< \S+ >>")

# Local output and caching file names
cached_video_source_url_file_name = "cache.video.url"
cached_transcript_html_file_name = "cache.transcript.html"
cached_html_ts_map_file_name = "cache.htmltsmap.npy"
cached_protocol_doc_source_url_file_name = "cache.protocol.doc.url"
cached_protocol_doc_file_name = "cache.protocol.doc"
cached_protocol_doc_xml_content_file_name = "cache.protocol.xml"

output_transcript_text_file_name = "transcript.txt"
output_transcript_time_index_file_name = "transcript.timeindex.csv"
output_video_file_name = "video.mp4"

HOW_TO_GET_HTTP_HEADERS = """
To Get the HTTP Headers needed to run this script perform the following steps manually:
=======================================================================================
- Open a new browser window - in private mode / Incognito (Firefox, Chrome would work)
- Open dev tools / Network tab
- Navigate to the player site at: https://online.knesset.gov.il/app
- Wait until the site loads
- Look for request to https://online.knesset.gov.il/app which resulted in a **301 redirect** (there are multiple requests to this URL)
- Click on the request and "Copy Request Headers"
- Paste within the headers input file, each header should occupy a separate line if done correctly
- Sorry.
"""

HOW_TO_GET_PLENUM_ID = """
To Get the Plenum ID needed to run this script perform the following steps manually:
=======================================================================================
- Open a new browser window
- Navigate to plenum sittings site at: https://online.knesset.gov.il/app#/search
- Set a date range and search
- Look for results of a full sitting not a segment (מקטע)
- Click the video preview link
- In the newly opened page - look for the "ProtocolID" query param
- This number is the plenum ID this script expects
"""

extract_header_user_agent = re.compile(r"User-Agent: (.*)")
extract_header_cookie = re.compile(r"Cookie: (.*)")


def load_http_headers(args) -> dict:
    if args.http_headers_file:
        with open(args.http_headers_file, "r") as f:
            http_headers_manual_data = f.read()

            return {
                "User-Agent": extract_header_user_agent.search(http_headers_manual_data).group(1),
                "Cookie": extract_header_cookie.search(http_headers_manual_data).group(1),
                "Referer": "https://online.knesset.gov.il/app",
                "Host": "online.knesset.gov.il",
            }


def set_http_headers_referer(http_headers, referer):
    http_headers["Referer"] = referer
    http_headers["Host"] = urlparse(referer).netloc


def check_http_access(knesset_http_headers: dict):
    try:
        print("Requesting main site page...")
        print("Using HTTP headers:")
        print(json.dumps(knesset_http_headers, indent=2))
        response = requests.get(f"{base_knesset_app_url}/", headers=knesset_http_headers)
        if response.status_code == 200:
            print("200 response recieved")
            # if the HTML content contains a <title> tag
            if 0 < len(response.content) and response.content.find(b"<title>") > 0:
                print("Detected <title>")
                return True
            else:
                print("Unable to detect <title>.")
        else:
            print("Status code is not 200...")
            return False
    except Exception as e:
        return False


# Plenum Custom HTML transcripts structure
plenum_html_text_idx_extractor_pattern = re.compile("txt_(\d+)")


# EType docx - Extend Parser to support bookmarks
class CT_Bookmark(BaseOxmlElement):
    """`<w:bookmarkStart>` element, containing a potnetial transcript timestamp"""

    id = RequiredAttribute("w:id", ST_DecimalNumber)
    name = RequiredAttribute("w:name", ST_String)

    def __str__(self) -> str:
        return f"||start {self.id}: {self.name}->"


class CT_MarkupRange(BaseOxmlElement):
    """`w:bookmarkEnd` element"""

    id = RequiredAttribute("w:id", ST_DecimalNumber)

    def __str__(self) -> str:
        return f"<-end {self.id}||"


register_element_cls("w:bookmarkStart", CT_Bookmark)
register_element_cls("w:bookmarkEnd", CT_MarkupRange)


def create_parliament_odata_client() -> pyodata.Client:
    http_session = requests.Session()
    # Use older "2" version of odata since the client does not support
    # "3" which is the current server supported version
    http_session.headers["MaxDataServiceVersion"] = "2.0"
    pareliamentInfoService = pyodata.Client(parliament_service_odata_manifest_url, http_session)

    return pareliamentInfoService


class TimeMarker:
    def __init__(self, start=True, seconds=0, id=None) -> None:
        self.start = start
        self.seconds = seconds
        self.id = id

    # Support serialziing as string
    def __repr__(self) -> str:
        return f"[[{'S' if self.start else 'E'}{self.seconds:g}]]"


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


def parse_plenum_transcript(content, time_pointers_arr_np) -> tuple[str, pd.DataFrame]:
    soup = BeautifulSoup(content, "html.parser")

    page_text_parts = []
    extraction_context = {
        "time_pointers_arr_np": time_pointers_arr_np,
        "current_text_ts": None,
        "latest_closed_ts_marker": 0,
    }
    root_element = soup.find("div", {"id": "root"})
    page_text_parts.extend(extract_transcript_parts_from_elements(root_element, extraction_context))

    # Ensure last marker is proplely closed
    if extraction_context["latest_closed_ts_marker"] < extraction_context["current_text_ts"]:
        page_text_parts.append(TimeMarker(False, extraction_context["current_text_ts"].total_seconds()))

    # Seperate text parts from to text and time index
    text_timestamp_index = {}
    text_only_parts = []
    text_len_so_far = 0
    for part in page_text_parts:
        if isinstance(part, TimeMarker):
            if part.start:
                text_timestamp_index[part.seconds] = {"text_start_idx": text_len_so_far}
            else:
                text_timestamp_index[part.seconds]["text_end_idx"] = text_len_so_far
        else:
            text_len_so_far += len(part)
            text_only_parts.append(part)

    text_timestamps = []
    text_timestamp_start_locs = []
    text_timestamp_end_locs = []
    for ts, start_end_pair in text_timestamp_index.items():
        text_timestamps.append(ts)
        text_timestamp_start_locs.append(start_end_pair["text_start_idx"])
        text_timestamp_end_locs.append(start_end_pair["text_end_idx"])

    timestamp_index_df = pd.DataFrame(
        {
            "start_loc": text_timestamp_start_locs,
            "end_loc": text_timestamp_end_locs,
            "timestamp": text_timestamps,
        },
    ).sort_values("timestamp")

    all_processed_text = "".join(text_only_parts)

    return all_processed_text, timestamp_index_df


def close_prev_committee_time_marker(
    text_parts: list[str | TimeMarker],
    last_marker_bookmark_id: int | None,
    time_stamp: float,
    text_length_in_timestampped_range: int,
):
    # Close previous marker.
    # Internal time marker bookmarks are not capturing text - the "range end"
    # of one is actually the beginning of another.
    end_time_marker = TimeMarker(start=False, seconds=time_stamp, id=last_marker_bookmark_id)
    text_parts.append(end_time_marker)

    # If the closed range does not contain any text
    # Remove those redundant markers
    if text_length_in_timestampped_range == 0:
        # Go backwards over the text parts and remove the end+start markers
        # For that id
        search_iterations_left = 100  # Safety mechism - not expecting to reach this
        inspecting_index = -1
        removed_start = False
        removed_end = False
        while (
            (
                # still need to remove the start or the end
                not removed_start
                or not removed_end
            )
            and search_iterations_left > 0  # No more allowed searches
            and -inspecting_index <= len(text_parts)  # Nothing left to search in
        ):
            removed = False
            if isinstance(text_parts[inspecting_index], TimeMarker):
                tm: TimeMarker = text_parts[inspecting_index]
                if tm.id == last_marker_bookmark_id:
                    if tm.start:
                        removed_start = True
                    else:
                        removed_end = True
                    text_parts.pop(inspecting_index)
                    removed = True

            # move on. If removed - we keep the index - now pointing at the next element
            if not removed:
                inspecting_index -= 1

            search_iterations_left -= 1


def parse_committee_timestampped_protocol_doc(
    committee_id_target_folder: pathlib.Path,
) -> tuple[str, pd.DataFrame]:
    document = Document(committee_id_target_folder / cached_protocol_doc_file_name)
    main_text_gather = []
    # Planned usage - speakers and other meta tags (Missing time ranges)
    # Currently not used in the pipeline
    ancillary_text_gather = []
    ancillary_text_bookmark_id = None
    range_start_time_stamp = None
    text_length_in_timestampped_range = 0
    time_marker_bookmark_id = None
    forced_ancillary_text_range_pattern = False

    # Dump entire XML doc to a readable file
    # Mainly for debug purposes or downstream post processing
    with open(committee_id_target_folder / cached_protocol_doc_xml_content_file_name, "w") as f:
        f.write(document.element.xml)

    for _parag in document.paragraphs:
        parag: Paragraph = _parag
        had_text_parts = False
        for c in parag._p.getchildren():
            if isinstance(c, CT_Bookmark):
                if etype_time_bookmark_name_matcher.match(c.name):
                    if range_start_time_stamp is not None:
                        close_prev_committee_time_marker(
                            main_text_gather,
                            time_marker_bookmark_id,
                            range_start_time_stamp,
                            text_length_in_timestampped_range,
                        )
                        range_start_time_stamp = None

                    text_length_in_timestampped_range = 0
                    range_start_time_stamp = int(c.name.split(etype_time_bookmark_split_pattern)[1]) / 1000.0
                    time_marker_bookmark_id = c.id
                    start_time_marker = TimeMarker(start=True, seconds=range_start_time_stamp, id=c.id)
                    main_text_gather.append(start_time_marker)
                elif etype_bookmark_name_matcher.match(c.name):
                    ancillary_text_bookmark_id = c.id
            elif isinstance(c, CT_MarkupRange):
                # This only applies to ancillary text.
                # Time marker do not encapsulate text - they are markers and not real ranges
                # so immediately close.
                if ancillary_text_bookmark_id == c.id:
                    ancillary_text_bookmark_id = None
                    ancillary_text_gather.append("\n")
            elif isinstance(c, CT_R):
                # the text, sometimes is wrongly not wrapped in a bookmark although
                # it describes a speaker (for example)
                # Check this for extra cleaning
                is_bookmark_like_text_pattern = etype_missed_bookmark_text_pattern.search(str(c.text))
                need_to_close_line_in_ancillary_text = False
                if is_bookmark_like_text_pattern and ancillary_text_bookmark_id is None:
                    if forced_ancillary_text_range_pattern:
                        forced_ancillary_text_range_pattern = False
                        need_to_close_line_in_ancillary_text = True
                    else:
                        forced_ancillary_text_range_pattern = True

                if ancillary_text_bookmark_id is None and not (
                    forced_ancillary_text_range_pattern or is_bookmark_like_text_pattern
                ):
                    main_text_gather.append(c.text)
                    text_length_in_timestampped_range += len(normalize_text_as_caption_text(c.text))
                    had_text_parts = True
                else:
                    ancillary_text_gather.append(c.text)

                if need_to_close_line_in_ancillary_text:
                    ancillary_text_gather.append("\n")
                    need_to_close_line_in_ancillary_text = False
            elif isinstance(c, CT_PPr):
                pass
            elif c.tag.endswith("proofErr"):
                pass
            else:
                # Not expected to encounter unknown elements - but in case
                # this happens - have some data to start debugging
                print("Parsing unexpected element in Docx XML: (Igoring)")
                print(c, c.tag[-10:])

        # finalize paragraph processing

        # Clear states which are expected to be closed
        # but might leak out of a paragraph
        forced_ancillary_text_range_pattern = False
        if ancillary_text_bookmark_id is not None:
            # Sometimes, data bugs may not properly close.
            ancillary_text_bookmark_id = None
            ancillary_text_gather.append("\n")

        # Append a new line for readability of any text parts had been observed
        if had_text_parts:
            main_text_gather.append("\n")
        had_text_parts = False

    # Close any open time marker ranges
    if range_start_time_stamp:
        close_prev_committee_time_marker(
            main_text_gather,
            time_marker_bookmark_id,
            range_start_time_stamp,
            text_length_in_timestampped_range,
        )
        range_start_time_stamp = None
        text_length_in_timestampped_range = 0

    # Try and unify too-small time marked ranges
    anchor_start_marker_idx = None
    anchor_start_marker_id = None
    anchor_start_marker_seconds = 0
    anchor_end_marker_idx = None

    to_merge_start_marker_idx = None
    to_merge_start_marker_id = None
    to_merge_start_marker_seconds = 0
    to_merge_end_marker_idx = None

    to_merge_marker_contains_non_whitespace = False
    for pidx, part in enumerate(main_text_gather):
        # Keep Track of time markers
        if isinstance(part, TimeMarker):
            ## Capture phase

            # When encountering a start marker
            if part.start:
                # If we are not currently pointing at an existing start marker
                # Capture this start marker
                if anchor_start_marker_idx is None:
                    anchor_start_marker_idx = pidx
                    anchor_start_marker_id = part.id
                    anchor_start_marker_seconds = part.seconds
                elif to_merge_start_marker_idx is None:
                    to_merge_start_marker_idx = pidx
                    to_merge_start_marker_id = part.id
                    to_merge_start_marker_seconds = part.seconds
                    to_merge_marker_contains_non_whitespace = False
            else:
                if anchor_start_marker_id == part.id:
                    anchor_end_marker_idx = pidx
                elif to_merge_start_marker_id == part.id:
                    to_merge_end_marker_idx = pidx

            ## Merge phase

            # Must have all 4 markers to consider the merge
            if (
                anchor_start_marker_idx is not None
                and anchor_end_marker_idx is not None
                and to_merge_start_marker_idx is not None
                and to_merge_end_marker_idx is not None
            ):
                anchor_len_seconds = to_merge_start_marker_seconds - anchor_start_marker_seconds
                if (
                    anchor_len_seconds >= 0 and anchor_len_seconds < 3  # Too short
                ) or not to_merge_marker_contains_non_whitespace:  # Only whitespaces
                    # "Swallow" the to_merge segment by moving the anchor end to the end
                    main_text_gather[to_merge_end_marker_idx] = main_text_gather[anchor_end_marker_idx]
                    # Replace two mid markers with spaces to keep the list indexing proper
                    main_text_gather[anchor_end_marker_idx] = ""
                    main_text_gather[to_merge_start_marker_idx] = ""
                    # New end is at the end of the merged segment
                    anchor_end_marker_idx = to_merge_end_marker_idx
                # Else - no need to merge
                else:
                    # The new anchor moves fwd, the to_merge resets
                    anchor_start_marker_idx = to_merge_start_marker_idx
                    anchor_start_marker_id = to_merge_start_marker_id
                    anchor_start_marker_seconds = to_merge_start_marker_seconds
                    anchor_end_marker_idx = to_merge_end_marker_idx

                # Reset the to_merge pointers so they can accept the next
                # potential segment
                to_merge_start_marker_idx = None
                to_merge_start_marker_id = None
                to_merge_start_marker_seconds = 0
                to_merge_end_marker_idx = None
        else:
            # Text parts which have only spaces will force the marker to merge
            # it's content with the previous marker. VTT cannot read such text parts.
            part_text = str(part)
            if part_text.strip():
                to_merge_marker_contains_non_whitespace = True

    # Seperate text parts from to text and time index
    text_timestamp_index = {}
    text_only_parts = []
    text_len_so_far = 0
    for part in main_text_gather:
        if isinstance(part, TimeMarker):
            if part.start:
                text_timestamp_index[part.seconds] = {"text_start_idx": text_len_so_far}
            else:
                text_timestamp_index[part.seconds]["text_end_idx"] = text_len_so_far
        else:
            text_len_so_far += len(part)
            text_only_parts.append(part)

    text_timestamps = []
    text_timestamp_start_locs = []
    text_timestamp_end_locs = []
    for ts, start_end_pair in text_timestamp_index.items():
        text_timestamps.append(ts)
        text_timestamp_start_locs.append(start_end_pair["text_start_idx"])
        text_timestamp_end_locs.append(start_end_pair["text_end_idx"])

    timestamp_index_df = pd.DataFrame(
        {
            "start_loc": text_timestamp_start_locs,
            "end_loc": text_timestamp_end_locs,
            "timestamp": cleanup_html_time_map_arr(np.asarray(text_timestamps)),
        },
    ).sort_values("timestamp")

    all_processed_text = "".join(text_only_parts)

    return all_processed_text, timestamp_index_df


def normalize_text_as_caption_text(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def create_caption(text: str, start: float, end: float) -> Caption:
    return Caption(
        # Format to hh:mm:ss.zzz
        start=strftime("%H:%M:%S.000", gmtime(start)),
        end=strftime("%H:%M:%S.000", gmtime(end)),
        text=normalize_text_as_caption_text(text),
    )


def create_recording_transcript_vtt(text: str, time_index: pd.DataFrame) -> WebVTT:
    vtt = WebVTT()
    time_index = time_index.copy()  # assumed to be sorted
    time_index["next_ts"] = time_index.timestamp.shift(-1).ffill()

    # Before any caption of text - there might be some "non timed text" that we want to keep
    # as the first "caption"
    if 0 < time_index.iloc[0]["timestamp"]:
        vtt.captions.append(
            create_caption(
                text[slice(0, int(time_index.iloc[0]["start_loc"]))],
                0,
                time_index.iloc[0]["timestamp"],
            )
        )

    merging_next = False
    merge_start_loc = None
    for _, data in time_index.iterrows():
        # merge consecutive slices sharing the same TS
        if data["next_ts"] == data["timestamp"]:
            merge_start_loc = merge_start_loc or data["start_loc"]
            merging_next = True
        else:
            slice_start_loc = data["start_loc"]
            if merging_next:
                merging_next = False
                slice_start_loc = merge_start_loc
                merge_start_loc = None

            text_range_slice = slice(int(slice_start_loc), int(data["end_loc"]))
            vtt.captions.append(create_caption(text[text_range_slice], data["timestamp"], data["next_ts"]))
    return vtt


def cleanup_html_time_map_arr(time_map_arr: np.ndarray) -> np.ndarray:
    min_referenced_time_idx = max(
        1, np.argmin(time_map_arr[time_map_arr > 0])
    )  # find index of the lowest non-0 time value is at
    # remove leading high values before that minimal value- reset to 0 to ignore those "future ts" which are bad data
    time_map_arr[:min_referenced_time_idx] = 0

    # go over values in ascending order - if it deviates too much - use the prev value instead
    # TODO - calc those numbers from diff histograms - or find a more robust algorithm to clean this up
    max_backwards_jump = 0
    max_forward_jump = 30
    for ith in range(min_referenced_time_idx, len(time_map_arr) - 1):
        if (
            time_map_arr[ith] - time_map_arr[ith + 1] > max_backwards_jump
        ):  # backwards time jump - over tolerance - fix it
            time_map_arr[ith + 1] = time_map_arr[ith]
        elif time_map_arr[ith + 1] - time_map_arr[ith] > max_forward_jump:  # Forward jump is too big - fix it
            # Skip the bad value and look forward a bit to get an estimate
            # of where the series continues
            future_baseline = np.median(time_map_arr[ith + 2 : ith + 10 : 2])
            # If the next value is closer to the future baseline - use it
            # despite it being an abnormal jump - perhaps this jump is required.
            if abs(time_map_arr[ith + 1] - future_baseline) < abs(time_map_arr[ith] - future_baseline):
                continue

            # Otherwise - fix it by taking the max allowed jump
            time_map_arr[ith + 1] = time_map_arr[ith] + max_forward_jump

    return time_map_arr


def get_plenum_video_resource_url(
    knesset_http_headers: dict,
    plenum_id_target_folder: pathlib.Path,
    plenum_recording_id: str,
) -> str:
    # If the video file url is cached - use that
    if pathlib.Path(plenum_id_target_folder / cached_video_source_url_file_name).exists():
        with open(plenum_id_target_folder / cached_video_source_url_file_name, "r") as f:
            video_resource_url = f.read()
    else:
        # Get the video path
        response = requests.post(
            f"{knesset_protocol_video_path_api_url}?protocolNum={plenum_recording_id}",
            data=plenum_recording_id,
            headers={**knesset_http_headers, "Content-Type": "application/json"},
        )
        video_resource_url = json.loads(response.text)

        if not video_resource_url:
            raise ValueError(
                f"Could not find video resource url for plenum recording {plenum_recording_id} - Is this a valid plenum ID?"
            )

        # cache the url
        with open(plenum_id_target_folder / cached_video_source_url_file_name, "w") as f:
            f.write(video_resource_url)

    return video_resource_url


def get_committee_video_resource_url(
    knesset_http_headers: dict,
    committee_id_target_folder: pathlib.Path,
    committee_session_id: str,
) -> str:
    # If the video file url is cached - use that
    if pathlib.Path(committee_id_target_folder / cached_video_source_url_file_name).exists():
        with open(committee_id_target_folder / cached_video_source_url_file_name, "r") as f:
            video_resource_url = f.read()
    else:
        print("Resolving committee session TopicID.")
        # Get the TopicID of the session - this identifies
        # the page that contains the MPEG-dash manifest URL
        odata_client = create_parliament_odata_client()
        committee_session_entity = odata_client.entity_sets.KNS_CommitteeSession.get_entity(
            int(committee_session_id)
        ).execute()
        video_recording_view_page_path = urlparse(committee_session_entity.BroadcastUrl)
        video_recording_topic_id = parse_qs(video_recording_view_page_path.query).get("TopicID", None)
        # this URL is actually broken - we only take the session topic id from it.
        video_recording_topic_id = video_recording_topic_id[0] if video_recording_topic_id is not None else None

        # Prepare the page we will scrape for the mpeg-dash manifest URL
        # Apparently - the "folder" of the committee does not matter - the "TopicID" works anyway. (hmm)
        # We need to load this page since it contains the edited - non "inferrable" video stream url
        # (There exists and easy way to get the non edited video recording, but it is almost always missing the first part and has extra junk at the end)
        broadcast_pointer_page_url = f"{committee_broadcast_player_page_base_url}?TopicID={video_recording_topic_id}"

        print("Scraping broadcast page by TopicID for video URL.")
        broadcast_pointer_page_html_content = requests.get(
            broadcast_pointer_page_url,
            headers=knesset_http_headers,
        )

        soup = BeautifulSoup(broadcast_pointer_page_html_content.content, "html.parser")

        broadcast_js_call_matcher = re.compile(r"javascript:SetAzurePlayerFileName")
        broadcast_video_feed_extractor = re.compile(r"javascript:SetAzurePlayerFileName\('(.*?)',")

        # We need to find a node that is an anchor and point to a JS call looking like this:
        # <a href="javascript:SetAzurePlayerFileName(
        anchor_elem = soup.find(
            "a",
            href=broadcast_js_call_matcher,
        )
        js_func_call_href = anchor_elem.attrs["href"]
        video_resource_url = broadcast_video_feed_extractor.match(js_func_call_href).group(1)

        # Found the Video Feed - this is an MPEG-Dash MPD file

        # cache the url
        with open(committee_id_target_folder / cached_video_source_url_file_name, "w") as f:
            f.write(video_resource_url)

    return video_resource_url


def get_html_ts_map(
    knesset_http_headers: dict,
    plenum_id_target_folder: pathlib.Path,
    plenum_recording_id: str,
):
    # check if a cache file of the HTML TS map exists
    if pathlib.Path(plenum_id_target_folder / cached_html_ts_map_file_name).exists():
        print("Reusing cached HTML->Timestamp mapping...")
        html_ts_map_js_array = np.load(plenum_id_target_folder / cached_html_ts_map_file_name)
    else:
        print("Loading HTML->Timestamp mapping...")

        # Load the HTML node id to timestamp map as a JS array
        node_id_timestamp_mapping_response = requests.get(
            f"{knesset_protocol_html_ts_map_api_url}?sProtocolNum={plenum_recording_id}",
            headers=knesset_http_headers,
        )
        html_ts_map_js_array = np.array(json.loads(node_id_timestamp_mapping_response.text))

        print("Cleaning up the HTML-Timestamp mapping...")
        html_ts_map_js_array = cleanup_html_time_map_arr(html_ts_map_js_array)

        # save a cached copy of the HTML TS map
        np.save(
            plenum_id_target_folder / cached_html_ts_map_file_name,
            html_ts_map_js_array,
        )

    return html_ts_map_js_array


def get_html_transcript(
    knesset_http_headers: dict,
    plenum_id_target_folder: pathlib.Path,
    plenum_recording_id: str,
):
    # If a temp HTML traanscript file exists, use that
    if pathlib.Path(plenum_id_target_folder / cached_transcript_html_file_name).exists():
        print("Reusing cached transcript HTML file from previouse run")
        with open(plenum_id_target_folder / cached_transcript_html_file_name, "r") as transcript_html_file:
            html_transcript = transcript_html_file.read()
    else:
        # Load how many content pages exists
        print("Determining the count of content pages.")
        page_count_response = requests.get(
            f"{knesset_protocol_page_count_api_url}?protocolNum={plenum_recording_id}",
            headers=knesset_http_headers,
        )
        page_count = int(page_count_response.text)  # length of page sequence

        # download the content for each page
        pages_content = []
        print(f"Downloading {page_count} pages of transcript content.")
        for page_idx in range(page_count):
            print(f"Downloading transcript HTML content - page {page_idx + 1}")

            page_content_response = requests.get(
                f"{knesset_protocol_page_content_api_url}?sProtocolNum={plenum_recording_id}&pageNum={page_idx}",
                headers=knesset_http_headers,
            )
            page_content = json.loads(page_content_response.text)
            pages_content.append(page_content)

        # Process all page contents into an output HTML snippet
        output_html_parts = [page_content.get("sContent") for page_content in pages_content]
        html_transcript_snippet = "".join(output_html_parts)
        html_transcript = f'<div id="root">{html_transcript_snippet}</div>'

        # Store the HTML transcript artifact in case a retry is needed
        with open(plenum_id_target_folder / cached_transcript_html_file_name, "w") as f:
            f.write(html_transcript)

    return html_transcript


def get_committee_protocol_doc_url(
    committee_id_target_folder: pathlib.Path,
    committee_session_id: str,
) -> str:
    # If the protocol doc file url is cached - use that
    if pathlib.Path(committee_id_target_folder / cached_protocol_doc_source_url_file_name).exists():
        with open(committee_id_target_folder / cached_protocol_doc_source_url_file_name, "r") as f:
            protocol_doc_download_path = f.read()
    else:
        odata_client = create_parliament_odata_client()
        committee_session_docs_query = odata_client.entity_sets.KNS_DocumentCommitteeSession.get_entities()
        committee_session_docs_query = committee_session_docs_query.filter(
            esf.and_(
                committee_session_docs_query.CommitteeSessionID == int(committee_session_id),
                committee_session_docs_query.GroupTypeID == committee_protocol_document_type_id,
            )
        )

        protocol_doc_download_path = None
        for doc in committee_session_docs_query.execute():
            protocol_doc_download_path = doc.FilePath
            break

        # Download and store the protocol file
        if protocol_doc_download_path is not None:
            with open(
                committee_id_target_folder / cached_protocol_doc_source_url_file_name,
                "w",
            ) as f:
                f.write(protocol_doc_download_path)

    return protocol_doc_download_path


def get_commitee_protocol_doc(
    committee_id_target_folder: pathlib.Path,
    protocol_doc_download_url: str,
):
    # If the protocol doc is cached - use that
    if pathlib.Path(committee_id_target_folder / cached_protocol_doc_file_name).exists():
        return

    # Download the protocol doc
    with open(committee_id_target_folder / cached_protocol_doc_file_name, "wb") as f:
        f.write(requests.get(protocol_doc_download_url).content)


def download_video_recording(
    plenum_id_target_folder: pathlib.Path,
    video_resource_url: str,
):
    # Check if the output video file already exists
    if not pathlib.Path(plenum_id_target_folder / output_video_file_name).exists():
        # Download the vidoe file
        print("Downloading video file...")
        command_parts = [
            "aria2c",
            "-x",
            "16",
            "-s",
            "16",
            "-k",
            "40M",
            "-q",
            video_resource_url,
            "-d",
            str(plenum_id_target_folder),
            "-o",
            output_video_file_name,
        ]
        subprocess.check_call(command_parts)
    else:
        print("Video file already downloaded. Skipping.")


def download_video_mpeg_dash_stream(
    committee_id_target_folder: pathlib.Path,
    video_resource_url: str,
):
    # Check if the output video file already exists
    if not pathlib.Path(committee_id_target_folder / output_video_file_name).exists():
        print("Downloading video file from MPEG-dash stream.")
        download_and_convert_command_args = [
            "ffmpeg",
            "-y",
            "-nostdin",
            "-nostats",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            video_resource_url,
            str(committee_id_target_folder / output_video_file_name),
        ]
        subprocess.check_call(download_and_convert_command_args)
    else:
        print("Video file already downloaded. Skipping.")


def transcode_video_to_audio(
    id: str,
    target_folder: pathlib,
    id_target_folder: pathlib,
):
    if not pathlib.Path(id_target_folder / output_video_file_name).exists():
        print("No video file exists - skipping audio transcoding.")
        return

    output_audio_file_name = f"{id}.mp3"
    if not pathlib.Path(target_folder / output_audio_file_name).exists():
        # Transcode the video to an audio file
        print("Transcoding video to audio...")
        convert_command_args = [
            "ffmpeg",
            "-y",
            "-nostdin",
            "-nostats",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(id_target_folder / output_video_file_name),
            "-vn",
            "-acodec",
            "mp3",
            str(target_folder / output_audio_file_name),
        ]
        subprocess.check_call(convert_command_args)
    else:
        print("Audio file already exists. Skipping transcoding from video file.")


def download_plenum(
    knesset_http_headers: dict,
    id: str,
    target_dir: pathlib.Path,
    skip_download_video_file: bool = False,
):
    # Stores the output audio an transcript - with name based on the plenum id
    plenum_target_folder = target_dir / "plenum"
    # Stores caches, intermediary, and other artifacts for this id
    plenum_id_target_folder = target_dir / "plenum" / id

    if not pathlib.Path(plenum_id_target_folder).exists():
        plenum_id_target_folder.mkdir(parents=True, exist_ok=True)

    video_resource_url = get_plenum_video_resource_url(knesset_http_headers, plenum_id_target_folder, id)

    print(f"Plenum video resource url: {video_resource_url} (Not downloading yet..)")

    html_ts_map_js_array = get_html_ts_map(knesset_http_headers, plenum_id_target_folder, id)

    html_transcript = get_html_transcript(knesset_http_headers, plenum_id_target_folder, id)

    print("Parsing HTML transcript...")
    transcript_text, transcript_time_index_df = parse_plenum_transcript(html_transcript, html_ts_map_js_array)

    # Store the output artifacts
    with open(plenum_id_target_folder / output_transcript_text_file_name, "w") as f:
        f.write(transcript_text)
    transcript_time_index_df.to_csv(plenum_id_target_folder / output_transcript_time_index_file_name)

    print("Creating a VTT file of the transcript...")
    vtt = create_recording_transcript_vtt(transcript_text, transcript_time_index_df)
    vtt.save(plenum_target_folder / f"{id}.vtt", add_bom=True)

    if not skip_download_video_file:
        download_video_recording(plenum_id_target_folder, video_resource_url)
    else:
        print("Skipping video file download.")

    transcode_video_to_audio(id, plenum_target_folder, plenum_id_target_folder)

    print(f"Download plenum {id} done.")


def download_committee_session(
    knesset_http_headers: dict,
    session_id: str,
    target_dir: pathlib.Path,
    skip_download_video_file: bool = False,
):
    # Stores the output audio and transcript - with name based on the committee id
    committee_target_folder = target_dir / "committee"
    # Stores caches, intermediary, and other artifacts for this id
    committee_id_target_folder = target_dir / "committee" / session_id

    if not pathlib.Path(committee_id_target_folder).exists():
        committee_id_target_folder.mkdir(parents=True, exist_ok=True)

    video_resource_url = get_committee_video_resource_url(knesset_http_headers, committee_id_target_folder, session_id)

    print(f"Committee video resource url: {video_resource_url} (Not downloading yet..)")

    protocol_doc_download_url = get_committee_protocol_doc_url(committee_id_target_folder, session_id)

    if protocol_doc_download_url is None:
        print(f"No protocol doc download url found. aborting download of session id: {session_id}")
        return

    print(f"Committee protocol doc resource url: {protocol_doc_download_url}, downloading...")

    get_commitee_protocol_doc(committee_id_target_folder, protocol_doc_download_url)

    print("Parsing protocol doc transcript...")
    transcript_text, transcript_time_index_df = parse_committee_timestampped_protocol_doc(committee_id_target_folder)

    # Store the output artifacts
    with open(committee_id_target_folder / output_transcript_text_file_name, "w") as f:
        f.write(transcript_text)
    transcript_time_index_df.to_csv(committee_id_target_folder / output_transcript_time_index_file_name)

    print("Creating a VTT file of the transcript...")
    vtt = create_recording_transcript_vtt(transcript_text, transcript_time_index_df)
    vtt.save(committee_target_folder / f"{session_id}.vtt", add_bom=True)

    if not skip_download_video_file:
        download_video_mpeg_dash_stream(committee_id_target_folder, video_resource_url)
    else:
        print("Skipping video file download.")

    transcode_video_to_audio(session_id, committee_target_folder, committee_id_target_folder)

    print(f"Download committee session {session_id} done.")


def download(args):
    """Downloads a Knesset recording."""
    target_dir = pathlib.Path(args.target_dir)

    if args.type == "plenum":
        knesset_http_headers = load_http_headers(args)
        set_http_headers_referer(knesset_http_headers, base_knesset_app_url)
        for id in args.ids:
            download_plenum(
                knesset_http_headers,
                id,
                target_dir,
                skip_download_video_file=args.skip_video_download,
            )
    elif args.type == "committee":
        knesset_http_headers = load_http_headers(args)
        set_http_headers_referer(knesset_http_headers, base_knesset_committee_url)
        for id in args.ids:
            download_committee_session(
                knesset_http_headers,
                id,
                target_dir,
                skip_download_video_file=args.skip_video_download,
            )
    else:
        raise Exception(f"Unknown Knesset recording type: {args.type}")


if __name__ == "__main__":
    # Define an argument parser
    parser = argparse.ArgumentParser(
        description="""Download Knesset recordings and extract existing transcripts.""",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Add the arguments
    parser.add_argument(
        "--type",
        type=str,
        required=False,
        default="plenum",
        choices=["plenum", "committee"],
        help="Type of recording.",
    )
    parser.add_argument(
        "--ids",
        action="append",
        type=str,
        required=False,  # Manually checked later
        help=f"Ids of the recording to download, for the type requested.\n{HOW_TO_GET_PLENUM_ID}",
    )
    parser.add_argument(
        "--http-headers-file",
        type=str,
        required=True,
        help=f"The http headers to use for Knessets access.\n{HOW_TO_GET_HTTP_HEADERS}",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        required=False,  # Manually checked later
        help="The directory where episodes will be stored.",
    )
    parser.add_argument(
        "--skip-video-download",
        action="store_true",
        required=False,
        help="Skip downloading the video file.",
    )
    parser.add_argument(
        "--only-check-http-access",
        action="store_true",
        required=False,
        help="Check if a valid HTTP access exists - report and abort without any processing",
    )

    # Parse the arguments
    args = parser.parse_args()

    if not args.only_check_http_access and (
        0 == len(args.ids or []) or not args.target_dir or not args.http_headers_file
    ):
        parser.error("--ids and --target-dir and --http-headers-file are required.")

    if args.only_check_http_access:
        print("Checking HTTP access...")
        http_headers = load_http_headers(args)
        if check_http_access(http_headers):
            print("OK - HTTP access is working as expected.")
            exit(0)
        else:
            print("Not OK - HTTP access is not working as expected.")
            exit(1)

    download(args)
