#!/usr/bin/python3

import argparse
import json
import pathlib
import re
import subprocess
from time import gmtime, strftime

from bs4 import BeautifulSoup, PageElement, NavigableString
import numpy as np
import pandas as pd
import requests
from webvtt import WebVTT, Caption

RECORDING_TYPES = ["plenum"]

base_kenesset_url = "https://online.knesset.gov.il"
base_knesset_app_url = f"{base_kenesset_url}/app/"
plenum_player_site_base_url = f"{base_kenesset_url}/app/#/player/peplayer.aspx"
knesset_protocol_ajax_api_base_url = f"{base_kenesset_url}/api/Protocol"
knesset_protocol_video_path_api_url = (
    f"{knesset_protocol_ajax_api_base_url}/GetVideoPath"
)
knesset_protocol_html_ts_map_api_url = (
    f"{knesset_protocol_ajax_api_base_url}/WriteBKArrayData"
)
knesset_protocol_page_count_api_url = (
    f"{knesset_protocol_ajax_api_base_url}/GetBlunkCount"
)
knesset_protocol_page_content_api_url = (
    f"{knesset_protocol_ajax_api_base_url}/GetProtocolBulk"
)


cached_video_source_url_file_name = "cache.video.url"
cached_transcript_html_file_name = "cache.transcript.html"
cached_html_ts_map_file_name = "cache.htmltsmap.npy"

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

extract_header_user_agent = re.compile(r"User-Agent: (.*)")
extract_header_cookie = re.compile(r"Cookie: (.*)")


def load_http_headers(args) -> dict:
    if args.http_headers_file:
        with open(args.http_headers_file, "r") as f:
            http_headers_manual_data = f.read()

            return {
                "User-Agent": extract_header_user_agent.search(
                    http_headers_manual_data
                ).group(1),
                "Cookie": extract_header_cookie.search(http_headers_manual_data).group(
                    1
                ),
                "Referer": "https://online.knesset.gov.il/app",
                "Host": "online.knesset.gov.il",
            }


def check_http_access(knesset_http_headers: dict):
    try:
        print("Requesting main site page...")
        print("Using HTTP headers:")
        print(json.dumps(knesset_http_headers, indent=2))
        response = requests.get(base_knesset_app_url, headers=knesset_http_headers)
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


class TimeMarker:
    def __init__(self, start=True, seconds=0) -> None:
        self.start = start
        self.seconds = seconds

    # Support serialziing as string
    def __repr__(self) -> str:
        return f"[[{'S' if self.start else 'E'}{self.seconds:g}]]"


text_idx_extractor_pattern = re.compile("txt_(\d+)")


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
            text_idx_matches = text_idx_extractor_pattern.findall(root_element["id"])
            text_ts_idx = None

            if len(text_idx_matches) > 0:
                text_ts_idx = int(text_idx_matches[0])

            if text_ts_idx is not None:
                prev_text_ts: pd.Timedelta = context["current_text_ts"]
                time_at_idx: pd.Timedelta = pd.to_timedelta(
                    context["time_pointers_arr_np"][text_ts_idx], "s"
                )

                if prev_text_ts is None or context["current_text_ts"] < time_at_idx:
                    context["current_text_ts"] = time_at_idx
                    extracted_list = [
                        TimeMarker(True, time_at_idx.total_seconds()),
                        *extracted_list,
                    ]

                # if we moved to a new TS - add an end marker before the begin marker
                if (
                    prev_text_ts is not None
                    and prev_text_ts.total_seconds() > 0
                    and prev_text_ts < time_at_idx
                ):
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
        raise ValueError(
            f"What should I do with type: {type(root_element)} and str: {str(root_element)}"
        )


def parse_plenum_transcript(content, time_pointers_arr_np) -> tuple[str, pd.DataFrame]:
    soup = BeautifulSoup(content, "html.parser")

    page_text_parts = []
    extraction_context = {
        "time_pointers_arr_np": time_pointers_arr_np,
        "current_text_ts": None,
        "latest_closed_ts_marker": 0,
    }
    root_element = soup.find("div", {"id": "root"})
    page_text_parts.extend(
        extract_transcript_parts_from_elements(root_element, extraction_context)
    )

    # Ensure last marker is proplely closed
    if (
        extraction_context["latest_closed_ts_marker"]
        < extraction_context["current_text_ts"]
    ):
        page_text_parts.append(
            TimeMarker(False, extraction_context["current_text_ts"].total_seconds())
        )

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


def noramzlie_text_as_caption_text(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def create_caption(text: str, start: float, end: float) -> Caption:
    return Caption(
        # Format to hh:mm:ss.zzz
        start=strftime("%H:%M:%S.000", gmtime(start)),
        end=strftime("%H:%M:%S.000", gmtime(end)),
        text=noramzlie_text_as_caption_text(text),
    )


def create_plenum_transcript_vtt(text: str, time_index: pd.DataFrame) -> WebVTT:
    vtt = WebVTT()
    time_index = time_index.copy()
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

    for _, data in time_index.iterrows():
        text_range_slice = slice(int(data["start_loc"]), int(data["end_loc"]))
        vtt.captions.append(
            create_caption(text[text_range_slice], data["timestamp"], data["next_ts"])
        )
    return vtt


def cleanup_html_time_map_arr(time_map_arr: np.ndarray) -> np.ndarray:
    min_referenced_time_idx = max(
        1, np.argmin(time_map_arr[time_map_arr > 0])
    )  # find index of the lowest non-0 time value is at
    # remove leading high values before that minimal value- reset to 0 to ignore those "future ts" which are bad data
    time_map_arr[:min_referenced_time_idx] = 0

    # go over values in ascending order - if it deviates too much - use the prev value instead
    # TODO - calc those numbers from diff histograms - or find a more robust algorithm to clean this up
    max_backwards_jump = 15
    max_forward_jump = 30
    for ith in range(min_referenced_time_idx, len(time_map_arr) - 1):
        if (
            time_map_arr[ith] - time_map_arr[ith + 1] > max_backwards_jump
        ):  # backwards time jump - over tolerance - fix it
            time_map_arr[ith + 1] = time_map_arr[ith]
        elif (
            time_map_arr[ith + 1] - time_map_arr[ith] > max_forward_jump
        ):  # Forward jump is too big - fix it
            # Skip the bad value and look forward a bit to get an estimate
            # of where the series continues
            future_baseline = np.median(time_map_arr[ith + 2 : ith + 10 : 2])
            # If the next value is closer to the future baseline - use it
            # despite it being an abnormal jump - perhaps this jump is required.
            if abs(time_map_arr[ith + 1] - future_baseline) < abs(
                time_map_arr[ith] - future_baseline
            ):
                continue

            # Otherwise - fix it by taking the max allowed jump
            time_map_arr[ith + 1] = time_map_arr[ith] + max_forward_jump

    return time_map_arr


def get_video_resource_url(
    knesset_http_headers: dict,
    plenum_id_target_folder: pathlib.Path,
    plenum_recording_id: str,
) -> str:
    # If the video file url is cached - use that
    if pathlib.Path(
        plenum_id_target_folder / cached_video_source_url_file_name
    ).exists():
        with open(
            plenum_id_target_folder / cached_video_source_url_file_name, "r"
        ) as f:
            video_resource_url = f.read()
    else:
        # Get the video path
        response = requests.post(
            f"{knesset_protocol_video_path_api_url}?protocolNum={plenum_recording_id}",
            data=plenum_recording_id,
            headers={**knesset_http_headers, "Content-Type": "application/json"},
        )
        video_resource_url = json.loads(response.text)

        # cache the url
        with open(
            plenum_id_target_folder / cached_video_source_url_file_name, "w"
        ) as f:
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
        html_ts_map_js_array = np.load(
            plenum_id_target_folder / cached_html_ts_map_file_name
        )
    else:
        print("Loading HTML->Timestamp mapping...")

        # Load the HTML node id to timestamp map as a JS array
        node_id_timestamp_mapping_response = requests.get(
            f"{knesset_protocol_html_ts_map_api_url}?sProtocolNum={plenum_recording_id}",
            headers=knesset_http_headers,
        )
        html_ts_map_js_array = np.array(
            json.loads(node_id_timestamp_mapping_response.text)
        )

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
    if pathlib.Path(
        plenum_id_target_folder / cached_transcript_html_file_name
    ).exists():
        print("Reusing cached transcript HTML file from previouse run")
        with open(
            plenum_id_target_folder / cached_transcript_html_file_name, "r"
        ) as transcript_html_file:
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
        output_html_parts = [
            page_content.get("sContent") for page_content in pages_content
        ]
        html_transcript_snippet = "".join(output_html_parts)
        html_transcript = f'<div id="root">{html_transcript_snippet}</div>'

        # Store the HTML transcript artifact in case a retry is needed
        with open(plenum_id_target_folder / cached_transcript_html_file_name, "w") as f:
            f.write(html_transcript)

    return html_transcript


def download_video_recording(
    plenum_id_target_folder: pathlib,
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


def transcode_video_to_audio(
    id: str,
    plenum_target_folder: pathlib,
    plenum_id_target_folder: pathlib,
):
    if not pathlib.Path(plenum_id_target_folder / output_video_file_name).exists():
        print("No video file exists - skipping audio transcoding.")
        return

    output_audio_file_name = f"{id}.mp3"
    if not pathlib.Path(plenum_target_folder / output_audio_file_name).exists():
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
            str(plenum_id_target_folder / output_video_file_name),
            "-vn",
            "-acodec",
            "mp3",
            str(plenum_target_folder / output_audio_file_name),
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

    video_resource_url = get_video_resource_url(
        knesset_http_headers, plenum_id_target_folder, id
    )

    print(f"Plenum video resource url: {video_resource_url} (Not downloading yet..)")

    html_ts_map_js_array = get_html_ts_map(
        knesset_http_headers, plenum_id_target_folder, id
    )

    html_transcript = get_html_transcript(
        knesset_http_headers, plenum_id_target_folder, id
    )

    print("Parsing HTML transcript...")
    transcript_text, transcript_time_index_df = parse_plenum_transcript(
        html_transcript, html_ts_map_js_array
    )

    # Store the output artifacts
    with open(plenum_id_target_folder / output_transcript_text_file_name, "w") as f:
        f.write(transcript_text)
    transcript_time_index_df.to_csv(
        plenum_id_target_folder / output_transcript_time_index_file_name
    )

    print("Creating a VTT file of the transcript...")
    vtt = create_plenum_transcript_vtt(transcript_text, transcript_time_index_df)
    vtt.save(plenum_target_folder / f"{id}.vtt", add_bom=True)

    if not skip_download_video_file:
        download_video_recording(plenum_id_target_folder, video_resource_url)
    else:
        print("Skipping video file download.")

    transcode_video_to_audio(id, plenum_target_folder, plenum_id_target_folder)

    print(f"Download plenum {id} done.")


def download(args):
    """Downloads a Knesset recording."""
    target_dir = pathlib.Path(args.target_dir)

    if args.type == "plenum":
        knesset_http_headers = load_http_headers(args)
        for id in args.ids:
            download_plenum(
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
        choices=["plenum"],
        help="Type of recording.",
    )
    parser.add_argument(
        "--ids",
        action="append",
        type=str,
        required=False,  # Manually checked later
        help="Ids of the recording to download, for the type requested.",
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
