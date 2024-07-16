import argparse
import json
import pathlib
import re
from time import strptime

from dtw import dtw
import numpy as np
from scipy.special import softmax
from tqdm import tqdm
from webvtt import WebVTT


def extract_text_and_time_index_from_vtt(vtt):
    all_text_parts = []
    time_index_coords = []  # touple of (ts, text start, text end)
    total_text_length = 0
    for caption in vtt:
        text_length_before_text_part = total_text_length
        all_text_parts.append(caption.text)
        total_text_length += len(caption.text)
        time_obj = strptime(caption.start, "%H:%M:%S.%f")
        start_seconds = time_obj.tm_hour * 3600 + time_obj.tm_min * 60 + time_obj.tm_sec
        time_index_coords.append(
            (start_seconds, text_length_before_text_part, total_text_length)
        )

    all_text = "".join(all_text_parts)
    time_index = np.asarray(time_index_coords)
    sorted_time_index = time_index[time_index[:, 0].argsort()]
    sorted_time_index = (
        sorted_time_index.T
    )  # 3 x N, row 0 = ts, row 1 = start_loc, row 2 = end_loc

    return all_text, sorted_time_index


def find_split_points(text):
    # Define a regex pattern for split points
    pattern = r"[\s.,;!?()\[\]{}]"

    # Find all occurrences of the pattern
    split_points = [match.start() for match in re.finditer(pattern, text)]

    if split_points[0] > 0:
        split_points = [0, *split_points]
    return np.asarray(split_points)


def get_approximate_text_split_points(text_split_points, low_target, high_target):
    # low target - find the closest natural split point below
    split_point_from_below = text_split_points[text_split_points < low_target].max()
    split_point_from_above = text_split_points[text_split_points > high_target].min()
    return (split_point_from_below, split_point_from_above)


def generate_transcription_character_dictionary(
    reference_text,
) -> tuple[dict[str, int], np.ndarray]:
    cleaned_up_text_for_dict_gen = re.sub(
        r"\s+", " ", reference_text
    )  # Consdier sampling if text here is huge
    dict_vocab = sorted(list(set(list(cleaned_up_text_for_dict_gen))))
    # create a dict which maps from a vocab character to an index in this character-histogram vector
    dict = {c: i for i, c in enumerate(dict_vocab)}
    # calc the weights for the dict - it is proportional to how "rare" a character is in the text
    # but make sure it is logistically approaching 0..1 range
    dict_weights = np.zeros(len(dict_vocab))
    for c in dict_vocab:
        dict_weights[dict.get(c)] = cleaned_up_text_for_dict_gen.count(c)
    dict_weights = softmax(1 / dict_weights)

    return dict, dict_weights


def get_transcript_for_time_range(
    text,
    ts_index,
    text_split_points,
    start_time,
    end_time,
    reference_text_neighborhood_size,
):
    start_time_with_buffer = start_time - reference_text_neighborhood_size
    end_time_with_buffer = end_time + reference_text_neighborhood_size

    start_range_idx = np.searchsorted(ts_index[0], start_time_with_buffer, side="left")
    end_range_idx = np.searchsorted(ts_index[0], end_time_with_buffer, side="right")

    # Cannot get an inclusive time range from the index
    if end_range_idx - start_range_idx == 0:
        # If start is within the index, then we can get the closest one
        if (
            start_time_with_buffer >= ts_index[0, 0]
            and start_time_with_buffer <= ts_index[0, -1]
        ):
            # Get the closest one
            start_range_idx = np.abs(ts_index[0] - start_time_with_buffer).argmin()
            end_range_idx = start_range_idx + 1

    if end_range_idx - start_range_idx == 0:
        raise ValueError("Transcript lookup out of range.")

    segment_split_points = (
        np.min(ts_index[1, start_range_idx:end_range_idx]),
        np.max(ts_index[1, start_range_idx:end_range_idx]),
    )

    split_below, split_above = get_approximate_text_split_points(
        text_split_points, *segment_split_points
    )

    return text[split_below:split_above]


def encode_text_stream(
    text_stream: str,
    window_size: int,
    char_dict: dict[str, int],
    dict_weights: np.ndarray,
) -> np.ndarray:
    half_window_size = window_size // 2
    window_hists = []
    # go over the input stream with windows of size window_size
    for i in range(0, len(text_stream)):

        curr_window_slice = slice(
            max(0, i - half_window_size),
            min(len(text_stream), i + window_size - half_window_size),
        )
        window = text_stream[curr_window_slice]

        # calculate the "histogram" over the dictionary as the X axis
        window_hist = np.zeros(len(char_dict))
        for j in range(0, len(window)):
            idx_of_vocab_char = char_dict.get(window[j])
            # Only count characters in dict
            if idx_of_vocab_char is not None:
                window_hist[idx_of_vocab_char] += 1

        # normalize the histogram
        window_sum = np.sum(window_hist)

        # If the histogram captured nothing - this window is not over
        # representable text from the dictionary
        if window_sum == 0:
            # Don't apprent it it the output vector
            continue

        window_hist = window_hist / window_sum
        window_hist_weighted = window_hist * dict_weights
        # print the histogram
        window_hists.append(window_hist_weighted)

    # append all hists to a np matrix and return
    return np.array(window_hists)


def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


# TODO: bound matching is not working well for all cases
# need to move to a more robust matching at edges to find correct split points
def find_nearest_above(array, value):
    idx = find_nearest_idx(array, value)
    idx += 1
    return array[min(idx, array.size - 1)]


def find_nearest_below(array, value):
    idx = find_nearest_idx(array, value)
    idx -= 1
    return array[max(0, idx)]


def align_split(
    args: dict,
    reference_text: str,
    reference_time_index: np.ndarray,
    text_natural_split_points: np.ndarray,
    char_dict: dict[str, int],
    char_dict_weights: np.ndarray,
    idx: int,
    split: dict,
    trans_store: dict,
) -> dict:
    if args.transcripts_split_id is not None:
        if idx != args.transcripts_split_id:
            return

    time_start_s, time_end_s = split
    split_transcript = trans_store["transcripts"][idx]
    split_seg_texts = []
    max_logprob = -np.inf

    for seg in split_transcript["segments"]:
        split_seg_texts.append(seg["text"].strip())
        max_logprob = max(max_logprob, seg["avg_logprob"])
    split_qry_text = " ".join(split_seg_texts).strip()

    is_fallback = False
    try:
        ref_text_around_split = re.sub(
            r"\s+",
            " ",
            get_transcript_for_time_range(
                reference_text,
                reference_time_index,
                text_natural_split_points,
                time_start_s,
                time_end_s,
                reference_text_neighborhood_size=args.reference_text_neighborhood_size,
            ),
        )
    except:
        # In some cases - ref does not contain the ranges this split
        # points at - fallback to the query as a last resort
        ref_text_around_split = ""
        is_fallback = True

    if not is_fallback:
        qury_embedding = encode_text_stream(
            split_qry_text, args.text_hist_window_size, char_dict, char_dict_weights
        )
        ref_embedding = encode_text_stream(
            ref_text_around_split,
            args.text_hist_window_size,
            char_dict,
            char_dict_weights,
        )

        # Match query onto reference
        alignment = dtw(
            qury_embedding,
            ref_embedding,
            keep_internals=False,
            dist_method="cosine",
            open_begin=True,
            open_end=True,
            step_pattern="asymmetric",
        )
        index_mapping_of_ref_to_qry = alignment.index2

        # Find the natural locations to cut out parts from the reference text
        ref_snippet_split_points = np.array(find_split_points(ref_text_around_split))

        start_to_pick_from_ref = find_nearest_below(
            ref_snippet_split_points, index_mapping_of_ref_to_qry[0]
        )
        end_to_pick_from_ref = find_nearest_above(
            ref_snippet_split_points, index_mapping_of_ref_to_qry[-1]
        )

        final_snippet = ref_text_around_split[
            start_to_pick_from_ref:end_to_pick_from_ref
        ].strip()

        if not final_snippet:
            is_fallback = True
        else:
            final_alignment = dtw(
                qury_embedding,
                ref_embedding,
                keep_internals=True,
                dist_method="correlation",
                open_begin=True,
                open_end=True,
                step_pattern="asymmetric",
            )

    if is_fallback:
        final_snippet = split_qry_text
        is_fallback = True

    return {
        "base": split_qry_text,
        "reference": ref_text_around_split,
        "aligned": final_snippet,
        "alignment_distance": 0 if is_fallback else final_alignment.normalizedDistance,
        "is_fallback": is_fallback,
    }


def align(args):
    output_transcripts_file = (
        pathlib.Path(args.transcripts_file).parent
        / f"{pathlib.Path(args.transcripts_file).stem}.aligned.json"
    )

    # If output transcript file exists - no need to reprocess
    if output_transcripts_file.exists():
        if not args.force_reprocess:
            print("Alignment already done. Skipping...")
        else:
            print("Alignment already done. Forced reprocessing...")

    # Read reference vtt file
    print("Loading VTT File.")
    vtt = WebVTT.read(args.reference_file)

    # Read splits file
    print("Loading Splits File.")
    with open(args.splits_file, "r") as f:
        splits_desc = json.load(f)

    # Read transcripts file
    print("Loading transcripts File.")
    with open(args.transcripts_file, "r") as f:
        trans_store = json.load(f)

    # Extract text and time index from vtt
    print("Parsing VTT reference transcription data.")
    reference_text, reference_time_index = extract_text_and_time_index_from_vtt(vtt)

    text_natural_split_points = find_split_points(reference_text)

    # Character dictionary for corpus
    char_dict, char_dict_weights = generate_transcription_character_dictionary(
        reference_text
    )

    # Go over all splits
    aligned_transcripts = []

    if args.transcripts_split_id is not None:
        print(f"Processing specifically split {args.transcripts_split_id}")
    else:
        print(f"Aligning {len(splits_desc['splits'])} splits.")

    aligned_transcripts = [
        align_split(
            args,
            reference_text,
            reference_time_index,
            text_natural_split_points,
            char_dict,
            char_dict_weights,
            idx,
            split,
            trans_store,
        )
        for idx, split in tqdm(enumerate(tqdm(splits_desc["splits"])))
    ]

    if not args.skip_output_file:
        json.dump(
            {
                "source": trans_store["source"],
                "episode": trans_store["episode"],
                "transcripts": aligned_transcripts,
            },
            open(output_transcripts_file, "w"),
            ensure_ascii=False,
        )


if __name__ == "__main__":
    # Define an argument parser
    parser = argparse.ArgumentParser(
        description="""Align external transcription with split transcriptions"""
    )

    # Add the arguments
    parser.add_argument(
        "--reference-file",
        type=str,
        required=True,
        help="Reference vtt transcription file to align",
    )
    parser.add_argument(
        "--splits-file",
        type=str,
        required=True,
        help="Splits json file describing splits to align against",
    )
    parser.add_argument(
        "--transcripts-file",
        type=str,
        required=True,
        help="Splits Transcriptions json file",
    )
    parser.add_argument(
        "--transcripts-split-id",
        type=int,
        required=False,
        help="Align only a specific split id for debugging purpouses",
    )
    parser.add_argument(
        "--skip-output-file",
        action="store_true",
        required=False,
        help="Skip output file creation",
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Force reprocessing of the transcripts alignment",
    )
    parser.add_argument(
        "--text-hist-window-size",
        type=int,
        required=False,
        default=10,
        help="Window size for text histogram embedding",
    )
    parser.add_argument(
        "--reference-text-neighborhood-size",
        type=int,
        required=False,
        default=10,
        help="Size before and after split time range to gather reference text to match against",
    )

    # Parse the arguments
    args = parser.parse_args()

    align(args)
