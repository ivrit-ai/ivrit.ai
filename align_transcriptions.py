import argparse
import json
import pathlib
import re
from time import strptime

from dtw import dtw
import numpy as np
from scipy.special import softmax
from scipy.spatial.distance import cosine
import scipy.stats as stats
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
        time_index_coords.append((start_seconds, text_length_before_text_part, total_text_length))

    all_text = "".join(all_text_parts)
    time_index = np.asarray(time_index_coords)
    sorted_time_index = time_index[time_index[:, 0].argsort()]
    sorted_time_index = sorted_time_index.T  # 3 x N, row 0 = ts, row 1 = start_loc, row 2 = end_loc

    return all_text, sorted_time_index


def find_split_points(text):
    # Define a regex pattern for split points
    pattern = r"[\s.,;!?()\[\]{}]"

    # Find all occurrences of the pattern
    split_points = [match.start() for match in re.finditer(pattern, text)]

    if len(split_points) == 0 or split_points[0] > 0:
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
    cleaned_up_text_for_dict_gen = re.sub(r"[\s,.?]+", " ", reference_text)  # Consdier sampling if text here is huge
    dict_vocab = sorted(list(set(list(cleaned_up_text_for_dict_gen))))
    # create a dict which maps from a vocab character to an index in this character-histogram vector
    dictionary = {c: i for i, c in enumerate(dict_vocab)}
    # calc the weights for the dict - it is proportional to how "rare" a character is in the text
    # but make sure it is logistically approaching 0..1 range
    dict_weights = np.zeros(len(dict_vocab))
    for c in dict_vocab:
        dict_weights[dictionary.get(c)] = cleaned_up_text_for_dict_gen.count(c)
    logged_up = np.log(dict_weights)
    max_range = np.max(logged_up)
    min_range = np.min(logged_up)
    scale = max_range - min_range
    scaled = (logged_up - min_range) / scale
    dict_weights = softmax(1 - scaled)

    return dictionary, dict_weights


def find_matching_locations_by_hist(
    all_text: str,
    query_text: str,
    bottom_k: int,
    ranges: list[tuple[int, int]],
    stride_size: int,
    dictionary: dict,
    dict_weights: np.ndarray,
) -> str:
    window_size = len(query_text)
    sample_hist = encode_text_window(query_text, dictionary, dict_weights)

    searched_start_locs = []
    start_locs_similarity_scores = []
    for search_range in ranges:
        start_loc, end_loc = search_range

        # Go over all text splits - and check matches
        window_locs_to_compare = np.arange(start_loc, end_loc - window_size, stride_size)

        for window_start_loc in window_locs_to_compare:
            timestampped_text_to_match_against = all_text[window_start_loc : window_start_loc + window_size]
            searched_start_locs.append(window_start_loc)
            start_locs_similarity_scores.append(
                cosine(
                    sample_hist,
                    encode_text_window(timestampped_text_to_match_against, dictionary, dict_weights),
                )
            )

    bottom_k_min_at = np.asarray(start_locs_similarity_scores).argsort()[:bottom_k]
    return np.asarray(searched_start_locs)[bottom_k_min_at]


def get_estimated_audio_lag(
    reference_text: str,
    reference_time_index: np.ndarray,
    sample_split_text: str,
    sample_split_start_at_time: float,
    dictionary: dict,
    dict_weights: np.ndarray,
) -> float:
    # First pass - Hone in on the potential match ranges
    # With faster search but less accurate
    max_window_size = 150
    query_text = sample_split_text[:max_window_size]
    top_matching_locs = find_matching_locations_by_hist(
        reference_text,
        query_text,
        5,  # TODO - relative to ratio between total search area and query size
        [[0, len(reference_text)]],
        # about 1/3rd stride size is good enough for first pass (But not too small)
        max(max_window_size // 3, 5),
        dictionary,
        dict_weights,
    )

    # Create ranges around each potential to search for more accurate matches
    ranges_to_fine_search = []
    for matched_loc in top_matching_locs:
        new_range_start_loc = max(0, matched_loc - max_window_size * 2)
        new_range_end_loc = min(len(reference_text), matched_loc + max_window_size * 3)

        # If a range overlaps with the new range - merge them
        merged = False
        for idx, existing_range in enumerate(ranges_to_fine_search):
            # If open is within an existing range
            if new_range_start_loc > existing_range[0] and new_range_start_loc <= existing_range[1]:
                # extend the end of this range with the right most one of the two ranges
                ranges_to_fine_search[idx] = [
                    existing_range[0],
                    max(new_range_end_loc, existing_range[1]),
                ]
                merged = True

            # If end is within an existing range
            if new_range_end_loc >= existing_range[0] and new_range_end_loc < existing_range[1]:
                # extend the start of this range with the left most one of the two ranges
                ranges_to_fine_search[idx] = [
                    min(new_range_start_loc, existing_range[0]),
                    existing_range[1],
                ]
                merged = True

            if merged:
                break

        if not merged:
            ranges_to_fine_search.append([new_range_start_loc, new_range_end_loc])

    # Perform a fine search on the ranges
    fine_search_max_window_len = 250
    query_text = sample_split_text[:fine_search_max_window_len]
    top_matching_locs = find_matching_locations_by_hist(
        reference_text,
        query_text,
        # Take the best one for the fine search (may need to improve this and consider disagreement between last K places)
        1,
        ranges_to_fine_search,
        5,  # Search in small strides to get a good match accuracy
        dictionary,
        dict_weights,
    )

    match_window_start_loc = top_matching_locs[0]
    transcript_approx_slice_indexes = (
        np.argwhere(
            np.logical_and(
                reference_time_index[1] <= match_window_start_loc,
                match_window_start_loc <= reference_time_index[2],
            )
        )
        .transpose()
        .flatten()
    )
    if transcript_approx_slice_indexes.size > 0:
        transcript_approx_slice_first_idx = transcript_approx_slice_indexes[0]
        # If we have a sample after this one - we can try and interpolate the timestamp
        # relative to text matchong location within that sample
        if transcript_approx_slice_first_idx < len(reference_time_index[1]) - 1:
            relative_loc_into_slice = (
                match_window_start_loc - reference_time_index[1][transcript_approx_slice_first_idx]
            ) / (
                reference_time_index[2][transcript_approx_slice_first_idx]
                - reference_time_index[1][transcript_approx_slice_first_idx]
            )
            total_slice_time_length = (
                reference_time_index[0][transcript_approx_slice_first_idx + 1]
                - reference_time_index[0][transcript_approx_slice_first_idx]
            )
            relative_time_into_slice = relative_loc_into_slice * total_slice_time_length
            transcript_approx_timestamp = (
                reference_time_index[0][transcript_approx_slice_first_idx] + relative_time_into_slice
            )
        # or fallback to the time stamp of this last slice
        else:
            transcript_approx_timestamp = reference_time_index[0][transcript_approx_slice_first_idx]

        sample_approx_timestamp = sample_split_start_at_time
        estimated_audio_lag = transcript_approx_timestamp - sample_approx_timestamp
        return estimated_audio_lag
    else:
        return None


def get_confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)
    h = std_err * stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return mean - h, mean + h


def is_confident_in_guess(samples, tolerance, confidence=0.95):
    lower, upper = get_confidence_interval(samples, confidence)
    return (upper - lower) <= tolerance


initial_audio_lag__estimation_buffer_size_secs = 120
guess_confidence_stop_criteria_confidence_interval = 0.95
guess_confidence_stop_criteria_confidence_seconds = 8
minimal_estimations_to_gather_for_guess = 5
max_estimations_to_gather = 35  # Don't guess forever
minimal_split_text_length_to_use = 25  # Balance accuracy and speed


def guess_audio_lag_behind_reference_transcript(
    output_folder: pathlib.Path,
    reference_text: str,
    reference_time_index: np.ndarray,
    splits: list[dict],
    transcripts_store: dict,
    char_dict: dict[str, int],
    char_dict_weights: np.ndarray,
) -> float:
    # Try and load from a JSON cache an alreafy guessed audio lag
    align_cache_filename = output_folder / "align_cache.json"
    align_cache = None
    if align_cache_filename.exists():
        with open(align_cache_filename, "r") as f:
            align_cache = json.load(f)
            if align_cache.get("audio_lag", None) is not None:
                print(f"Using guessed audio lag of {align_cache['audio_lag']} seconds")
                return align_cache["audio_lag"]

    print("Guessing audio lag behind reference transcript")
    available_samples_count = len(transcripts_store["transcripts"])
    total_audio_recording_length_seconds = splits[-1][1]
    total_transcription_length_seconds = np.max(reference_time_index[0])  # max ts
    # If we assume audio and transcript end at the same time (seems to be about right)
    # we could infer the expected mininal audio lag by looking at the difference.
    # This is not accurate - but should allow a better way to filter out outliers due
    # to bad sample matching cases
    estimated_minimal_audio_lag = total_transcription_length_seconds - total_audio_recording_length_seconds
    lower_audio_lag_limit = estimated_minimal_audio_lag - initial_audio_lag__estimation_buffer_size_secs
    upper_audio_lag_limit = estimated_minimal_audio_lag + initial_audio_lag__estimation_buffer_size_secs
    print(f"Considering lag Estimations in the range: {lower_audio_lag_limit:.2f} - {upper_audio_lag_limit:.2f} secs.")

    estimated_lags = []
    estimaged_lags_within_probable_range = None
    print("Sampling until audio lag guess is acquired...")
    while True:
        split_id_to_sample = np.random.randint(0, available_samples_count, 1).squeeze()
        split_transcript = None
        for t_s in transcripts_store["transcripts"]:
            # List is a shitty data structure for random access by id.
            # reconsider using lists or at least create a lookup
            if t_s["split_idx"] == split_id_to_sample:
                split_transcript = t_s
                break

        if split_transcript is None:
            # cannot find transcription for this split - move on
            continue

        ts_sample_split_segs = split_transcript["segments"]
        sample_split_text = re.sub(r"[\s,.?]+", " ", "".join([seg["text"] for seg in ts_sample_split_segs]))
        if len(sample_split_text) < minimal_split_text_length_to_use:
            # Too short (text len wise) splits are skipped - they are too expensive and may not be specific enough
            continue

        sample_split_start_at_time = splits[split_id_to_sample][0]
        estimated_lag = get_estimated_audio_lag(
            reference_text,
            reference_time_index,
            sample_split_text,
            sample_split_start_at_time,
            char_dict,
            char_dict_weights,
        )
        if estimated_lag is None:
            # Unable to find a match for this split - try to move on to other splits
            pass
        else:
            estimated_lags.append(estimated_lag)

        estimaged_lags_within_probable_range = np.asarray(
            [lag for lag in estimated_lags if lower_audio_lag_limit <= lag <= upper_audio_lag_limit]
        )

        if len(
            estimaged_lags_within_probable_range
        ) >= minimal_estimations_to_gather_for_guess and is_confident_in_guess(
            estimaged_lags_within_probable_range,
            guess_confidence_stop_criteria_confidence_seconds,
            guess_confidence_stop_criteria_confidence_interval,
        ):
            print("Sampled enough to guess the time lag. Stopping.")
            break

        if len(estimated_lags) >= max_estimations_to_gather:
            print("No more sampling allowed. Stopping.")
            break

    guessed_audio_lag = (
        np.mean(estimaged_lags_within_probable_range)
        if estimaged_lags_within_probable_range.size > 0
        else estimated_minimal_audio_lag
    )

    # Store the gussed audio lag in the align cache file
    if align_cache is None:
        align_cache = {}
    align_cache = {"audio_lag": guessed_audio_lag}
    with open(align_cache_filename, "w") as f:
        json.dump(align_cache, f)

    return guessed_audio_lag

def find_closest_search_sorted(array: np.ndarray, value: float, side: str = "left") -> int:
    """
    Find the index of the closest value in a sorted ascending array using binary search.

    This function leverages the fact that the array is sorted to achieve a more efficient
    search with logarithmic time complexity, O(log n). It uses numpy's searchsorted method
    to find the position where the value should be inserted to maintain order and then
    determines the closest value by comparing the nearest neighbors.

    Parameters:
    array (np.ndarray): A sorted (ascending) numpy array to search in.
    value (float): The value to find the closest match for in the array.
    side  (str): Which side to return if the value is not in the array. Default is "left".

    Returns:
    int: The index of the closest value in the array.

    Raises:
    ValueError: If the input array is empty.
    """
    if array.size == 0:
        raise ValueError("Array is empty.")

    # Find the index where 'value' should be inserted to maintain order
    idx = np.searchsorted(array, value, side=side)

    # Handle edge cases where 'value' is out of the array bounds
    if idx == 0:
        return 0
    elif idx == len(array):
        return len(array) - 1
    else:
        # Check the closest value between idx-1 and idx
        if abs(array[idx - 1] - value) <= abs(array[idx] - value):
            return idx - 1
        else:
            return idx


def get_transcript_for_time_range(
    text: str,
    ts_index,
    text_split_points,
    start_time,
    end_time,
    reference_text_neighborhood_size,
):
    start_time_with_buffer = start_time - reference_text_neighborhood_size
    end_time_with_buffer = end_time + reference_text_neighborhood_size

    start_range_idx = find_closest_search_sorted(ts_index[0], start_time_with_buffer, side="left")
    end_range_idx = find_closest_search_sorted(ts_index[0], end_time_with_buffer, side="right")

    # Cannot get an inclusive time range from the index
    if end_range_idx - start_range_idx == 0:
        # If start is within the index, then we can get the closest one
        if start_time_with_buffer >= ts_index[0, 0] and start_time_with_buffer <= ts_index[0, -1]:
            # Get the closest one
            start_range_idx = np.abs(ts_index[0] - start_time_with_buffer).argmin()
            end_range_idx = start_range_idx + 1

    if end_range_idx - start_range_idx == 0:
        raise ValueError("Transcript lookup out of range.")

    segment_split_points = (
        np.min(ts_index[1, start_range_idx : end_range_idx + 1]),
        np.max(ts_index[1, start_range_idx : end_range_idx + 1]),
    )

    split_below, split_above = get_approximate_text_split_points(text_split_points, *segment_split_points)

    return text[split_below:split_above]


def encode_text_window(text_window: str, char_dict: dict[str, int], dict_weights: np.ndarray) -> np.ndarray:
    # calculate the "histogram" over the dictionary as the X axis
    window_hist = np.zeros(len(char_dict))
    window_sum = 0
    for j in range(0, len(text_window)):
        idx_of_vocab_char = char_dict.get(text_window[j])
        # Only count characters in dict
        if idx_of_vocab_char is not None:
            window_hist[idx_of_vocab_char] += 1

        # normalize the histogram
        window_sum = np.sum(window_hist)

    # If the histogram captured nothing - this window is not over
    # representable text from the dictionary
    if window_sum == 0:
        # Don't consider this a valid window
        return None

    window_hist = window_hist / window_sum
    window_hist_weighted = window_hist * dict_weights
    return window_hist_weighted


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

        window_hist = encode_text_window(window, char_dict, dict_weights)
        if window_hist is None:
            continue

        # print the histogram
        window_hists.append(window_hist)

    # append all hists to a np matrix and return
    return np.array(window_hists)


def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def find_matching_edge(
    anchor_split_point: int,
    edge_hist: np.ndarray,
    ref_text: str,
    ref_split_points: np.ndarray,
    char_dict: dict[str, int],
    dict_weights: np.ndarray,
):
    max_dist_to_consider = 0.35
    best_edge_dist = np.inf
    best_edge_idx = anchor_split_point
    for split_idx_on_ref in range(anchor_split_point - 2, anchor_split_point + 2):
        if split_idx_on_ref < 0 or split_idx_on_ref + 1 >= len(ref_split_points):
            continue

        checked_edge_slice = slice(
            ref_split_points[split_idx_on_ref],
            ref_split_points[split_idx_on_ref + 1],
        )
        checked_edge = ref_text[checked_edge_slice]
        checked_edge_hist = encode_text_window(checked_edge, char_dict, dict_weights)
        if checked_edge_hist is None:
            continue

        edge_match_dist = cosine(edge_hist, checked_edge_hist)
        if edge_match_dist < best_edge_dist and edge_match_dist < max_dist_to_consider:
            best_edge_dist = edge_match_dist
            best_edge_idx = split_idx_on_ref

    return best_edge_idx


def find_exact_ref_edges(
    ref_text: str,
    query_text: str,
    index_mapping_of_ref_to_qry: np.ndarray,
    char_dict: dict[str, int],
    dict_weights: np.ndarray,
):
    # if the splitted edge it too short
    # It could be a punctuation or a word boundary
    # take another one until enough signal is cpatured for that edge
    minimal_edge_split_text_len = 4

    ref_snippet_split_points = np.array(find_split_points(ref_text))
    query_snippet_split_points = np.array(find_split_points(query_text))

    ## Find Lower Bound in ref

    # Find where the split of the query edge should be taken from
    valid_split_points_idx_for_lower_edge = (query_snippet_split_points - minimal_edge_split_text_len >= 0).nonzero()[0]
    if valid_split_points_idx_for_lower_edge.size > 0:
        lower_edge_split_idx = valid_split_points_idx_for_lower_edge[0]
        query_lower_edge_take_to = query_snippet_split_points[lower_edge_split_idx]
    else:
        query_lower_edge_take_to = len(query_text)

    # Take the query edge - and use that to find the best edge in the reference
    query_lower_edge = query_text[:query_lower_edge_take_to]
    lower_edge_hist = encode_text_window(query_lower_edge, char_dict, dict_weights)
    if lower_edge_hist is not None:
        lower_edge_anchor_aplit_idx_on_ref = find_nearest_idx(ref_snippet_split_points, index_mapping_of_ref_to_qry[0])
        lower_ref_split_point_idx = find_matching_edge(
            lower_edge_anchor_aplit_idx_on_ref,
            lower_edge_hist,
            ref_text,
            ref_snippet_split_points,
            char_dict,
            dict_weights,
        )
    # Fallback when the query text cannot produce a valid histogram
    else:
        lower_ref_split_point_idx = 0

    ## Find Upper Bound in ref

    # Find where the split of the query edge should be taken from
    valid_split_points_idx_for_upper_edge = (
        len(query_text) - minimal_edge_split_text_len - query_snippet_split_points >= 0
    ).nonzero()[0]
    if valid_split_points_idx_for_upper_edge.size > 0:
        upper_edge_split_idx = valid_split_points_idx_for_upper_edge[-1]
        query_upper_edge_take_from = query_snippet_split_points[upper_edge_split_idx]
    else:
        query_upper_edge_take_from = 0

    # Take the query edge - and use that to find the est edge in the reference
    query_upper_edge = query_text[query_upper_edge_take_from:]
    upper_edge_hist = encode_text_window(query_upper_edge, char_dict, dict_weights)
    if upper_edge_hist is not None:
        upper_edge_anchor_aplit_idx_on_ref = find_nearest_idx(ref_snippet_split_points, index_mapping_of_ref_to_qry[-1])
        upper_ref_split_point_idx = find_matching_edge(
            upper_edge_anchor_aplit_idx_on_ref,
            upper_edge_hist,
            ref_text,
            ref_snippet_split_points,
            char_dict,
            dict_weights,
        )
    # Fallback when the query text cannot produce a valid histogram
    else:
        upper_ref_split_point_idx = len(ref_snippet_split_points) - 1

    return (
        ref_snippet_split_points[lower_ref_split_point_idx],
        # Include the last splitted text
        ref_snippet_split_points[min(upper_ref_split_point_idx + 1, len(ref_snippet_split_points) - 1)],
    )


def align_split(
    args: dict,
    reference_text: str,
    reference_time_index: np.ndarray,
    text_natural_split_points: np.ndarray,
    char_dict: dict[str, int],
    char_dict_weights: np.ndarray,
    guessed_audio_lag: float,
    idx: int,
    split: dict,
    trans_store: dict,
) -> dict:
    if args.transcripts_split_id is not None:
        if idx != args.transcripts_split_id:
            return

    # Audio times
    time_start_s, time_end_s = split
    # Shift the audio times according to the guessed lag
    # Lag is of audio behind transcript
    # Positive - means audio is behind => TS are lower than they
    # should be to match th reference text
    # Thus - add the lag value to the audio times
    time_start_s += guessed_audio_lag
    time_end_s += guessed_audio_lag
    split_transcript = None

    # TODO: The list structure for lookup by a sub prop of a json element
    # Is inefficient - Consider storing the transcription store as an id: val lookup dict
    # The reason we need to lookup by the split idx is that some splits would not have a transcription
    for t_s in trans_store["transcripts"]:
        if t_s["split_idx"] == idx:
            split_transcript = t_s
            break

    # A missing transcript is possible, if for example
    # The transcription server could not find any sppeach parts
    # In the audio split. Skip this transcription - this sample
    # Will be dropped from the output dataset
    if split_transcript is None:
        return None

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
        query_embedding = encode_text_stream(split_qry_text, args.text_hist_window_size, char_dict, char_dict_weights)
        ref_embedding = encode_text_stream(
            ref_text_around_split,
            args.text_hist_window_size,
            char_dict,
            char_dict_weights,
        )

        # Match query onto reference
        alignment = dtw(
            query_embedding,
            ref_embedding,
            keep_internals=False,
            dist_method="cosine",
            open_begin=True,
            open_end=True,
            step_pattern="asymmetric",
        )
        index_mapping_of_ref_to_qry = alignment.index2

        # Find the natural locations to cut out parts from the reference text
        start_to_pick_from_ref, end_to_pick_from_ref = find_exact_ref_edges(
            ref_text_around_split,
            split_qry_text,
            index_mapping_of_ref_to_qry,
            char_dict,
            char_dict_weights,
        )

        final_snippet = ref_text_around_split[start_to_pick_from_ref:end_to_pick_from_ref].strip()

        if not final_snippet:
            is_fallback = True
        else:
            final_alignment = dtw(
                query_embedding,
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
        "split_idx": split_transcript["split_idx"],
        "base": split_qry_text,
        "reference": ref_text_around_split,
        "aligned": final_snippet,
        "alignment_distance": 0 if is_fallback else final_alignment.normalizedDistance,
        "is_fallback": is_fallback,
    }


def align(args):
    output_transcripts_folder = pathlib.Path(args.transcripts_file).parent
    output_transcripts_file = output_transcripts_folder / f"{pathlib.Path(args.transcripts_file).stem}.aligned.json"

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
    char_dict, char_dict_weights = generate_transcription_character_dictionary(reference_text)

    # Go over all splits
    aligned_transcripts = []

    guessed_audio_lag = guess_audio_lag_behind_reference_transcript(
        output_transcripts_folder,
        reference_text,
        reference_time_index,
        splits_desc["splits"],
        trans_store,
        char_dict,
        char_dict_weights,
    )
    print(f"Guessed audio lag: {guessed_audio_lag}sec (how much audio is behind reference text)")

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
            guessed_audio_lag,
            idx,
            split,
            trans_store,
        )
        for idx, split in tqdm(enumerate(tqdm(splits_desc["splits"])))
    ]
    # Some transcripts may not work out - drop them from final output
    aligned_transcripts = [at for at in aligned_transcripts if at is not None]

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
    parser = argparse.ArgumentParser(description="""Align external transcription with split transcriptions""")

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
        default=25,
        help="Size in seconds before and after split time range to gather reference text to match against",
    )

    # Parse the arguments
    args = parser.parse_args()

    align(args)
