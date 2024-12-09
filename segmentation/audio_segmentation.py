import numpy as np
from numpy.random import default_rng, Generator
from scipy.cluster.vq import kmeans2

from utils.audio import WHISPER_EXPECTED_SAMPLE_RATE
from vad.definitions import SPEECH_PROB_FRAME_DURATION

default_tuning_parameters = {
    "max_processed_slice_duration": 200,  # Limit the processing size of slices of the audio
    "max_processed_slice_overlap": 10,  # try and overlap slices, but not more than this duration
    "speech_median_filter_duration_from_median_word_duration_factor": [  # One sizer per sensitivity level
        2,
        0,
        0,
    ],
    "speech_prob_cluster_kernels": [  # One set of kernels per sensitivity level
        [0, 1],
        [0, 0.85, 1],
        [0, 0.95, 1],
    ],
    "word_gap_from_median_word_duration_factor": [  # One duration factor per sensitivity level
        1.7,
        1.6,
        1.2,
    ],
    "word_bound_search_area_radius_from_median_word_duration_factor": 2,
    "word_gap_search_area_radius_from_median_word_duration_factor": 0.7,
    "min_range_duration_to_force_subdivide": 8,
    "max_range_duration_to_force_merge": 1,
    # so for a 0.02, a 0.98 and below, is, when we must, considered as a non speech
    # multiple force division levels
    "abs_significance_to_consider_non_speech": [
        0.02,
        0.01,
        0.001,
        0.001,
    ],
    # don't go below % of the top detected no speech significance
    # multiple force division levels
    "relative_significance_pct_to_keep_searching": [
        0.3,
        0.2,
        0.2,
        0.2,
    ],
    # When a slice point is found - radius to consider "no need to find here more"
    # this is correlated to the length of the minimal duration to segment
    "slice_point_padding_duration": 2,
    # Smoothing using gauss, std of the gauss - multiple force division levels
    "speech_gauss_filter_sigma_from_median_word_duration_factor": [
        0.8,
        0.6,
        0.4,
        0.4,
    ],
    "high_confidence_min_no_speech_range_duration": 3.5,
    "no_speech_range_edge_safety_margin_duration": 0.3,
}
DEFAULT_RNG_SEED = 683444


def create_segmentation_processing_context(
    speech_probs_per_frame: np.ndarray,
    words_list: list[dict] = [],
    seed: Generator = None,
    parameters_override: dict = {},
):
    if seed is None:
        seed = default_rng(DEFAULT_RNG_SEED)

    context = {
        "rng": seed,
        "sample_rate": WHISPER_EXPECTED_SAMPLE_RATE,
        "speech_prob_frame_length_sec": SPEECH_PROB_FRAME_DURATION,
        "duration": speech_probs_per_frame.size * SPEECH_PROB_FRAME_DURATION,
        "speech_probs": speech_probs_per_frame,
        "words_list": words_list,  # Loaded and parsed from StableTs or directly from whisper output with word level ts
    }

    context["params"] = {**default_tuning_parameters, **parameters_override}

    return context


def times_to_speech_prob_frames(context, times):
    """This assumes prob and times share the same 0"""
    # If inputs times is a numpy ndarry - can do this in one go
    if isinstance(times, np.ndarray):
        return np.round(times / context["speech_prob_frame_length_sec"]).astype(int)

    return [int(t / context["speech_prob_frame_length_sec"]) for t in times]


def get_sliced_context(context, start_time, end_time):
    slice_context = {**context}
    slice = {}
    slice_context["slice"] = slice
    slice["start_time"] = start_time
    slice["end_time"] = end_time
    slice["words"] = [w for w in context["words_list"] if w["start"] > start_time and w["end"] < end_time]
    slice_start_frame, slice_end_frame = times_to_speech_prob_frames(context, (start_time, end_time))
    slice["speech_probs_start_frame"] = slice_start_frame
    slice["speech_probs_end_frame"] = slice_end_frame
    slice["speech_probs"] = context["speech_probs"][slice_start_frame:slice_end_frame]
    return slice_context


def prob_frames_to_times(context, frames):
    """This assumes prob and times share the same 0"""
    if isinstance(frames, np.ndarray):
        return frames * context["speech_prob_frame_length_sec"]

    return [f * context["speech_prob_frame_length_sec"] for f in frames]


def tighten_ranges(ranges, amount):
    """Start are push forward, ends are pulled backwards
        Using the provided amount
        This effectively "shrinks" the ranges like so:
        ---[----]--------[-----------]-----
        ----[--]----------[---------]------
        But it if the two edges collapse - the range is removed

    Args:
        context
        ranges (np.ndarray): (2, N) 2D array of start, end pairs, first row is "starts"
        amount (number): In the same units as the range
    """
    tighter_ranges = ranges + np.array([[amount], [-amount]])
    tighter_ranges_collapsed = tighter_ranges[0, :] >= tighter_ranges[1, :]
    tighter_ranges = tighter_ranges[:, ~tighter_ranges_collapsed]
    return tighter_ranges


def find_high_confidence_no_speech_ranges(context, min_no_speech_range_duration):
    probs = context["slice"]["speech_probs"]

    clusters, speech_or_not_classes = kmeans2(
        probs,
        [0, 1],
        seed=context["rng"],
    )

    # Find the base line speech probability.
    # we want it to be "speec" so we can detect the "no speech"
    # compared to it
    speech_probs_seg_quantized = np.round(probs, 1)
    unique_values, counts = np.unique(speech_probs_seg_quantized, return_counts=True)
    most_common_value_index = np.argmax(counts)
    common_baseline = unique_values[most_common_value_index]

    # Sanity - if this is not 1 - we may not know what is speech and what is not.
    if common_baseline != 1:
        return np.empty((2, 0), dtype=np.float32)

    # Find silence ranges
    transitions_to_no_speech_idx = np.nonzero(speech_or_not_classes[1:] < speech_or_not_classes[:-1])[0]
    first_no_speech_idxs = transitions_to_no_speech_idx + 1
    transitions_from_no_speech_idx = np.nonzero(speech_or_not_classes[1:] > speech_or_not_classes[:-1])[0]
    last_no_speech_idxs = transitions_from_no_speech_idx

    starts_with_no_speech = speech_or_not_classes[0] == 0
    ends_with_no_speech = speech_or_not_classes[-1] == 0
    if starts_with_no_speech:
        first_no_speech_idxs = np.insert(first_no_speech_idxs, 0, 0)
    if ends_with_no_speech:
        last_no_speech_idxs = np.append(last_no_speech_idxs, speech_or_not_classes.size)

    no_speech_ranges = np.stack([first_no_speech_idxs, last_no_speech_idxs], axis=1)
    no_speech_ranges_times = prob_frames_to_times(context, no_speech_ranges)

    # from relative to slice start to entire audio start
    no_speech_ranges_times = no_speech_ranges_times + context["slice"]["start_time"]

    # Remove silence ranges which are too short
    no_speech_ranges_durations = no_speech_ranges_times[:, 1] - no_speech_ranges_times[:, 0]
    no_speech_ranges_long_enough = no_speech_ranges_times[no_speech_ranges_durations > min_no_speech_range_duration, :]

    no_speech_range_edge_safety_margin_duration = context["params"]["no_speech_range_edge_safety_margin_duration"]

    # Tighten ranges to try and avoid speech edges - we prefer false negatives - miss some no-speech
    # parts at the edges
    # THIS IS ROW WISE structured like most ranges in this modules
    no_speech_ranges_long_enough_safety_trimmed = tighten_ranges(
        no_speech_ranges_long_enough.T, no_speech_range_edge_safety_margin_duration
    )

    return no_speech_ranges_long_enough_safety_trimmed
