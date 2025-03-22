from collections import deque
from types import MethodType

import numpy as np
import stable_whisper

def get_confusion_zone(
    aligned: stable_whisper.WhisperResult,
    detection_window_duration: int = 120,
    hop_length: int = 30,
    good_seg_prob_threshold: float = 0.4,
    bad_to_good_probs_detection_threshold: float = 0.8,
):
    """Detect zones in the alignment where the model is confused.

    This function uses a sliding window approach to find regions where the ratio
    of low-confidence words to high-confidence words exceeds a threshold.

    Args:
        aligned: The aligned WhisperResult to analyze.
        detection_window_duration: The duration of the sliding window in seconds.
        hop_length: The step size for the sliding window in seconds.
        good_seg_prob_threshold: Probability threshold above which a word is considered good.
        bad_to_good_probs_detection_threshold: Threshold for the ratio of bad to good durations.
            If this threshold is exceeded, a confusion zone is detected.

    Returns:
        A tuple of (start_time, end_time) for the confusion zone, or (None, None) if no
        confusion zone is detected.
    """
    # Pre-fetch all segments
    all_words = aligned.all_words()
    start_time = all_words[0].start
    end_time = all_words[-1].end

    # Sliding window setup
    window_start = start_time
    window_end = window_start + detection_window_duration
    window_words = deque()
    word_index = 0  # Index for iterating through all_words

    while window_end < end_time:
        # Add new words entering the window
        while word_index < len(all_words) and all_words[word_index].start < window_end:
            word = all_words[word_index]
            if word.end > window_start:  # At least partially overlaps with window
                window_words.append(word)
            word_index += 1

        # Remove words exiting the window
        while window_words and window_words[0].end <= window_start:
            window_words.popleft()

        # Compute durations (treat 0 words with some durations to capture their bad probs)
        bad_durations = [max(0.1, w.end - w.start) for w in window_words if w.probability < good_seg_prob_threshold]
        good_durations = [w.end - w.start for w in window_words if w.probability >= good_seg_prob_threshold]

        total_bad = sum(bad_durations)
        total_good = sum(good_durations)

        if total_good == 0 or (total_bad / total_good) > bad_to_good_probs_detection_threshold:
            # Confusion window detected - return bounds
            return window_start, window_end

        # Slide window
        window_start += hop_length
        window_end = window_start + detection_window_duration

    return None, None


def create_transcript_from_segments(
    slice_segments: list[stable_whisper.result.Segment],
):
    """Create a WhisperResult from a list of segments.

    Args:
        slice_segments: A list of Segment objects to include in the result.

    Returns:
        A WhisperResult containing the provided segments.
    """
    slice_segments_as_dict = [s.to_dict() for s in slice_segments]
    return stable_whisper.WhisperResult({"segments": slice_segments_as_dict})


def get_text_from_segments(segments: list[stable_whisper.result.Segment]):
    """Extract and concatenate text from a list of segments.

    Args:
        segments: A list of Segment objects to extract text from.

    Returns:
        A string containing the concatenated text from all segments.
    """
    return "".join([s.text for s in segments])


def find_probable_segment_before_time(
    aligned: stable_whisper.WhisperResult,
    find_before_time: float,
    go_back_duration: float,
    minimal_seg_prob_to_consider_retry_start_segment: float = 0.8,
    max_seek_start_seg_backward_hops: int = 6,
):
    """Find a high-confidence segment before a given time.

    This function searches for a segment with high confidence before a specified time,
    which can be used as a reliable starting point for re-alignment.

    Args:
        aligned: The aligned WhisperResult to search in.
        find_before_time: The time before which to search for a segment.
        go_back_duration: The duration to look back from find_before_time.
        minimal_seg_prob_to_consider_retry_start_segment: Minimum average probability
            for a segment to be considered reliable.
        max_seek_start_seg_backward_hops: Maximum number of backward hops to make
            if no suitable segment is found in the initial window.

    Returns:
        A Segment object if a suitable segment is found, or None otherwise.
    """
    # need to circle back and try again.
    # in some cases, starting in some slightly different locations is enough
    # to get over the confusion zone
    segment_to_start_after = None
    search_for_segment_start_time = find_before_time - go_back_duration
    while not segment_to_start_after and max_seek_start_seg_backward_hops > 0:
        pre_confusion_zone_segments = aligned.get_content_by_time(
            (
                search_for_segment_start_time,
                search_for_segment_start_time + go_back_duration,
            ),
            segment_level=True,
        )
        if pre_confusion_zone_segments:
            for pre_seg in pre_confusion_zone_segments:
                seg_word_probs = [w.probability for w in pre_seg.words]
                avg_seg_probabilty = np.mean(seg_word_probs) if seg_word_probs else 0
                if avg_seg_probabilty >= minimal_seg_prob_to_consider_retry_start_segment:
                    segment_to_start_after = pre_seg
                    break

        if not segment_to_start_after:
            max_seek_start_seg_backward_hops -= 1
            search_for_segment_start_time -= go_back_duration
            if search_for_segment_start_time < 0:
                break

    return segment_to_start_after


def get_breakable_align_model(model: str, device: str, compute_type: str):
    """Get a model with the breakable_align method attached.

    Args:
        model: The model to load, either a path to a local model or a
            model identifier from the Hugging Face Hub.
        device: The device to use for inference, e.g., "cpu", "cuda", "auto".
            Can include a device index, e.g., "cuda:0".
        compute_type: The compute type to use for the model, e.g., "int8", "float16".

    Returns:
        A model with the breakable_align method attached.
    """
    from alignment.breakable_aligner import breakable_align

    device_index = None
    if len(device.split(":")) == 2:
        device, device_index = device.split(":")
        device_index = int(device_index)

    model = stable_whisper.load_faster_whisper(
        model,
        device=device,
        device_index=device_index,
        compute_type=compute_type,
    )
    model.align = MethodType(breakable_align, model)

    return model
