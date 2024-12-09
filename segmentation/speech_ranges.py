import numpy as np

from segmentation.audio_segmentation import (
    find_high_confidence_no_speech_ranges,
    create_segmentation_processing_context,
    get_sliced_context,
)


def _convert_no_speech_ranges_to_speech_ranges(no_speech_ranges: np.ndarray, total_duration: float) -> np.ndarray:
    """Convert no speech ranges to speech ranges.
    As we assume that all the remaining areas are speech, the speech ranges are the complement of the no speech ranges.
    Each speech range is defined by start and end of the speech.

    Args:
        no_speech_ranges (np.ndarray): array of no speech ranges, shape (n, 2) where n is the number of no speech ranges
    Returns:
        np.ndarray: array of speech ranges, shape (m, 2) where m is the number of speech ranges
    """

    # if there are no no-speech ranges, the entire duration is speech
    if no_speech_ranges.shape[0] == 0:
        return np.array([[0, total_duration]])

    speech_ranges = []
    prev_no_speech_end = None
    for nospeech_start, nospeech_end in no_speech_ranges:  # row iterator
        if prev_no_speech_end is None:
            if nospeech_start > 0:
                speech_ranges.append([0, nospeech_start])
            prev_no_speech_end = nospeech_end
        else:
            speech_ranges.append([prev_no_speech_end, nospeech_start])
            prev_no_speech_end = nospeech_end

    if prev_no_speech_end < total_duration:
        speech_ranges.append([prev_no_speech_end, total_duration])

    return np.array(speech_ranges)


def get_audio_speech_ranges(frame_level_speech_probs: np.ndarray, min_no_speech_range_duration: float) -> np.ndarray:
    """Considering per-frame speech probabilities (from a vad model) would define speech ranges.
    A speech range would be any range which is not a distinct "no speech" range.
    A distinct no-speech range needs to be at least "min duration" length, and have consecutive low speech
    probabilities compared to it's surrounding audio area.
    Processing is done in slices of the audio

    Args:
        frame_level_speech_probs (np.ndarray): per-frame speech probs from a vad model
        min_no_speech_range_duration (float): duration in seconds of a minimal span of no-speech range to detect

    Returns:
        np.ndarray: (N, 2) shape, each row is a (start, end) time in seconds of a speech range
    """

    # Prepare a dummy context to be able to reuse the logic from the segmentation module
    context = create_segmentation_processing_context(
        speech_probs_per_frame=frame_level_speech_probs,
        parameters_override={"max_processed_slice_duration": 20, "max_processed_slice_overlap": 5},
    )

    latest_processed_to = 0
    total_duration = context["duration"]  # Total audio duration
    max_slice_duration = context["params"]["max_processed_slice_duration"]
    max_slice_overlap = context["params"]["max_processed_slice_overlap"]
    # an upper limit which should never be reached.
    # we will abort processing if the loop seems to never end
    # the overlap compensation logic, in extreme cases and unfortunate
    # param choices could cause that
    safety_max_iterations_limit = total_duration / max_slice_overlap
    all_no_speech_ranges = []
    while True:
        safety_max_iterations_limit -= 1
        next_process_until = min(total_duration, latest_processed_to + max_slice_duration)
        slice_context = get_sliced_context(context, latest_processed_to, next_process_until)

        no_speech_ranges_in_slice = find_high_confidence_no_speech_ranges(
            slice_context, min_no_speech_range_duration=min_no_speech_range_duration
        )

        done_processing = False
        if next_process_until < total_duration:
            # if a no speech region starts within the overlap range
            # start next slice before that speech to essentially
            # redetect it on the next slice (it might overflow onto the next)
            if no_speech_ranges_in_slice.shape[0] > 0:
                overlap_starts_at_least_at = next_process_until - max_slice_overlap
                no_speech_range_starts = no_speech_ranges_in_slice[0]
                no_speech_starts_within_overlap = no_speech_range_starts[
                    no_speech_range_starts > overlap_starts_at_least_at
                ]
                if no_speech_starts_within_overlap.size > 0:
                    # This is where the next slice will start
                    # Add safety margin before the start so next
                    # slice has ample duration to capture the start
                    # of the no speech range
                    next_process_until = max(
                        no_speech_starts_within_overlap[-1]
                        - context["params"]["no_speech_range_edge_safety_margin_duration"],
                        # But don't go all the way back to start of slice.
                        # this will result in an endless loop
                        # At least keep an "overlap" from the start of the slice
                        # which hopefully will roll forwar don the next iteration
                        latest_processed_to + max_slice_overlap,
                    )

                    # and we drop the last no speech range
                    # so the next slice can re-find it
                    print(no_speech_ranges_in_slice)
                    no_speech_ranges_in_slice = no_speech_ranges_in_slice[:, :-1]
                    print(no_speech_ranges_in_slice)
        else:
            done_processing = True

        if no_speech_ranges_in_slice.shape[0] > 0:
            all_no_speech_ranges.append(no_speech_ranges_in_slice)

        if done_processing or safety_max_iterations_limit == 0:
            break

        latest_processed_to = next_process_until

    if len(all_no_speech_ranges) > 0:
        all_no_speech_ranges = np.concatenate(all_no_speech_ranges, axis=1)
    else:
        all_no_speech_ranges = np.empty((2, 0))

    # return row-wise result ranges
    return _convert_no_speech_ranges_to_speech_ranges(all_no_speech_ranges.T, total_duration)
