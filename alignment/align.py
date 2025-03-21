from pathlib import Path
from typing import Union

import stable_whisper
from faster_whisper import WhisperModel
from stable_whisper.whisper_compatibility import SAMPLE_RATE
from tqdm import tqdm

from alignment.utils import (
    create_transcript_from_segments,
    find_probable_segment_before_time,
    get_breakable_align_model,
    get_confusion_zone,
    get_text_from_segments,
)
from utils.vtt import vtt_to_whisper_result


def align_transcript_to_audio(
    audio_file: Path,
    transcript: Union[Path, stable_whisper.result.WhisperResult],
    model: Union[str, WhisperModel] = "ivrit-ai/whisper-large-v3-turbo-ct2",
    device: str = "auto",
    align_model_compute_type: str = "int8",
    language: str = "he",
    pre_confusion_zone_backward_skip_search_duration_window: int = 30,
    max_pre_confusion_zone_tries_before_skip: int = 2,
    unaligned_start_text_match_search_radius: int = 140,
    zero_duration_segments_failure_ratio: float = 0.2,
) -> stable_whisper.WhisperResult:
    """Align a transcript to audio using a robust, confusion-aware alignment algorithm.

    This function aligns a transcript to audio by breaking the alignment into pieces
    when it detects confusion zones. It uses a strategy of trying to align from before
    the confusion zone a few times, and if that fails, it skips the confusion zone
    and continues from after it.

    Note - Transcript is assumed to have relatively close (+-30s) timestamps across
    segments. If the transcripts has 0 timestamps - This algorithm would not work.
    Consider using stable_ts alignment directly.

    Args:
        audio_file: Path to the audio file to align to.
        transcript: Either a Path to a VTT file or a WhisperResult object containing
            the transcript to align.
        model: The model to use for alignment, either a path to a local model or a
            model identifier from the Hugging Face Hub.
        device: The device to use for inference, e.g., "cpu", "cuda", "auto".
            Can include a device index, e.g., "cuda:0".
        align_model_compute_type: The compute type to use for the model, e.g., "int8", "float16".
        language: The language code of the transcript.
        pre_confusion_zone_backward_skip_search_duration_window: Duration in seconds to look back
            when searching for a segment before a confusion zone.
        max_pre_confusion_zone_tries_before_skip: Maximum number of attempts to align from
            before a confusion zone before skipping it.
        unaligned_start_text_match_search_radius: Radius in characters to search for matching
            text when finding the start of a confusion zone in the unaligned transcript.
        zero_duration_segments_failure_ratio: Threshold for the ratio of zero-duration segments
            to total segments, above which alignment is considered failed.

    Returns:
        A WhisperResult containing the aligned transcript.
    """
    if isinstance(transcript, Path):
        unaligned = vtt_to_whisper_result(str(transcript))
    else:
        unaligned = transcript

    # If model is a string, load it using get_breakable_align_model
    if isinstance(model, str):
        model = get_breakable_align_model(model, device, align_model_compute_type)

    audio_metadata = stable_whisper.audio.utils.get_metadata(audio_file)
    audio_duration = audio_metadata["duration"] or 0

    # Initialize outer alignment loop
    slice_start = 0
    top_matched_unaligned_timestamp = 0
    done = False

    aligned_pieces: list[stable_whisper.Segment] = []
    min_confusion_zone_start = 0
    max_confusion_zone_end = 0
    current_pre_confusion_zone_tries = 0
    to_align_next = unaligned.text  # We aligns text not segments

    # Create a progress bar for the alignment process
    progress_bar = tqdm(total=audio_duration, unit="sec", desc="Aligning transcript to audio")
    while not done:
        # Get the audio slice
        audio = stable_whisper.audio.AudioLoader(
            str(audio_file), sr=SAMPLE_RATE, stream=True, load_sections=[[slice_start, None]]
        )

        # Align it until it stops
        aligned: stable_whisper.WhisperResult = model.align(
            audio, to_align_next, language=language, failure_threshold=zero_duration_segments_failure_ratio
        )

        # Update progress bar
        progress_bar.update(slice_start)

        # rebase aligned segments to full audio
        aligned.offset_time(slice_start)

        # Check if done == No confusion zone exists
        confusion_zone_start, confusion_zone_end = get_confusion_zone(aligned)
        if confusion_zone_start is None:
            # Keep aligned segments and stop aligning
            aligned_pieces.extend(aligned.segments)
            break

        # If the new confusion zone is outside the old one
        # we treat it as a new confusion zone
        if confusion_zone_start > max_confusion_zone_end:
            min_confusion_zone_start = confusion_zone_start
            max_confusion_zone_end = confusion_zone_end
            current_pre_confusion_zone_tries = 0
        else:
            # Expends the confusion zone with all previous tries
            # in this area
            min_confusion_zone_start = min(min_confusion_zone_start, confusion_zone_start)
            max_confusion_zone_end = max(max_confusion_zone_end, confusion_zone_end)

        probable_segment_before_confusion_zone = find_probable_segment_before_time(
            aligned,
            confusion_zone_start,
            pre_confusion_zone_backward_skip_search_duration_window,
        )

        # If there is a probable segment before confusion zone
        if probable_segment_before_confusion_zone is not None:
            # Keep properly aligned segments up to it including
            segments_already_aligned = aligned.segments[: probable_segment_before_confusion_zone.id + 1]
            aligned_pieces.extend(segments_already_aligned)

            # point to audio start for next try
            slice_start = probable_segment_before_confusion_zone.end

            # Prepare not aligned text for next try
            to_align_next = get_text_from_segments(aligned.segments[probable_segment_before_confusion_zone.id + 1 :])

            # If we have more tries left for the "pre confusion zone" retry strategy
            if current_pre_confusion_zone_tries < max_pre_confusion_zone_tries_before_skip:
                progress_bar.write(f"Retry alignment from before confusion zone: {slice_start}")
                # another pre confusion zone try is done
                current_pre_confusion_zone_tries += 1
                continue
        else:
            # Else - to_align_next is all the text we tried to align in this attempt
            # of course we will skip some of it after deciding where to skip to

            # Also, slice_start still points to where we need to start aligning
            # from - again, probably we will skip forward soon
            # but since there was no probable segment to mark the anchor the start of the confusion zone
            # just assume the confusion zone start is the start of the audio
            min_confusion_zone_start = slice_start

        # skip the confusion zone if no pre confusion zone retry
        # is possible or allowed.

        progress_bar.write(f"Skipping confusion zone: {min_confusion_zone_start} - {max_confusion_zone_end}")

        # skipping - resets the jump back strategy allowed tries
        current_pre_confusion_zone_tries = 0

        # Discover which segments will be skipped
        # we don't want to lose any text - we want to skip aligning it
        # and keep the unaligned as a reasonable estimate

        # problem is - we have the text of the unaligned but the segmentation
        # does not match the unaligned which we use to pickup the estimated skip point.
        # this - we will use text matching to find the locations we are skipping over
        # and which segments (or parts of them) are left unaligned.

        # first get some prefix from the text to align next - this is what we could not align
        # after the last try ended.
        # this usually cover the entire confusion zone up to the end
        # of the entire text for the audio file
        text_at_start_of_confusion_zone = to_align_next[: unaligned_start_text_match_search_radius // 2]

        # get segments from the unaligned around confusion zone to find the text within
        # we assume timing is off by not too much, so we expand the search area a little
        search_around_time = (
            min_confusion_zone_start
            if probable_segment_before_confusion_zone is None
            else probable_segment_before_confusion_zone.end
        )
        search_in_unaligned_window_time_start = max(
            # never search in unaligned parts already matched against
            # so we reduce the risk of matching to the past
            top_matched_unaligned_timestamp,
            search_around_time - unaligned_start_text_match_search_radius,
        )
        search_in_unaligned_window_time_end = search_around_time + unaligned_start_text_match_search_radius
        segments_around_confusion_zone = unaligned.get_content_by_time(
            (search_in_unaligned_window_time_start, search_in_unaligned_window_time_end),
            segment_level=True,
        )
        text_around_confusion_zone = get_text_from_segments(segments_around_confusion_zone)

        # where can we find the prefix text ?
        found_at_text_idx = text_around_confusion_zone.find(text_at_start_of_confusion_zone)
        if found_at_text_idx == -1:
            raise ValueError(
                f"Could not find matching text in confusion zone, slice_start: {slice_start}\ntext: `{text_at_start_of_confusion_zone}`"
                f"\ntext in zone: `{text_around_confusion_zone}`"
            )

        # find the segment that contains the start idx
        # and the index of the text within that segment
        curr_matched_segment_idx = 0
        index_within_segment = found_at_text_idx
        text_len_so_far = len(segments_around_confusion_zone[curr_matched_segment_idx].text)
        while text_len_so_far <= found_at_text_idx:
            index_within_segment = found_at_text_idx - text_len_so_far
            curr_matched_segment_idx += 1
            text_len_so_far += len(segments_around_confusion_zone[curr_matched_segment_idx].text)

        # get the initial segment
        initial_unaligned_segment_in_confusion_zone = segments_around_confusion_zone[curr_matched_segment_idx]
        initial_unaligned_segment_id_in_confusion_zone = initial_unaligned_segment_in_confusion_zone.id

        # Recall the latest known aligned timestamp - upcoming segments cannot
        # timestamp below it
        top_aligned_timestamp = aligned_pieces[-1].end if aligned_pieces else 0

        # and if the text is mid-point within the segment. we need only the matched part.
        if index_within_segment > 0:
            initial_unaligned_segment_in_confusion_zone = stable_whisper.result.Segment(
                # The start of the sliced segment is:
                # Where we intended to start - if probable prev segment was found (slice_start)
                # or the original start which had no probable segment at all (slice_start)
                # and never less than the highest aligned timestamp since this
                # will break the monotonicity of the aligned timestamps (top_aligned_timestamp)
                start=min(
                    max(
                        top_aligned_timestamp,
                        slice_start,
                    ),
                    # But cannot go beyond the end of this segment
                    initial_unaligned_segment_in_confusion_zone.end,
                ),
                end=max(top_aligned_timestamp, initial_unaligned_segment_in_confusion_zone.end),
                text=initial_unaligned_segment_in_confusion_zone.text[index_within_segment:],
            )

        # in the confusion zone we cannot trust the aligned times - so we work on the unaligned
        # which is expected to have reasonable approximate timing
        # Find the segments we will continue aligning next
        segments_after_assumed_confusion_zone = unaligned.get_content_by_time(
            (max_confusion_zone_end, unaligned.segments[-1].end), segment_level=True
        )

        first_segment_to_continue_aligning = segments_after_assumed_confusion_zone[0]

        # find the segments within the confusion zone that we will skip
        confusing_segments_to_skip = unaligned.segments[
            initial_unaligned_segment_id_in_confusion_zone
            # +1, since we will prepend this first segment (it might required prefix removal)
            + 1 : first_segment_to_continue_aligning.id
        ]
        # prepend the initial segment (which might have had prefix removal)
        # containing only the part which is skipped
        confusing_segments_to_skip = [initial_unaligned_segment_in_confusion_zone] + confusing_segments_to_skip

        # Ensure none of the segments has a start/end below the top aligned timestamp
        for segment in confusing_segments_to_skip:
            segment.start = max(segment.start, top_aligned_timestamp)
            segment.end = max(segment.end, top_aligned_timestamp)

        aligned_pieces.extend(confusing_segments_to_skip)  # consider this done (although it's unaligned == estimated)
        to_align_next = get_text_from_segments(segments_after_assumed_confusion_zone)

        # Mark the top text we took from the unaligned - so we cannot match earlier than that
        # on next iterations
        top_matched_unaligned_timestamp = confusing_segments_to_skip[-1].end

        # next audio start is the confusion zone end
        slice_start = first_segment_to_continue_aligning.start

        # Forget prev confusion zone
        min_confusion_zone_start = 0
        max_confusion_zone_end = 0

    # some skipped segments have no words and cannot
    # co exist with the aligned segments.
    # pick those up and align them specifically, using "align words"
    # which ensure they will not drift - we don't expect them to be aligned
    # eventually, they are probably the confusing ones - but at least they will
    # have words
    segs_to_align_words = []
    segs_to_align_positions = []
    for seg_pos, seg in enumerate(aligned_pieces):
        if seg.has_words:
            continue
        seg = seg.to_dict()
        segs_to_align_words.append(seg)
        segs_to_align_positions.append(seg_pos)

    to_word_align = stable_whisper.WhisperResult({"segments": segs_to_align_words})

    audio = stable_whisper.audio.AudioLoader(
        str(audio_file),
        sr=SAMPLE_RATE,
        stream=True,
    )
    word_aligned: stable_whisper.WhisperResult = model.align_words(
        audio, to_word_align, language=language, regroup=False
    )

    # replace each skipped segment with it's aligned match
    for seg_pos, word_aligned_seg in zip(segs_to_align_positions, word_aligned.segments):
        assert (
            aligned_pieces[seg_pos].text == word_aligned_seg.text
        ), f"before {aligned_pieces[seg_pos].text} is not {word_aligned_seg.text}"
        aligned_pieces[seg_pos] = word_aligned_seg

    final_aligned = create_transcript_from_segments(aligned_pieces)

    # Close the progress bar
    progress_bar.close()

    return final_aligned
