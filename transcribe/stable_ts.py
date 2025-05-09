import os
from pathlib import Path

import stable_whisper
import torch
from faster_whisper import WhisperModel
from stable_whisper.result import WhisperResult

from segmentation.speech_ranges import get_audio_speech_ranges
from transcribe.definitions import FULL_TRANSCRIPTION_FILENAME
from utils.audio import load_audio_in_whisper_format
from utils.utils import parse_source_and_episode_from_filename
from vad.vad_io import get_frame_vad_probs_filename, load_frame_vad_probs


def get_speech_clips_from_vad(audio_file: str, input_vad_root_dir: str, min_no_speech_range_duration: float):
    clips = "0"
    found = False
    source, episode = parse_source_and_episode_from_filename(audio_file)
    vad_frame_filename = get_frame_vad_probs_filename(input_vad_root_dir, source, episode)
    if os.path.exists(vad_frame_filename):
        found = True
        speech_probs = load_frame_vad_probs(vad_frame_filename)
        speech_ranges = get_audio_speech_ranges(
            frame_level_speech_probs=speech_probs, min_no_speech_range_duration=min_no_speech_range_duration
        )
        clips_list = [f"{start:.2f},{end:.2f}" for start, end in speech_ranges]
        clips = ",".join(clips_list)
    return clips, found


def get_output_filename(audio_file_input: str, final_output_dir: str) -> str:
    source, episode = parse_source_and_episode_from_filename(audio_file_input)
    output_file_directory = os.path.join(final_output_dir, source, episode)
    return os.path.join(output_file_directory, FULL_TRANSCRIPTION_FILENAME)


def exclude_already_transcribed(audio_files: str, final_output_dir: str):
    pruned_audio_files = []
    for audio_file in audio_files:
        output_filename = get_output_filename(audio_file, final_output_dir)
        if not os.path.exists(output_filename):
            pruned_audio_files.append(audio_file)
    return pruned_audio_files


def transcribe(audio_files: str, final_output_dir: str, config: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prune audio files which are already processed
    if not config.get("force_reprocess", False):
        audio_files = exclude_already_transcribed(audio_files, final_output_dir)

    if len(audio_files) == 0:
        return

    input_vad_root_dir = config.get("vad_input_dir") or final_output_dir
    without_timestamps = config.get("without_timestamps", False)
    get_word_level_timestamp = config.get("get_word_level_timestamp", True)
    use_stable_ts = config.get("use_stable_ts", True)
    whisper_model_name = config.get("whisper_model_name", "tiny")
    language = config.get("language", "he")
    compute_type = config.get("compute_type", "int8")
    consider_speech_clips = config.get("consider_speech_clips", True)
    require_speech_clips = config.get("require_speech_clips", True)
    speech_clips_no_speech_min_duration = config.get("speech_clips_no_speech_min_duration", 4)

    faster_whisper_model_init_options = {
        "device": device,
        "compute_type": compute_type,
    }

    # Setup the model
    transcribe_fn = None
    if use_stable_ts:
        model = stable_whisper.load_faster_whisper(whisper_model_name, **faster_whisper_model_init_options)
        transcribe_fn = model.transcribe_stable
    else:
        model = WhisperModel(whisper_model_name, **faster_whisper_model_init_options)
        transcribe_fn = model.transcribe

    for audio_file in audio_files:
        output_filename = get_output_filename(audio_file, final_output_dir)

        transcribe_options = dict(
            # TODO - beam search, other configs?
            without_timestamps=without_timestamps,
            word_timestamps=get_word_level_timestamp,
        )

        # If vad probs exist, get speech clips to transcribe over the input file
        # Unless configured to not use clips
        if consider_speech_clips:
            clip_timestamps, vad_found = get_speech_clips_from_vad(
                audio_file, input_vad_root_dir, speech_clips_no_speech_min_duration
            )
            if not vad_found and require_speech_clips:
                continue
            transcribe_options["clip_timestamps"] = clip_timestamps

        audio = load_audio_in_whisper_format(
            audio_file,
        )

        result_raw = transcribe_fn(audio, language=language, **transcribe_options)
        if use_stable_ts:
            result: WhisperResult = result_raw
        else:
            segments, _ = result_raw
            final_segments = []
            for segment in segments:
                segment = segment._asdict()
                if (words := segment.get("words")) is not None:
                    segment["words"] = [w._asdict() for w in words]
                else:
                    del segment["words"]
                final_segments.append(segment)
            result: WhisperResult = WhisperResult(final_segments)

        output_filename = get_output_filename(audio_file, final_output_dir)
        os.makedirs(Path(output_filename).parent, exist_ok=True)
        result.save_as_json(output_filename)
