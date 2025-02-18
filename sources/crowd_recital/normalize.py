import argparse
import json
import pathlib
import sys
from dataclasses import asdict

import numpy as np
import stable_whisper
from tqdm import tqdm

from sources.crowd_recital.metadata import SessionMetadata
from utils.vtt import vtt_to_whisper_result

# Add these constants at the top of the file
DEFAULT_ALIGN_MODEL = "ivrit-ai/whisper-large-v3-turbo-ct2"
DEFAULT_ALIGN_DEVICE = "auto"
DEFAULT_FAILURE_THRESHOLD = 0.2


def calculate_quality_score(align_result: stable_whisper.WhisperResult, session_name="") -> float:
    """Calculate the quality score based on the median word probabilities from the alignment result."""
    try:
        per_segment_scores = []
        all_word_probs = []

        segments = list(align_result.segments)
        for segment in segments:
            segment_word_probs = []
            if segment.has_words:
                for word in segment.words:
                    if hasattr(word, "probability"):
                        prob = word.probability
                        segment_word_probs.append(prob)
                        all_word_probs.append(prob)

            if segment_word_probs:
                per_segment_scores.append(
                    {
                        "start": segment.start,
                        "end": segment.end,
                        "probability": round(float(np.median(segment_word_probs)), 4),
                    }
                )

        global_quality_score = round(float(np.median(all_word_probs)), 4) if all_word_probs else 0.0

    except Exception as e:
        if session_name:
            tqdm.write(f" - Failed to calculate quality score for session {session_name}: {e}")
        global_quality_score = 0.0
        per_segment_scores = []

    return global_quality_score, per_segment_scores


def add_normalize_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--align-model",
        type=str,
        default=DEFAULT_ALIGN_MODEL,
        help=f"Alignment model to use (default: {DEFAULT_ALIGN_MODEL})",
    )
    parser.add_argument(
        "--align-device",
        type=str,
        default=DEFAULT_ALIGN_DEVICE,
        help=f"Device for alignment. Use 'auto' to use cuda if available otherwise cpu (default: {DEFAULT_ALIGN_DEVICE})",
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Force reprocessing sessions even if transcript.aligned.vtt exists",
    )
    parser.add_argument(
        "--force-rescore",
        action="store_true",
        help="Force recalculation of quality score even if aligned transcript exists",
    )
    parser.add_argument(
        "--failure-threshold",
        type=float,
        default=DEFAULT_FAILURE_THRESHOLD,
        help=f"Failure threshold for alignment - portion of segments with 0 length tolerated before failing the alignment (default: {DEFAULT_FAILURE_THRESHOLD})",
    )


def normalize_sessions(
    input_folder: pathlib.Path,
    align_model: str = DEFAULT_ALIGN_MODEL,
    align_device: str = DEFAULT_ALIGN_DEVICE,
    force_reprocess: bool = False,
    force_rescore: bool = False,
    failure_threshold: float = DEFAULT_FAILURE_THRESHOLD,
    session_ids: list[str] = None,
) -> None:
    # Validate input folder exists before proceeding
    if not input_folder.exists() or not input_folder.is_dir():
        print(f"Input folder '{input_folder}' does not exist or is not a directory.", file=sys.stderr)
        return

    # Resolve device: if 'auto', use cuda if available for better performance
    device = align_device
    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    # Load the alignment model with int8 quantization for memory efficiency
    try:
        model = stable_whisper.load_faster_whisper(align_model, device=device, compute_type="int8")
    except Exception as e:
        print(f"Failed to load alignment model: {e}", file=sys.stderr)
        return

    # Find all session directories by looking for metadata.json files
    meta_files = list(input_folder.glob("*/metadata.json"))
    if session_ids:
        meta_files = [mf for mf in meta_files if mf.parent.name in session_ids]

    if not meta_files:
        print("No session metadata.json files found in input folder.")
        return

    # Process each session with progress tracking
    for meta_file in tqdm(meta_files, desc="Normalizing sessions"):
        session_dir = meta_file.parent
        tqdm.write(f"Processing session: {session_dir.name}")

        aligned_transcript_file = session_dir / "transcript.aligned.json"
        need_align = force_reprocess or (not aligned_transcript_file.exists())
        need_rescore = force_rescore or need_align
        if not (need_align or need_rescore):
            continue

        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                metadata = SessionMetadata(**json.load(f))
        except Exception as e:
            tqdm.write(f" - Failed to read metadata.json for session {session_dir.name}: {e}")
            continue

        if need_align:
            # Check if session language is supported
            doc_lang = metadata.document_language.lower()
            if doc_lang != "he":
                tqdm.write(
                    f" - Skipping session {session_dir.name}: unsupported language '{doc_lang}'. Only 'he' is supported."
                )
                continue
            language = doc_lang

            audio_file = session_dir / "audio.mka"
            transcript_vtt = session_dir / "transcript.vtt"
            if not audio_file.exists() or not transcript_vtt.exists():
                tqdm.write(f" - Skipping session {session_dir.name} because required files are missing.")
                continue

            try:
                with open(transcript_vtt, "r", encoding="utf-8") as f:
                    vtt_content = f.read()
                whisper_result = vtt_to_whisper_result(vtt_content)
            except Exception as e:
                tqdm.write(f" - Error processing transcript.vtt for session {session_dir.name}: {e}")
                continue

            try:
                align_result: stable_whisper.WhisperResult = model.align(
                    str(audio_file), whisper_result, language, failure_threshold=failure_threshold
                )
            except Exception as e:
                tqdm.write(f" - Alignment failed for session {session_dir.name}: {e}")
                continue
        else:
            try:
                align_result = stable_whisper.WhisperResult(str(aligned_transcript_file))
            except Exception as e:
                tqdm.write(f" - Failed to load aligned transcript for session {session_dir.name}: {e}")
                continue

        quality_score, per_segment_scores = calculate_quality_score(align_result, session_dir.name)
        metadata.quality_score = quality_score
        metadata.per_segment_quality_scores = per_segment_scores
        try:
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(asdict(metadata), f, indent=2)
        except Exception as e:
            tqdm.write(f" - Failed to update metadata.json for session {session_dir.name}: {e}")
            continue

        if need_align:
            try:
                align_result.save_as_json(str(aligned_transcript_file))
            except Exception as e:
                tqdm.write(f" - Failed to save aligned transcript for session {session_dir.name}: {e}")
                continue

        tqdm.write(f" - Processed session {session_dir.name}: quality score = {quality_score}")
