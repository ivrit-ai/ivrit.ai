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


def add_normalize_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--align-model",
        type=str,
        default="ivrit-ai/whisper-large-v3-turbo-ct2",
        help="Alignment model to use (default: ivrit-ai/whisper-large-v3-turbo-ct2)",
    )
    parser.add_argument(
        "--align-device",
        type=str,
        default="auto",
        help="Device for alignment. Use 'auto' to use cuda if available otherwise cpu (default: auto)",
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Force reprocessing sessions even if transcript.aligned.vtt exists",
    )
    parser.add_argument(
        "--failure-threshold",
        type=float,
        default=0.2,
        help="Failure threshold for alignment - portion of segments with 0 length tolerated before failing the alignment (default: 0.2)",
    )


def normalize_sessions(
    input_folder: pathlib.Path, align_model: str, align_device: str, force_reprocess: bool, failure_threshold: float
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
    if not meta_files:
        print("No session metadata.json files found in input folder.")
        return

    # Process each session with progress tracking
    for meta_file in tqdm(meta_files, desc="Normalizing sessions"):
        session_dir = meta_file.parent
        tqdm.write(f"Processing session: {session_dir.name}")

        # Skip if already processed unless force_reprocess is True
        aligned_transcript_file = session_dir / "transcript.aligned.json"
        if aligned_transcript_file.exists() and not force_reprocess:
            continue

        # Check for required input files
        audio_file = session_dir / "audio.mka"
        transcript_vtt = session_dir / "transcript.vtt"
        if not audio_file.exists() or not transcript_vtt.exists():
            tqdm.write(f" - Skipping session {session_dir.name} because required files are missing.")
            continue

        # Load session metadata to get language info
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                metadata: SessionMetadata = SessionMetadata(**json.load(f))
        except Exception as e:
            tqdm.write(f" - Error reading metadata.json in session {session_dir.name}: {e}")
            continue

        # Currently only supporting Hebrew language sessions
        doc_lang = metadata.document_language.lower()
        if doc_lang != "he":
            tqdm.write(
                f" - Skipping session {session_dir.name}: unsupported language '{doc_lang}'. Only 'he' is supported."
            )
            continue
        language = doc_lang

        # Convert VTT transcript to Whisper format for alignment
        try:
            with open(transcript_vtt, "r", encoding="utf-8") as f:
                vtt_content = f.read()
            whisper_result = vtt_to_whisper_result(vtt_content)
        except Exception as e:
            tqdm.write(f" - Error processing transcript.vtt in session {session_dir.name}: {e}")
            continue

        # Perform alignment between audio and transcript
        try:
            align_result: stable_whisper.WhisperResult = model.align(
                str(audio_file), whisper_result, language, failure_threshold=failure_threshold
            )
        except Exception as e:
            tqdm.write(f" - Alignment failed for session {session_dir.name}: {e}")
            continue

        # Save aligned transcript in WhisperResult format
        try:
            align_result.save_as_json(str(aligned_transcript_file))
        except Exception as e:
            tqdm.write(f" - Failed to save aligned transcript for session {session_dir.name}: {e}")
            continue

        # Calculate quality score based on median word alignment confidence
        try:
            quality_score = 0.0
            if align_result:
                words = list(align_result.all_words())
                if words:
                    probabilities = [word.probability for word in words]
                    quality_score = float(np.median(probabilities))
        except Exception as e:
            tqdm.write(f" - Failed to calculate quality score for session {session_dir.name}: {e}")
            quality_score = 0.0

        # Update metadata with quality score for future reference
        metadata.quality_score = quality_score
        try:
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(asdict(metadata), f, indent=2)
        except Exception as e:
            tqdm.write(f" - Failed to update metadata.json for session {session_dir.name}: {e}")
            continue

        tqdm.write(f" - Processed session {session_dir.name}: quality score = {quality_score}")
