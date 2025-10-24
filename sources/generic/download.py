from itertools import chain
import argparse
import json
import logging
import pathlib
import re
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from sources.generic.normalize import add_normalize_args, normalize_generic_entries
from sources.generic.metadata import GenericMetadata, source_type
from sources.generic.manifest import build_manifest
from utils.audio import extract_audio_from_media, get_audio_info


def process_transcripts(
    transcript_file: pathlib.Path,
    output_dir: pathlib.Path,
    entry_id: str,
    force_reprocess: bool = False,
    abort_on_error: bool = False,
):
    """
    Process the transcript files for a generic entry.

    Args:
        transcript_file: Path to the transcript file
        output_dir: Output directory where processed files will be saved
        entry_id: ID of the entry
        force_reprocess: Whether to force reprocessing even if files already exist
        abort_on_error: Whether to abort on error

    Returns:
        bool: True if processing was successful, False otherwise
    """
    try:
        # Validate file extension - only support .txt files for now
        if transcript_file.suffix.lower() != ".txt":
            error_msg = f"Unsupported transcript file type: {transcript_file.suffix}. Only .txt files are supported."
            if abort_on_error:
                raise ValueError(error_msg)
            logging.warning(error_msg)
            return False

        # Create entry-specific output directory
        entry_output_dir = output_dir / entry_id
        entry_output_dir.mkdir(parents=True, exist_ok=True)

        # Define target transcript file path
        target_transcript = entry_output_dir / "transcript.txt"

        # Check if we need to reprocess
        if not force_reprocess and target_transcript.exists():
            logging.info(f"Transcript already exists for entry {entry_id}, skipping.")
            return True

        # Copy the transcript file to the output directory
        import shutil

        shutil.copy2(transcript_file, target_transcript)
        logging.info(f"Copied transcript from {transcript_file} to {target_transcript}")

        return True

    except Exception as e:
        error_msg = f"Failed to process transcript for entry {entry_id}: {e}"
        if abort_on_error:
            raise e
        logging.error(error_msg)
        return False


def process_av(
    audio_file: pathlib.Path,
    output_dir: pathlib.Path,
    entry_id: str,
    force_reprocess: bool = False,
) -> Tuple[bool, Optional[float]]:
    """
    Process the audio/video file for a generic entry.

    Args:
        audio_file: Path to the audio file
        output_dir: Output directory where processed files will be saved
        entry_id: ID of the entry
        force_reprocess: Whether to force reprocessing even if files already exist

    Returns:
        Tuple[bool, Optional[float]]: (success, duration)
    """
    try:
        # Create entry-specific output directory
        entry_output_dir = output_dir / entry_id
        entry_output_dir.mkdir(parents=True, exist_ok=True)

        # Define target audio file path (without extension, extract_audio_from_media will add it)
        target_audio_base = entry_output_dir / "audio"

        # Check if we need to reprocess - look for any audio file in the output directory
        existing_audio_files = list(entry_output_dir.glob("audio.*"))
        if not force_reprocess and existing_audio_files:
            logging.info(f"Audio already exists for entry {entry_id}, skipping extraction.")
            # Get duration from existing file
            if existing_audio_files:
                audio_info = get_audio_info(str(existing_audio_files[0]))
                duration = audio_info.duration if audio_info else None
                return True, duration

        # Extract audio from the media file
        output_audio_file = extract_audio_from_media(str(audio_file), str(target_audio_base), create_light_version=True)
        logging.info(f"Extracted audio from {audio_file} to {output_audio_file}")

        # Get audio information including duration
        audio_info = get_audio_info(output_audio_file)
        duration = audio_info.duration if audio_info else None

        if duration is None:
            logging.warning(f"Could not determine duration for {output_audio_file}")
        else:
            logging.info(f"Audio duration: {duration:.2f} seconds")

        return True, duration

    except Exception as e:
        error_msg = f"Failed to process audio for entry {entry_id}: {e}"
        logging.error(error_msg)
        return False, None


def extract_metadata_fields(metadata_file: pathlib.Path, field_specs_to_extract: List[str]) -> Dict[str, Any]:
    # Parse field specs: <source>:<target>[:should_hash]
    parsed_fields = []
    for spec in field_specs_to_extract:
        parts = spec.split(":")
        if len(parts) == 2:
            src, target = parts
            should_hash = False
        elif len(parts) == 3:
            src, target, should_hash_str = parts
            should_hash = should_hash_str in ("1", "true", "yes")
        else:
            continue  # Invalid format
        parsed_fields.append((src, target, should_hash))

    # only supported targets are accepted
    supported_target_md_fields = ["user_id"]
    parsed_fields = [field for field in parsed_fields if field[1] in supported_target_md_fields]

    if not parsed_fields:
        return {}

    with open(metadata_file, "r") as f:
        metadata = json.load(f)
        target_md_fields = {}
        for src, target, should_hash in parsed_fields:
            value = metadata.get(src, None)
            if value is not None and should_hash:
                import hashlib

                value = hashlib.md5(str(value).encode()).hexdigest()
            target_md_fields[target] = value
        return target_md_fields


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and process generic audio/transcript datasets.")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Generic Dataset Input directory",
    )
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        choices=["he", "yi"],
        help="Language of the dataset (Hebrew or Yiddish)",
    )
    parser.add_argument("--input-structure-format", type=str, default="flat", help="Input directory structure format")
    parser.add_argument(
        "--input-audio-file-globs", type=str, default=["*.mp3"], help="Glob pattern for audio files", nargs="+"
    )
    parser.add_argument(
        "--input-transcript-file-globs",
        type=str,
        default=["*.txt"],
        help="Glob pattern for transcript files",
        nargs="+",
    )
    parser.add_argument(
        "--input-metadata-file-globs",
        type=str,
        default=["*.json"],
        help="Glob pattern for metadata files",
        nargs="+",
    )
    parser.add_argument(
        "--extract-md-fields",
        type=str,
        nargs="+",
        default=[],
        help=(
            "Fields to extract from metadata files. Format is <source_md_field_name>:<target_md_field_name>[:should_hash]."
            " should_hash (optional, 1 to hash the value with SHA256 for PII fields like user_id)."
            " Can be specified multiple times. Supports only 'user_id' target md field"
        ),
    )
    parser.add_argument("--input-source-id", type=str, default="unknown", help="Source ID for the input dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory where normalized dataset will be saved.",
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Force re-process of all content.",
    )
    parser.add_argument(
        "--force-av-reprocess",
        action="store_true",
        help="Force re-process of audio even if it exist.",
    )
    parser.add_argument(
        "--force-transcript-reprocess",
        action="store_true",
        help="Force re-process of transcripts even if it exist.",
    )
    parser.add_argument(
        "--force-generate-metadata",
        action="store_true",
        help="Force re-generate of metadata even if it exist.",
    )
    parser.add_argument(
        "--abort-on-error",
        action="store_true",
        help="Will not skip processing errors, rather throw the error and abort the whole run.",
    )
    parser.add_argument(
        "--skip-normalize",
        action="store_true",
        help="Skip normalization process",
    )
    parser.add_argument(
        "--skip-manifest",
        action="store_true",
        help="Skip generating manifest CSV",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=None,
        help="Maximum number of entries to process",
    )
    parser.add_argument(
        "--entry-ids",
        type=str,
        nargs="+",
        default=[],
        help="Specific entry IDs to process (if not specified, process all)",
    )
    parser.add_argument(
        "--ignore-missing-files",
        action="store_true",
        help="Continue processing even if some audio or transcript files are missing",
    )
    parser.add_argument(
        "--logs-folder",
        type=str,
        help="Folder to store log files. If not specified, logging is disabled.",
    )

    # Add normalization-related arguments
    add_normalize_args(parser)

    args = parser.parse_args()

    input_dir = pathlib.Path(args.input_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging based on logs_folder
    # By default, disable all logging
    logging.basicConfig(level=logging.CRITICAL + 1)  # Set level higher than CRITICAL to disable all logging

    if hasattr(args, "logs_folder") and args.logs_folder:
        logs_folder = pathlib.Path(args.logs_folder)
        logs_folder.mkdir(parents=True, exist_ok=True)

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        # Remove any existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Create a rotating file handler
        log_file = logs_folder / "download_log"
        file_handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=5)  # 5MB
        file_handler.setLevel(logging.INFO)

        # Create formatter and add it to the handler
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)

        # Add the handler to the logger
        root_logger.addHandler(file_handler)

        # Log start of process
        logging.info(f"Starting Generic download process with output directory: {output_dir}")

    # Validate input directories
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input directory '{input_dir}' does not exist or is not a directory.")
        return

    if args.input_structure_format != "flat":
        raise ValueError(f"Unsupported input structure format: {args.input_structure_format}")

    entries = []

    # Currently the only supported structure is flat
    # This structure is a single folder, containing two files per entry - audio and transcript
    # optionally, also include a metadata file with a json suffix and format.
    # the files name is the same and is the entry ID, the extension mark the type of file.
    # we use above glob patterns to find the files.

    # Find all audio files using the provided glob patterns
    input_audio_files = list(chain.from_iterable(input_dir.glob(patt) for patt in args.input_audio_file_globs))
    logging.info(f"Found {len(input_audio_files)} audio files matching patterns: {args.input_audio_file_globs}")

    # Find all transcript files using the provided glob patterns
    input_transcript_files = list(
        chain.from_iterable(input_dir.glob(patt) for patt in args.input_transcript_file_globs)
    )
    logging.info(
        f"Found {len(input_transcript_files)} transcript files matching patterns: {args.input_transcript_file_globs}"
    )

    # Find all metadata files using the provided glob patterns
    input_metadata_files = []
    if args.input_metadata_file_globs is not None:
        input_metadata_files = list(
            chain.from_iterable(input_dir.glob(patt) for patt in args.input_metadata_file_globs)
        )
        logging.info(
            f"Found {len(input_metadata_files)} metadata files matching patterns: {args.input_metadata_file_globs}"
        )

    # Create dictionaries mapping base names to file paths
    audio_files_by_basename = {}
    for audio_file in input_audio_files:
        basename = audio_file.stem  # Get filename without extension
        audio_files_by_basename[basename] = audio_file

    transcript_files_by_basename = {}
    for transcript_file in input_transcript_files:
        basename = transcript_file.stem  # Get filename without extension
        transcript_files_by_basename[basename] = transcript_file

    metadata_files_by_basename = {}
    for metadata_file in input_metadata_files:
        basename = metadata_file.stem  # Get filename without extension
        metadata_files_by_basename[basename] = metadata_file

    # Match audio and transcript files by base name
    all_basenames = set(audio_files_by_basename.keys()) | set(transcript_files_by_basename.keys())

    for basename in all_basenames:
        audio_file = audio_files_by_basename.get(basename)
        transcript_file = transcript_files_by_basename.get(basename)
        maybe_metadata_file = metadata_files_by_basename.get(basename, None)

        # Check for missing files
        if audio_file is None:
            msg = f"Missing audio file for entry '{basename}'"
            if args.ignore_missing_files:
                logging.warning(f"{msg} - skipping entry")
                continue
            else:
                raise FileNotFoundError(f"{msg}. Use --ignore-missing-files to skip missing entries.")

        if transcript_file is None:
            msg = f"Missing transcript file for entry '{basename}'"
            if args.ignore_missing_files:
                logging.warning(f"{msg} - skipping entry")
                continue
            else:
                raise FileNotFoundError(f"{msg}. Use --ignore-missing-files to skip missing entries.")

        # Add matched pair to entries list
        entries.append((audio_file, transcript_file, maybe_metadata_file, basename))

    logging.info(f"Successfully matched {len(entries)} audio-transcript-metadata? pairs")

    # Filter by entry IDs if specified
    if args.entry_ids:
        original_count = len(entries)
        entries = [
            (audio, transcript, md, entry_id)
            for audio, transcript, md, entry_id in entries
            if entry_id in args.entry_ids
        ]
        logging.info(f"Filtered by entry IDs {args.entry_ids}: {original_count} -> {len(entries)} entries")

    # Take first max_entries if specified
    if args.max_entries is not None and len(entries) > args.max_entries:
        entries = entries[: args.max_entries]
        logging.info(f"Limited to first {args.max_entries} entries")

    if not entries:  # entries is a list of tuples (audio_file, transcript_file, entry_id)
        logging.info("No entries found in the input media directory.")
        return

    logging.info(f"Found {len(entries)} entry IDs.")

    # Process each entry
    any_reprocess = (
        args.force_reprocess
        or args.force_av_reprocess
        or args.force_transcript_reprocess
        or args.force_normalize_reprocess
        or args.force_generate_metadata
    )
    for audio_file, transcript_file, maybe_metadata_file, entry_id in tqdm(
        entries, desc="Processing entries", total=len(entries)
    ):
        try:
            # Check if this entry has already been processed
            entry_output_dir = output_dir / entry_id
            if entry_output_dir.exists() and not any_reprocess:
                metadata_file = entry_output_dir / "metadata.json"
                if metadata_file.exists():
                    tqdm.write(f" - Entry {entry_id} already processed. Skipping.")
                    continue

            tqdm.write(f"Processing entry: {entry_id}")

            # Process transcripts
            tqdm.write(" - Processing transcripts...")
            transcript_success = process_transcripts(
                transcript_file,
                output_dir,
                entry_id,
                args.force_transcript_reprocess or args.force_reprocess,
                abort_on_error=args.abort_on_error,
            )
            if not transcript_success:
                tqdm.write(f" - Failed to process transcripts for entry {entry_id}. Skipping.")
                continue

            # Process AV
            tqdm.write(" - Processing AV...")
            av_success, duration = process_av(
                audio_file, output_dir, entry_id, args.force_av_reprocess or args.force_reprocess
            )
            if not av_success:
                msg = f" - Failed to process AV for entry {entry_id}. Skipping."
                tqdm.write(msg)
                logging.warning(msg)
                continue

            extracted_md = {}
            if maybe_metadata_file and args.extract_md_fields:
                tqdm.write(" - Extracting metadata fields...")
                extracted_md = extract_metadata_fields(maybe_metadata_file, args.extract_md_fields)

            # Create metadata
            entry_metadata = GenericMetadata(
                source_type=source_type,
                source_id=args.input_source_id,
                source_entry_id=entry_id,
                language=args.language,
                duration=round(duration or 0.0, 2),
                **extracted_md,
            )

            # Save metadata
            metadata_file = entry_output_dir / "metadata.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                f.write(entry_metadata.model_dump_json(indent=2))

            tqdm.write(f" - Successfully processed entry {entry_id}")
        except Exception as e:
            msg = f" - ERROR: Unexpected error processing entry {entry_id}: {e}"
            tqdm.write(msg)
            logging.warning(msg)
            if args.abort_on_error:
                raise e
            tqdm.write(f" - Skipping to next entry")

    # After downloads complete, process normalization if not skipped
    if not args.skip_normalize:
        print("Starting normalization process...")
        normalize_generic_entries(
            output_dir,
            align_model=args.align_model,
            align_devices=args.align_devices,
            align_device_density=args.align_device_density,
            # any force processing upstream or directly force normalize?
            force_normalize_reprocess=args.force_reprocess
            or args.force_av_reprocess
            or args.force_transcript_reprocess
            or args.force_normalize_reprocess,
            # Any reason to write out scores to md or forced rescore?
            force_rescore=args.force_rescore or args.force_generate_metadata,
            failure_threshold=args.failure_threshold,
            entry_ids=args.entry_ids,
            abort_on_error=args.abort_on_error,
        )

    # Generate manifest if not skipped
    if not args.skip_manifest:
        print("Generating manifest CSV...")
        build_manifest(str(output_dir))


if __name__ == "__main__":
    import sys

    print("This module is not intended to be executed directly. Please use the top-level download.py.", file=sys.stderr)
