import argparse
import logging
import pathlib
import re
from logging.handlers import RotatingFileHandler
from typing import List, Optional, Tuple

from tqdm import tqdm

from sources.knesset.extraction import process_transcripts
from sources.knesset.manifest import build_manifest
from sources.knesset.metadata import PlenumMetadata, plenum_source_id, source_type
from sources.knesset.normalize import add_normalize_args, normalize_plenums
from utils.audio import extract_audio_from_media, get_audio_info


def find_media_file(plenum_dir: pathlib.Path) -> Optional[pathlib.Path]:
    """Find the media file (mp3, mp4, wmv, or avi) in the plenum directory.

    If multiple files exist:
    1. Prefer the largest mp4 file if any mp4 files exist
    2. Otherwise return the largest file by size among all media types
    """
    found_media_files = []
    for ext in [".mp3", ".mp4", ".wmv", ".avi"]:
        for file in plenum_dir.glob(f"*{ext}"):
            found_media_files.append(file)

    if not found_media_files:
        return None

    # If only one file, return it
    if len(found_media_files) == 1:
        return found_media_files[0]

    # If multiple files exist, check for mp4 first
    mp4_files = [file for file in found_media_files if file.suffix.lower() == ".mp4"]
    if mp4_files:
        # Return the largest mp4 file
        return max(mp4_files, key=lambda file: file.stat().st_size)

    # Otherwise return the largest file among all media types
    return max(found_media_files, key=lambda file: file.stat().st_size)


def find_protocol_xml(plenum_transcripts_dir: pathlib.Path) -> Optional[pathlib.Path]:
    if not plenum_transcripts_dir.exists() or not plenum_transcripts_dir.is_dir():
        return None

    # only one is expected
    found_protocol_xml_files = list(plenum_transcripts_dir.glob(f"protocol_*.xml"))
    assert len(found_protocol_xml_files) <= 1
    if found_protocol_xml_files:
        return found_protocol_xml_files[0]
    return None


def find_protocol_html_files(plenum_transcripts_dir: pathlib.Path) -> List[pathlib.Path]:
    if not plenum_transcripts_dir.exists() or not plenum_transcripts_dir.is_dir():
        return []

    return list(plenum_transcripts_dir.glob(f"protocol_*.html"))


def extract_date_from_media_filename(media_path: pathlib.Path) -> Optional[str]:
    """
    Extract the date from the media filename in the format YYYY_MM_DD and convert it to ISO format YYYY-MM-DD.

    Args:
        media_path: Path to the media file

    Returns:
        Optional[str]: The date in ISO format (YYYY-MM-DD) or None if the pattern doesn't match
    """
    filename = media_path.name
    # Pattern to match YYYY_MM_DD at the beginning of the filename
    date_pattern = re.compile(r"^(\d{4})_(\d{2})_(\d{2})_")
    match = date_pattern.match(filename)

    if match:
        year, month, day = match.groups()
        return f"{year}-{month}-{day}"

    return None


def process_av(
    media_path: pathlib.Path,
    output_dir: pathlib.Path,
    plenum_id: str,
    force_reprocess: bool = False,
) -> Tuple[bool, Optional[float]]:
    """
    Process the audio/video file for a plenum.

    Args:
        media_path: Path to the media file
        output_dir: Output directory where processed files will be saved
        plenum_id: ID of the plenum

    Returns:
        Tuple[bool, Optional[float]]: (success, duration)
    """
    # This is a stub function that will be implemented later
    # For now, we'll just copy the media file to the output directory

    try:
        plenum_output_dir = output_dir / plenum_id
        plenum_output_dir.mkdir(parents=True, exist_ok=True)

        # Copy the media file to the output directory as audio.mka
        audio_output_base_path = plenum_output_dir / "audio"

        # find any existing audio file
        audio_files = list(plenum_output_dir.glob("audio.*"))
        audio_output_path = None
        if audio_files:
            if force_reprocess:
                for file in audio_files:
                    file.unlink()
            else:
                audio_output_path = pathlib.Path(audio_files[0])

        if audio_output_path is None:
            audio_output_path = extract_audio_from_media(str(media_path), str(audio_output_base_path))

        audio_info = get_audio_info(str(audio_output_path))
        duration = audio_info.duration if audio_info else None

        return True, duration
    except Exception as e:
        print(f"Error processing AV for plenum {plenum_id}: {e}")
        return False, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and process Knesset recordings and transcripts.")
    parser.add_argument(
        "--input-media-dir",
        type=str,
        required=True,
        help="Input directory containing media files organized by plenum ID.",
    )
    parser.add_argument(
        "--input-transcripts-dir",
        type=str,
        required=True,
        help="Input directory containing transcript files (XML and HTML).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory where processed files will be saved.",
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
        "--abort-on-error",
        action="store_true",
        help="Will not skip processing errors, rather throw the error and abort the whole run.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of files even if they exist.",
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
        "--plenum-ids",
        type=str,
        nargs="+",
        default=[],
        help="Filter to process only the specified plenum ids (can be specified multiple times)",
    )
    parser.add_argument(
        "--max-recordings",
        type=int,
        default=None,
        help="Maximum number of plenum recordings to process in this run",
    )
    parser.add_argument(
        "--logs-folder",
        type=str,
        help="Folder to store log files. If not specified, logging is disabled.",
    )

    # Add normalization-related arguments
    add_normalize_args(parser)

    args = parser.parse_args()

    input_media_dir = pathlib.Path(args.input_media_dir)
    input_transcripts_dir = pathlib.Path(args.input_transcripts_dir)
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
        logging.info(f"Starting Knesset download process with output directory: {output_dir}")

    # Validate input directories
    if not input_media_dir.exists() or not input_media_dir.is_dir():
        print(f"Input media directory '{input_media_dir}' does not exist or is not a directory.")
        return
    if not input_transcripts_dir.exists() or not input_transcripts_dir.is_dir():
        print(f"Input transcripts directory '{input_transcripts_dir}' does not exist or is not a directory.")
        return

    # Scan the media input directory to find all top-level folders (plenum IDs)
    plenum_av_dirs = [d for d in input_media_dir.iterdir() if d.is_dir()]
    plenum_ids = [d.name for d in plenum_av_dirs]

    # Filter by plenum IDs if specified
    if args.plenum_ids:
        plenum_ids = [pid for pid in plenum_ids if pid in args.plenum_ids]

    if args.max_recordings:
        plenum_ids = plenum_ids[: args.max_recordings]

    if not plenum_ids:
        logging.info("No plenum IDs found in the input media directory.")
        return

    logging.info(f"Found {len(plenum_ids)} plenum IDs.")

    # Process each plenum
    for plenum_id in tqdm(plenum_ids, desc="Processing plenums", total=len(plenum_ids)):
        try:
            tqdm.write(f"Processing plenum: {plenum_id}")

            plenum_av_dir = input_media_dir / plenum_id

            # Check if this plenum has already been processed
            plenum_output_dir = output_dir / plenum_id
            any_reprocess = args.force_reprocess or args.force_av_reprocess or args.force_transcript_reprocess
            if plenum_output_dir.exists() and not any_reprocess:
                metadata_file = plenum_output_dir / "metadata.json"
                if metadata_file.exists():
                    tqdm.write(f" - Plenum {plenum_id} already processed. Skipping.")
                    continue

            # Find the media file
            media_path = find_media_file(plenum_av_dir)
            if not media_path:
                tqdm.write(f" - No media file found for plenum {plenum_id}. Skipping.")
                continue

            plenum_transcripts_dir = input_transcripts_dir / plenum_id

            # Find the protocol XML file
            protocol_xml_path = find_protocol_xml(plenum_transcripts_dir)
            if not protocol_xml_path:
                tqdm.write(f" - No protocol XML file found for plenum {plenum_id}. Skipping.")
                continue

            # Find all protocol HTML files
            protocol_html_paths = find_protocol_html_files(plenum_transcripts_dir)
            if not protocol_html_paths:
                tqdm.write(f" - No protocol HTML files found for plenum {plenum_id}. Skipping.")
                continue

            # Process transcripts
            tqdm.write(" - Processing transcripts...")
            transcript_success = process_transcripts(
                protocol_xml_path,
                protocol_html_paths,
                output_dir,
                plenum_id,
                args.force_transcript_reprocess or args.force_reprocess,
                abort_on_error=args.abort_on_error,
            )
            if not transcript_success:
                tqdm.write(f" - Failed to process transcripts for plenum {plenum_id}. Skipping.")
                continue

            # Process AV
            tqdm.write(" - Processing AV...")
            av_success, duration = process_av(
                media_path, output_dir, plenum_id, args.force_av_reprocess or args.force_reprocess
            )
            if not av_success:
                msg = f" - Failed to process AV for plenum {plenum_id}. Skipping."
                tqdm.write(msg)
                logging.warning(msg)
                continue

            # Extract plenum date from media filename
            plenum_date = extract_date_from_media_filename(media_path)
            if plenum_date:
                tqdm.write(f" - Extracted plenum date: {plenum_date}")
            else:
                tqdm.write(f" - Could not extract plenum date from filename: {media_path.name}")

            # Create metadata
            plenum_metadata = PlenumMetadata(
                source_type=source_type,
                source_id=plenum_source_id,
                source_entry_id=plenum_id,
                plenum_id=plenum_id,
                duration=round(duration, 2),
                plenum_date=plenum_date,
            )

            # Save metadata
            metadata_file = plenum_output_dir / "metadata.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                f.write(plenum_metadata.model_dump_json(indent=2))

            tqdm.write(f" - Successfully processed plenum {plenum_id}")
        except Exception as e:
            msg = f" - ERROR: Unexpected error processing plenum {plenum_id}: {e}"
            tqdm.write(msg)
            logging.warning(msg)
            if args.abort_on_error:
                raise e
            tqdm.write(f" - Skipping to next plenum")

    # After downloads complete, process normalization if not skipped
    if not args.skip_normalize:
        print("Starting normalization process...")
        normalize_plenums(
            output_dir,
            align_model=args.align_model,
            align_devices=args.align_devices,
            align_device_density=args.align_device_density,
            force_normalize_reprocess=args.force_normalize_reprocess or args.force_reprocess,
            force_rescore=args.force_rescore,
            failure_threshold=args.failure_threshold,
            plenum_ids=args.plenum_ids,
            abort_on_error=args.abort_on_error,
        )

    # Generate manifest if not skipped
    if not args.skip_manifest:
        print("Generating manifest CSV...")
        build_manifest(str(output_dir))


if __name__ == "__main__":
    import sys

    print("This module is not intended to be executed directly. Please use the top-level download.py.", file=sys.stderr)
