import argparse
import pathlib
import re
from typing import List, Optional, Tuple

from tqdm import tqdm

from sources.knesset.extraction import process_transcripts
from sources.knesset.manifest import build_manifest
from sources.knesset.metadata import PlenumMetadata, plenum_source_id, source_type
from sources.knesset.normalize import add_normalize_args, normalize_plenums
from utils.audio import extract_audio_from_media, get_audio_info


def find_media_file(plenum_dir: pathlib.Path) -> Optional[pathlib.Path]:
    """Find the media file (mp3, mp4, or wmv) in the plenum directory."""

    found_media_files = []
    for ext in [".mp3", ".mp4", ".wmv", "avi"]:
        for file in plenum_dir.glob(f"*{ext}"):
            found_media_files.append(file)

    if not found_media_files:
        return None

    # expect only one
    assert len(found_media_files) == 1
    return found_media_files[0]


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
        action="append",
        default=[],
        help="Filter to process only the specified plenum ids (can be specified multiple times)",
    )

    # Add normalization-related arguments
    add_normalize_args(parser)

    args = parser.parse_args()

    input_media_dir = pathlib.Path(args.input_media_dir)
    input_transcripts_dir = pathlib.Path(args.input_transcripts_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    if not plenum_ids:
        print("No plenum IDs found in the input media directory.")
        return

    print(f"Found {len(plenum_ids)} plenum IDs.")

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
                tqdm.write(f" - Failed to process AV for plenum {plenum_id}. Skipping.")
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
                duration=duration,
                plenum_date=plenum_date,
            )

            # Save metadata
            metadata_file = plenum_output_dir / "metadata.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                f.write(plenum_metadata.model_dump_json(indent=2))

            tqdm.write(f" - Successfully processed plenum {plenum_id}")
        except Exception as e:
            tqdm.write(f" - ERROR: Unexpected error processing plenum {plenum_id}: {e}")
            tqdm.write(f" - Skipping to next plenum")

    # After downloads complete, process normalization if not skipped
    if not args.skip_normalize:
        print("Starting normalization process...")
        normalize_plenums(
            output_dir,
            align_model=args.align_model,
            align_devices=args.align_devices,
            force_normalize_reprocess=args.force_normalize_reprocess or args.force_reprocess,
            force_rescore=args.force_rescore,
            failure_threshold=args.failure_threshold,
            plenum_ids=args.plenum_ids,
        )

    # Generate manifest if not skipped
    if not args.skip_manifest:
        print("Generating manifest CSV...")
        build_manifest(str(output_dir))


if __name__ == "__main__":
    import sys

    print("This module is not intended to be executed directly. Please use the top-level download.py.", file=sys.stderr)
