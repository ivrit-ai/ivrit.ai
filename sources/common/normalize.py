import argparse
import pathlib
import sys
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Tuple

import numpy as np
import stable_whisper
import torch
from tqdm import tqdm

from utils.vtt import vtt_to_whisper_result
from sources.common.metadata import NormalizedEntryMetadata

# Common constants
DEFAULT_ALIGN_MODEL = "ivrit-ai/whisper-large-v3-turbo-ct2"
DEFAULT_ALIGN_DEVICE = "auto"
DEFAULT_FAILURE_THRESHOLD = 0.2


def calculate_quality_score(
    align_result: stable_whisper.WhisperResult, entry_id: str = ""
) -> Tuple[float, List[Dict[str, float]]]:
    """
    Calculate the quality score based on the median word probabilities from the alignment result.

    Args:
        align_result: The alignment result from stable_whisper
        entry_id: Optional identifier for logging purposes

    Returns:
        Tuple of (global_quality_score, per_segment_scores)
    """
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
        if entry_id:
            tqdm.write(f" - Failed to calculate quality score for entry {entry_id}: {e}")
        global_quality_score = 0.0
        per_segment_scores = []

    return global_quality_score, per_segment_scores


def add_common_normalize_args(parser: argparse.ArgumentParser) -> None:
    """
    Add common normalization arguments to an argument parser.

    Args:
        parser: The argument parser to add arguments to
    """
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
        "--failure-threshold",
        type=float,
        default=DEFAULT_FAILURE_THRESHOLD,
        help=f"Failure threshold for alignment - portion of segments with 0 length tolerated before failing the alignment (default: {DEFAULT_FAILURE_THRESHOLD})",
    )
    parser.add_argument(
        "--force-normalize-reprocess",
        action="store_true",
        help="Force reprocessing plenums even if transcript.aligned.vtt exists",
    )
    parser.add_argument(
        "--force-rescore",
        action="store_true",
        help="Force recalculation of quality score even if aligned transcript exists",
    )


class BaseNormalizer(ABC):
    """Base class for normalizing transcripts across different sources."""

    def __init__(
        self,
        align_model: str = DEFAULT_ALIGN_MODEL,
        align_device: str = DEFAULT_ALIGN_DEVICE,
        failure_threshold: float = DEFAULT_FAILURE_THRESHOLD,
    ):
        self.align_model = align_model
        self.align_device = align_device
        self.failure_threshold = failure_threshold
        self.model = None

    @abstractmethod
    def get_entry_id(self, entry_dir: pathlib.Path) -> str:
        """Get the entry ID from the directory name."""
        pass

    @abstractmethod
    def get_audio_file(self, entry_dir: pathlib.Path) -> pathlib.Path:
        """Get the audio file path for the entry."""
        pass

    @abstractmethod
    def get_language(self, metadata: Any) -> str:
        """Get the language for the entry."""
        pass

    @abstractmethod
    def get_duration(self, metadata: Any) -> float:
        """Get the duration from metadata."""
        pass

    @abstractmethod
    def load_metadata(self, meta_file: pathlib.Path) -> Any:
        """Load metadata from file."""
        pass

    @abstractmethod
    def save_metadata(self, meta_file: pathlib.Path, metadata: Any) -> None:
        """Save metadata to file."""
        pass

    def update_metadata_with_stats(
        self,
        metadata: NormalizedEntryMetadata,
        align_result: stable_whisper.WhisperResult,
        quality_score: float,
        per_segment_scores: List[Dict[str, float]],
    ) -> None:
        """Update plenum metadata with statistics from alignment result."""
        metadata.quality_score = quality_score
        metadata.per_segment_quality_scores = per_segment_scores

        segments = list(align_result.segments)
        segments_count = len(segments)
        words_count = 0
        total_segment_duration = 0

        for segment in segments:
            # Count words if available
            if segment.has_words and hasattr(segment, "words"):
                words_count += len(segment.words)
            total_segment_duration += segment.end - segment.start

        avg_words_per_segment = words_count / segments_count if segments_count > 0 else 0
        avg_segment_duration = total_segment_duration / segments_count if segments_count > 0 else 0
        entry_duration = self.get_duration(metadata)
        avg_words_per_minute = words_count / (entry_duration / 60) if entry_duration and entry_duration > 0 else 0

        metadata.segments_count = segments_count
        metadata.words_count = words_count
        metadata.avg_words_per_segment = round(avg_words_per_segment, 4)
        metadata.avg_segment_duration = round(avg_segment_duration, 4)
        metadata.avg_words_per_minute = round(avg_words_per_minute, 4)

    def load_model(self):
        """Load the alignment model."""
        # Resolve device: if 'auto', use cuda if available for better performance
        device = self.align_device
        if device == "auto":
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        # Load the alignment model with int8 quantization for memory efficiency
        try:
            model = stable_whisper.load_faster_whisper(self.align_model, device=device, compute_type="int8")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load alignment model: {e}")

    def normalize_entry(
        self,
        entry_dir: pathlib.Path,
        force_reprocess: bool = False,
        force_rescore: bool = False,
    ) -> bool:
        """
        Normalize a single entry.

        Args:
            entry_dir: Directory containing the entry files
            force_reprocess: Whether to force reprocessing even if aligned transcript exists
            force_rescore: Whether to force recalculation of quality score

        Returns:
            True if processing was successful, False otherwise
        """
        entry_id = self.get_entry_id(entry_dir)
        tqdm.write(f"Processing entry: {entry_id}")

        meta_file = entry_dir / "metadata.json"
        aligned_transcript_file = entry_dir / "transcript.aligned.json"

        need_align = force_reprocess or (not aligned_transcript_file.exists())
        need_rescore = force_rescore or need_align

        if not (need_align or need_rescore):
            tqdm.write(f" - Skipping entry {entry_id}: already processed")
            return True

        try:
            metadata = self.load_metadata(meta_file)
        except Exception as e:
            tqdm.write(f" - Failed to read metadata.json for entry {entry_id}: {e}")
            return False

        if need_align:
            # Get language
            language = self.get_language(metadata)

            # Get audio file
            audio_file = self.get_audio_file(entry_dir)
            transcript_vtt = entry_dir / "transcript.vtt"

            if not audio_file.exists() or not transcript_vtt.exists():
                tqdm.write(f" - Skipping entry {entry_id} because required files are missing.")
                return False

            try:
                with open(transcript_vtt, "r", encoding="utf-8") as f:
                    vtt_content = f.read()
                whisper_result = vtt_to_whisper_result(vtt_content)
            except Exception as e:
                tqdm.write(f" - Error processing transcript.vtt for entry {entry_id}: {e}")
                return False

            try:
                if self.model is None:
                    self.model = self.load_model()

                align_result: stable_whisper.WhisperResult = self.model.align(
                    str(audio_file), whisper_result, language, failure_threshold=self.failure_threshold
                )
            except Exception as e:
                tqdm.write(f" - Alignment failed for entry {entry_id}: {e}")
                return False
        else:
            try:
                align_result = stable_whisper.WhisperResult(str(aligned_transcript_file))
            except Exception as e:
                tqdm.write(f" - Failed to load aligned transcript for entry {entry_id}: {e}")
                return False

        # Calculate quality score
        quality_score, per_segment_scores = calculate_quality_score(align_result, entry_id)

        # Update metadata with statistics
        self.update_metadata_with_stats(metadata, align_result, quality_score, per_segment_scores)

        # Save updated metadata
        try:
            self.save_metadata(meta_file, metadata)
        except Exception as e:
            tqdm.write(f" - Failed to update metadata.json for entry {entry_id}: {e}")
            return False

        # Save aligned transcript if needed
        if need_align:
            try:
                align_result.save_as_json(str(aligned_transcript_file))
            except Exception as e:
                tqdm.write(f" - Failed to save aligned transcript for entry {entry_id}: {e}")
                return False

        tqdm.write(f" - Processed entry {entry_id}: quality score = {quality_score}")
        return True

    def normalize_entries(
        self,
        input_folder: pathlib.Path,
        force_reprocess: bool = False,
        force_rescore: bool = False,
        entry_ids: Optional[List[str]] = None,
    ) -> None:
        """
        Normalize multiple entries.

        Args:
            input_folder: Folder containing entry directories
            force_reprocess: Whether to force reprocessing even if aligned transcript exists
            force_rescore: Whether to force recalculation of quality score
            entry_ids: Optional list of entry IDs to process (if None, process all)
        """
        # Validate input folder exists before proceeding
        if not input_folder.exists() or not input_folder.is_dir():
            print(f"Input folder '{input_folder}' does not exist or is not a directory.", file=sys.stderr)
            return

        # Find all entry directories by looking for metadata.json files
        meta_files = list(input_folder.glob("*/metadata.json"))
        if entry_ids:
            meta_files = [mf for mf in meta_files if mf.parent.name in entry_ids]

        if not meta_files:
            print("No entry metadata.json files found in input folder.")
            return

        # Load model once for all entries
        if self.model is None:
            self.model = self.load_model()

        # Process each entry with progress tracking
        for meta_file in tqdm(meta_files, desc="Normalizing entries"):
            entry_dir = meta_file.parent
            self.normalize_entry(entry_dir, force_reprocess, force_rescore)
