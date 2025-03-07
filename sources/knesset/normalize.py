import json
import pathlib
from typing import List, Optional

from sources.common.normalize import (
    DEFAULT_ALIGN_DEVICE,
    DEFAULT_ALIGN_MODEL,
    DEFAULT_FAILURE_THRESHOLD,
    BaseNormalizer,
)
from sources.common.normalize import add_common_normalize_args as add_normalize_args
from sources.knesset.metadata import PlenumMetadata


class KnessetNormalizer(BaseNormalizer):
    """Normalizer for Knesset plenum entries."""

    def get_entry_id(self, entry_dir: pathlib.Path) -> str:
        """Get the plenum ID from the directory name."""
        return entry_dir.name

    def get_audio_file(self, entry_dir: pathlib.Path) -> pathlib.Path:
        """Get the audio file path for the plenum."""
        # Find the audio file in the plenum folder.
        # It starts with "audio" and the extension can be anything
        audio_file = next(entry_dir.glob("audio*"), None)
        if not audio_file:
            raise FileNotFoundError(f"No audio file found in {entry_dir}")
        return audio_file

    def get_language(self, metadata: PlenumMetadata) -> str:
        """Get the language for the plenum (always Hebrew for Knesset)."""
        return "he"  # Hebrew is the default language for Knesset

    def get_duration(self, metadata: PlenumMetadata) -> float:
        """Get the duration from metadata."""
        if metadata.duration is None:
            return 0.0
        return metadata.duration

    def load_metadata(self, meta_file: pathlib.Path) -> PlenumMetadata:
        """Load plenum metadata from file."""
        with open(meta_file, "r", encoding="utf-8") as f:
            return PlenumMetadata(**json.load(f))

    def save_metadata(self, meta_file: pathlib.Path, metadata: PlenumMetadata) -> None:
        """Save plenum metadata to file."""
        with open(meta_file, "w", encoding="utf-8") as f:
            f.write(metadata.model_dump_json(indent=2))


def normalize_plenums(
    input_folder: pathlib.Path,
    align_model: str = DEFAULT_ALIGN_MODEL,
    align_device: str = DEFAULT_ALIGN_DEVICE,
    force_normalize_reprocess: bool = False,
    force_rescore: bool = False,
    failure_threshold: float = DEFAULT_FAILURE_THRESHOLD,
    plenum_ids: Optional[List[str]] = None,
) -> None:
    """
    Normalize Knesset plenums.

    Args:
        input_folder: Path to the folder containing plenum directories
        align_model: Model to use for alignment
        align_device: Device to use for alignment
        force_normalize_reprocess: Whether to force reprocessing even if aligned transcript exists
        force_rescore: Whether to force recalculation of quality score
        failure_threshold: Threshold for alignment failure
        plenum_ids: Optional list of plenum IDs to process (if None, process all)
    """
    # Create normalizer
    normalizer = KnessetNormalizer(
        align_model=align_model,
        align_device=align_device,
        failure_threshold=failure_threshold,
    )

    # Normalize plenums
    normalizer.normalize_entries(
        input_folder=input_folder,
        force_reprocess=force_normalize_reprocess,
        force_rescore=force_rescore,
        entry_ids=plenum_ids,
    )


__all__ = ["normalize_sessions", "add_normalize_args"]
