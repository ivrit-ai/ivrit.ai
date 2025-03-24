import json
import pathlib
from typing import List, Optional

from sources.common.normalize import (
    DEFAULT_ALIGN_MODEL,
    DEFAULT_FAILURE_THRESHOLD,
    BaseNormalizer,
)
from sources.common.normalize import add_common_normalize_args as add_normalize_args
from sources.common.normalize import normalize_entries
from sources.crowd_recital.metadata import SessionMetadata


class CrowdRecitalNormalizer(BaseNormalizer):
    """Normalizer for crowd recital entries."""

    def get_entry_id(self, entry_dir: pathlib.Path) -> str:
        """Get the session ID from the directory name."""
        return entry_dir.name

    def get_audio_file(self, entry_dir: pathlib.Path) -> pathlib.Path:
        """Get the audio file path for the session."""
        return entry_dir / "audio.mka"

    def get_language(self, metadata: SessionMetadata) -> str:
        """Get the language for the session."""
        doc_lang = metadata.document_language.lower()
        if doc_lang != "he":
            raise ValueError(f"Unsupported language '{doc_lang}'. Only 'he' is supported.")
        return doc_lang

    def get_duration(self, metadata: SessionMetadata) -> float:
        """Get the duration from metadata."""
        return metadata.session_duration

    def load_metadata(self, meta_file: pathlib.Path) -> SessionMetadata:
        """Load session metadata from file."""
        with open(meta_file, "r", encoding="utf-8") as f:
            return SessionMetadata(**json.load(f))

    def save_metadata(self, meta_file: pathlib.Path, metadata: SessionMetadata) -> None:
        """Save session metadata to file."""
        with open(meta_file, "w", encoding="utf-8") as f:
            f.write(metadata.model_dump_json(indent=2))


def normalize_sessions(
    input_folder: pathlib.Path,
    align_model: str = DEFAULT_ALIGN_MODEL,
    align_devices: list[str] = [],
    force_normalize_reprocess: bool = False,
    force_rescore: bool = False,
    failure_threshold: float = DEFAULT_FAILURE_THRESHOLD,
    session_ids: Optional[List[str]] = None,
) -> None:
    """
    Normalize crowd recital sessions.

    Args:
        input_folder: Path to the folder containing session directories
        align_model: Model to use for alignment
        align_devices: List of devices to use for alignment (e.g., ["cuda:0", "cuda:1"]) - this also defines the number of workers
        force_normalize_reprocess: Whether to force reprocessing even if aligned transcript exists
        force_rescore: Whether to force recalculation of quality score
        failure_threshold: Threshold for alignment failure
        session_ids: Optional list of session IDs to process (if None, process all)
    """

    # Normalize sessions
    normalize_entries(
        input_folder=input_folder,
        align_devices=align_devices,
        align_model=align_model,
        normalizer_class=CrowdRecitalNormalizer,
        failure_threshold=failure_threshold,
        force_reprocess=force_normalize_reprocess,
        force_rescore=force_rescore,
        entry_ids=session_ids,
    )


__all__ = ["normalize_sessions", "add_normalize_args"]
