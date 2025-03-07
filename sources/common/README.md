# Common Normalization Code

This directory contains common code for normalizing transcripts across different sources. The normalization process involves:

1. Loading a VTT transcript
2. Converting it to a whisper result format
3. Using stable_whisper to align with the source audio
4. Calculating quality scores
5. Updating metadata with statistics

## Structure

- `metadata.py`: Contains the `NormalizedEntryMetadata` class that extends `EntryMetadata` with common quality and statistics fields.
- `normalize.py`: Contains the `BaseNormalizer` abstract class and common normalization functions.

## Usage

### Creating a New Source Normalizer

To create a normalizer for a new source:

1. Create a new class that extends `BaseNormalizer`
2. Implement the abstract methods:
   - `get_entry_id`: Get the entry ID from the directory name
   - `get_audio_file`: Get the audio file path for the entry
   - `get_language`: Get the language for the entry
   - `get_duration`: Get the duration from metadata
   - `load_metadata`: Load metadata from file
   - `save_metadata`: Save metadata to file
   - `update_metadata_with_stats`: Update metadata with statistics from alignment result

Example:

```python
from sources.common.normalize import BaseNormalizer

class MySourceNormalizer(BaseNormalizer):
    def get_entry_id(self, entry_dir):
        return entry_dir.name
    
    def get_audio_file(self, entry_dir):
        return entry_dir / "audio.mp3"
    
    def get_language(self, metadata):
        return "he"
    
    def get_duration(self, metadata):
        return metadata.duration
    
    def load_metadata(self, meta_file):
        # Load metadata from file
        pass
    
    def save_metadata(self, meta_file, metadata):
        # Save metadata to file
        pass
    
    def update_metadata_with_stats(self, metadata, align_result, quality_score, per_segment_scores):
        # Update metadata with statistics
        pass
```

### Using the Normalizer

To use the normalizer in your source's normalize.py file:

```python
import argparse
import pathlib
from typing import List, Optional

from sources.common.normalize import add_common_normalize_args, DEFAULT_ALIGN_MODEL, DEFAULT_ALIGN_DEVICE, DEFAULT_FAILURE_THRESHOLD
from sources.my_source.normalizer import MySourceNormalizer

def add_normalize_args(parser: argparse.ArgumentParser) -> None:
    # Add common arguments
    add_common_normalize_args(parser)
    
    # Add source-specific arguments
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Force reprocessing entries even if transcript.aligned.vtt exists",
    )
    parser.add_argument(
        "--force-rescore",
        action="store_true",
        help="Force recalculation of quality score even if aligned transcript exists",
    )

def normalize_entries(
    input_folder: pathlib.Path,
    align_model: str = DEFAULT_ALIGN_MODEL,
    align_device: str = DEFAULT_ALIGN_DEVICE,
    force_reprocess: bool = False,
    force_rescore: bool = False,
    failure_threshold: float = DEFAULT_FAILURE_THRESHOLD,
    entry_ids: Optional[List[str]] = None,
) -> None:
    # Create normalizer
    normalizer = MySourceNormalizer(
        align_model=align_model,
        align_device=align_device,
        failure_threshold=failure_threshold,
    )
    
    # Normalize entries
    normalizer.normalize_entries(
        input_folder=input_folder,
        force_reprocess=force_reprocess,
        force_rescore=force_rescore,
        entry_ids=entry_ids,
    )
```
