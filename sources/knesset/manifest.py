import json
import pathlib

import pandas as pd


def build_manifest(input_folder: str) -> None:
    base_path = pathlib.Path(input_folder)
    # Recursively find all metadata.json files
    metadata_files = list(base_path.glob("**/metadata.json"))
    entries = []
    for meta_file in metadata_files:
        try:
            with meta_file.open("r") as f:
                data = json.load(f)
            # Compute min_segment_quality_score from per_segment_quality_scores if available
            per_segment_quality_scores = data.get("per_segment_quality_scores")
            if isinstance(per_segment_quality_scores, list) and per_segment_quality_scores:
                data["min_segment_quality_score"] = min(score["probability"] for score in per_segment_quality_scores)
            else:
                data["min_segment_quality_score"] = None
            entries.append(data)
        except Exception as e:
            print(f"Error reading {meta_file}: {e}")

    if not entries:
        print("No metadata files found to build manifest.")
        return

    columns = [
        "source_type",
        "source_id",
        "source_entry_id",
        "plenum_id",
        "duration",
        "quality_score",
        "min_segment_quality_score",
        "avg_words_per_segment",
        "avg_segment_duration",
        "avg_words_per_minute",
        "segments_count",
        "words_count",
    ]

    df = pd.DataFrame(entries)
    # Ensure all required columns exist
    for col in columns:
        if col not in df.columns:
            df[col] = None
    df = df[columns]

    manifest_path = base_path / "manifest.csv"
    df.to_csv(manifest_path, index=False)
    print(f"Manifest built and saved to {manifest_path}")
