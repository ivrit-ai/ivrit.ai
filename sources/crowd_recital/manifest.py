from sources.common.manifest import build_manifest as common_build_manifest, COMMON_COLUMNS


# Source-specific columns for crowd_recital
CROWD_RECITAL_COLUMNS = [
    "session_id",
    "session_duration",
    "user_id",
    "document_language",
    "document_title",
    "document_source_type",
    "year_of_birth",
    "biological_sex",
]


def build_manifest(input_folder: str) -> None:
    """
    Build a manifest CSV file for crowd_recital source.
    
    Args:
        input_folder: Path to the folder containing metadata.json files
    """
    # Combine common columns with source-specific columns
    columns = COMMON_COLUMNS + CROWD_RECITAL_COLUMNS
    
    # Call the common build_manifest function
    common_build_manifest(input_folder, columns)
