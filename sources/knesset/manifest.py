from sources.common.manifest import build_manifest as common_build_manifest, COMMON_COLUMNS


# Source-specific columns for knesset
KNESSET_COLUMNS = [
    "plenum_id",
    "duration",
    "plenum_date",
]


def build_manifest(input_folder: str) -> None:
    """
    Build a manifest CSV file for knesset source.
    
    Args:
        input_folder: Path to the folder containing metadata.json files
    """
    # Combine common columns with source-specific columns
    columns = COMMON_COLUMNS + KNESSET_COLUMNS
    
    # Call the common build_manifest function
    common_build_manifest(input_folder, columns)
