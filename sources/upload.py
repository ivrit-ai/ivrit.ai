from huggingface_hub import HfApi, login
from pathlib import Path
from typing import Optional


def upload_dataset(
    folder_path: str | Path,
    target_repo_id: str,
    create_as_private: bool = True,
    hf_token: str = None,
) -> None:
    """
    Upload a local folder to Hugging Face Hub as a dataset.

    Args:
        folder_path: Path to the local folder to upload
        target_repo_id: The repository ID where to upload the dataset
        create_as_private: Whether to create the repository as private (default: True)
        commit_message: The commit message for the upload (default: "Upload dataset")
    """
    # Convert string path to Path object if necessary
    folder_path = Path(folder_path) if isinstance(folder_path, str) else folder_path

    # Validate folder path
    if not folder_path.exists():
        raise ValueError(f"Folder path does not exist: {folder_path}")
    if not folder_path.is_dir():
        raise ValueError(f"Path is not a directory: {folder_path}")

    # Initialize Hugging Face API
    api = HfApi()

    try:
        if hf_token is not None:
            login(token=hf_token)

        # Create if does not exist
        api.create_repo(target_repo_id, private=create_as_private, exist_ok=True, repo_type="dataset")
        # Upload the folder
        api.upload_large_folder(
            folder_path=str(folder_path),
            repo_id=target_repo_id,
            repo_type="dataset",
        )
    except Exception as e:
        raise Exception(f"Failed to upload dataset: {str(e)}")
