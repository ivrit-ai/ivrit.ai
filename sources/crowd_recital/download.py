import argparse
import json
import pathlib
from dataclasses import asdict

import boto3
import psycopg
from botocore.exceptions import ClientError
from psycopg.rows import dict_row
from tqdm import tqdm

from sources.crowd_recital.metadata import SessionMetadata, source_id, source_type

from .normalize import add_normalize_args, normalize_sessions


def load_download_state(state_file: pathlib.Path) -> dict:
    if state_file.exists():
        with open(state_file, "r") as f:
            return json.load(f)
    return {}


def save_download_state(state: dict, state_file: pathlib.Path) -> None:
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)


def list_recording_sessions(s3_client, bucket: str, prefix: str = "") -> list:
    """
    List sub-folders (recording sessions) in the bucket under the given prefix.
    Each CommonPrefix represents a session id. Returns sessions sorted by creation date.
    """
    params = {"Bucket": bucket}
    if prefix:
        params["Prefix"] = prefix.rstrip("/") + "/"

    # Get all objects to check creation dates
    all_objects = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(**params):
        if "Contents" in page:
            all_objects.extend(page["Contents"])

    # Group objects by session and get earliest creation date for each
    session_dates = {}
    for obj in all_objects:
        key = obj["Key"]
        parts = key.rstrip("/").split("/")
        if len(parts) > 1:  # Ensure we have a session ID
            session_id = parts[-2]  # Session ID is second to last part
            created = obj["LastModified"]
            if session_id not in session_dates or created < session_dates[session_id]:
                session_dates[session_id] = created

    # Sort sessions by creation date
    sessions = sorted(session_dates.keys(), key=lambda x: session_dates[x])
    return sessions


def download_file(s3_client, bucket: str, key: str, destination: pathlib.Path) -> None:
    try:
        s3_client.download_file(bucket, key, str(destination))
        print(f"Downloaded {key} to {destination}")
    except ClientError as e:
        print(f"Error downloading {key}: {e}")


def download_session_data_files(
    s3_client, bucket: str, prefix: str, session_id: str, output_dir: pathlib.Path, force_download: bool
) -> None:
    session_output_dir = output_dir / session_id
    session_output_dir.mkdir(parents=True, exist_ok=True)
    # Expected files in each session folder
    files_to_download = [("transcript.vtt", "transcript.vtt"), ("main.audio.mka", "audio.mka")]
    for remote_name, local_name in files_to_download:
        local_file = session_output_dir / local_name
        if local_file.exists() and not force_download:
            print(f"File {local_file} already exists. Skipping.")
            continue
        # Construct the S3 key: if prefix provided, then prefix/session_id/file_name
        if prefix:
            s3_key = f"{prefix.rstrip('/')}/{session_id}/{remote_name}"
        else:
            s3_key = f"{session_id}/{remote_name}"
        download_file(s3_client, bucket, s3_key, local_file)


def load_uploaded_sessions_from_db(connection_string: str, languages: list[str] | None = None) -> list:
    """
    Connects to the PostgreSQL database using the provided connection string and
    loads metadata for uploaded sessions from the database. Returns a list of dictionaries with keys:
    session_id, session_duration, user_id, document_language, document_title, document_source_type,
    year_of_birth, and biological_sex.

    Args:
        connection_string: PostgreSQL connection string
        languages: Optional list of language codes to filter sessions by
    """
    query = """
    SELECT r.id AS session_id,
           r.duration AS session_duration,
           r.user_id,
           td.lang AS document_language,
           td.title AS document_title,
           td.source_type AS document_source_type,
           um.year_of_birth,
           um.biological_sex
    FROM public.recital_sessions r
    LEFT JOIN public.text_documents td ON r.document_id = td.id
    LEFT JOIN public.users_metadata um ON r.user_id = um.id
    WHERE r.status = 'uploaded'
    """
    if languages:
        query += " AND td.lang = ANY(%s)"
    query += " ORDER BY r.created_at ASC;"

    try:
        conn = psycopg.connect(connection_string)
    except Exception as e:
        print(f"Error connecting to PostgreSQL: {e}")
        return []
    try:
        with conn.cursor(row_factory=dict_row) as cur:
            if languages:
                cur.execute(query, (languages,))
            else:
                cur.execute(query)
            sessions = cur.fetchall()
            return [dict(row) for row in sessions]
    except Exception as e:
        print(f"Error loading sessions from DB: {e}")
        return []
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download recording sessions from an S3 bucket and maintain download state."
    )
    parser.add_argument("--s3-bucket", type=str, required=True, help="S3 bucket name where recordings are stored.")
    parser.add_argument(
        "--s3-prefix", type=str, default="", help="Optional prefix in the S3 bucket where session folders are located."
    )
    parser.add_argument(
        "--aws-access-key", type=str, help="AWS access key id. If not provided, env credentials will be used."
    )
    parser.add_argument(
        "--aws-secret-key", type=str, help="AWS secret access key. If not provided, env credentials will be used."
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Local output directory where sessions will be downloaded."
    )
    parser.add_argument("--force-download", action="store_true", help="Force re-download of files even if they exist.")
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=None,
        help="Maximum number of sessions to process from the database (applied before checking download state).",
    )
    parser.add_argument(
        "--pg-connection-string",
        type=str,
        required=True,
        help="PostgreSQL connection string including the database name.",
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=["he"],
        help="Language codes to filter sessions by. Default is 'he'. Multiple languages can be specified.",
    )
    parser.add_argument("--skip-normalize", action="store_true", help="Skip normalization process")
    parser.add_argument(
        "--skip-download", action="store_true", help="Skip downloading new sessions and only perform normalization."
    )

    # Add normalization-related arguments (input-folder is not needed since we use output-dir as the folder to normalize)
    add_normalize_args(parser)

    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_file = output_dir / "download_state.json"
    download_state = load_download_state(state_file)

    # Load sessions from PostgreSQL database
    sessions = load_uploaded_sessions_from_db(args.pg_connection_string, languages=args.languages)
    if args.max_sessions is not None:
        sessions = sessions[: args.max_sessions]
    if not sessions:
        print("No recording sessions found in the database.")
        return

    if args.skip_download:
        print("Skipping session downloads (--skip-download enabled).")
    else:
        # Setup the S3 client using provided AWS credentials or default env
        if args.aws_access_key and args.aws_secret_key:
            s3_client = boto3.client(
                "s3", aws_access_key_id=args.aws_access_key, aws_secret_access_key=args.aws_secret_key
            )
        else:
            s3_client = boto3.client("s3")

        for session in tqdm(sessions, desc="Downloading sessions"):
            tqdm.write(
                f"Processing session {session.get('session_id')} | Title: {session.get('document_title')} | Lang: {session.get('document_language')} | Duration: {session.get('session_duration')} | User: {session.get('user_id')} | YOB: {session.get('year_of_birth')} | Sex: {session.get('biological_sex')}"
            )
            if not args.force_download and download_state.get(session.get("session_id"), False):
                continue
            download_session_data_files(
                s3_client, args.s3_bucket, args.s3_prefix, session.get("session_id"), output_dir, args.force_download
            )
            # Write session metadata to metadata.json in the session output directory
            session_output_dir = output_dir / session.get("session_id")
            metadata_file = session_output_dir / "metadata.json"
            session_id = str(session.get("session_id"))
            session_metadata_instance = SessionMetadata(
                source_type=source_type,
                source_id=source_id,
                source_entry_id=session_id,
                session_id=session_id,
                session_duration=float(session.get("session_duration")),
                user_id=str(session.get("user_id")),
                document_language=str(session.get("document_language")),
                document_title=str(session.get("document_title")),
                document_source_type=str(session.get("document_source_type")),
                year_of_birth=int(session.get("year_of_birth")) if session.get("year_of_birth") is not None else None,
                biological_sex=session.get("biological_sex"),
                quality_score=None,  # Possibly determined later during alignment
            )
            with open(metadata_file, "w") as f:
                json.dump(asdict(session_metadata_instance), f, indent=2)
            # Mark session as downloaded and update state file
            download_state[session.get("session_id")] = True
            save_download_state(download_state, state_file)

    # After downloads complete, process normalization if not skipped
    if not args.skip_normalize:
        print("Starting normalization process...")
        normalize_sessions(
            output_dir,
            align_model=args.align_model,
            align_device=args.align_device,
            force_reprocess=args.force_reprocess,
            failure_threshold=args.failure_threshold,
        )


if __name__ == "__main__":
    import sys

    print("This module is not intended to be executed directly. Please use the top-level download.py.", file=sys.stderr)
