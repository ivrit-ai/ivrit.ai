#!/usr/bin/python3

import argparse
import json
import pathlib

import asyncio
import numpy as np
from openai import AsyncOpenAI
from openai.types.audio import Transcription
import pandas as pd

from utils import utils


def map_transcription_response_to_result(transcription: Transcription):
    return {
        "segments": [
            {
                "id": segment["id"],
                "seek": segment.get("seek") or 0.0,
                "start": segment["start"] or 0.0,
                "end": segment["end"] or 0.0,
                "text": segment["text"],
                "avg_logprob": segment["avg_logprob"] or 0.0,
                "compression_ratio": segment.get("compression_ratio") or 0.0,
                "no_speech_prob": segment.get("no_speech_prob") or 0.0,
            }
            for segment in transcription.segments
        ],
    }


async def fetch(client: AsyncOpenAI, file_path: str, split_idx: int):
    audio_file = open(file_path, "rb")
    transcription = await client.audio.transcriptions.create(
        file=audio_file,
        # Open AI request params
        model="local",  # Ignored by local server
        response_format="verbose_json",  # To get avg log probs
        temperature=0.0,  # Hard coded atm - auto temp selection by server
        language="he",  # Don't rely on auto language detection - target is hebrew
    )

    return {
        "split_idx": split_idx,
        "result": map_transcription_response_to_result(transcription),
    }


async def transcribe_splits(
    run_state_df: pd.DataFrame,
    desc_filename: pathlib.Path,
    # Batching
    num_splits: int,
    max_paralle_requests: int,
    sleep_between_batches_sec: int,
    # Partial state
    checkpoint_every: int,
    partial_state_store_filename: str,
):
    # OAI client for this batch
    client = AsyncOpenAI(
        # This is the default and can be omitted
        api_key="none",
        base_url=args.server_url,
    )

    next_state_store_in = checkpoint_every
    start_at_split = run_state_df["split_idx"].max() + 1
    if np.isnan(start_at_split):
        start_at_split = 0
    else:
        print(f"Resuming from split {start_at_split + 1}.")

    for split_base in range(start_at_split, num_splits, max_paralle_requests):
        tasks = []
        for split_idx in range(split_base, min(split_base + max_paralle_requests, num_splits)):
            split_audio_path = desc_filename.parent / f"{split_idx}.mp3"
            if not split_audio_path.exists():
                raise Exception(f"Unable to find split {split_idx}.")

            tasks.append(fetch(client, split_audio_path, split_idx))

        responses = await asyncio.gather(*tasks)

        for r in responses:
            # Per split: id, seek, start, end, text, avg_logprob, compression_ratio, no_speech_prob
            # Add results to the state dataframe
            for split_seg_data in r["result"]["segments"]:
                split_idx = r["split_idx"]
                split_seg_idx = int(split_seg_data["id"])
                run_state_df.loc[len(run_state_df)] = [
                    split_idx,
                    split_seg_idx,
                    split_seg_data["seek"],
                    split_seg_data["start"],
                    split_seg_data["end"],
                    split_seg_data["text"],
                    split_seg_data["avg_logprob"],
                    split_seg_data["compression_ratio"],
                    split_seg_data["no_speech_prob"],
                ]

        print(f"Done {split_idx + 1}/{num_splits}.")

        if sleep_between_batches_sec > 0:
            await asyncio.sleep(sleep_between_batches_sec)

        next_state_store_in -= len(responses)

        if 0 == next_state_store_in:
            next_state_store_in = checkpoint_every
            run_state_df.to_parquet(partial_state_store_filename)

    if next_state_store_in < checkpoint_every:
        run_state_df.to_parquet(partial_state_store_filename)


async def transcribe_audio_source(desc_filename, args):
    source = desc_filename.parent.parent.name
    episode = desc_filename.parent.name
    target_dir = pathlib.Path(args.target_dir) / source / episode
    partial_state_store_filename = target_dir / f"partial.parquet"
    pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)

    json_fn = target_dir / f"transcripts.json"

    if not args.force_reprocess:
        try:
            desc_filename = json.load(open(json_fn, "r"))
            print("Already transcribed, skipping.")
            return
        except:
            pass
    else:
        partial_state_store_filename.unlink(missing_ok=True)

    if partial_state_store_filename.exists() and not args.ignore_partial_runs:
        run_state_df = pd.read_parquet(partial_state_store_filename)
    else:
        partial_state_store_filename.unlink(missing_ok=True)
        run_state_df = pd.DataFrame(
            {
                "split_idx": pd.Series(dtype="int"),
                "segment_idx": pd.Series(dtype="int"),
                "seek": pd.Series(dtype="float"),
                "start": pd.Series(dtype="float"),
                "end": pd.Series(dtype="float"),
                "text": pd.Series(dtype="str"),
                "avg_logprob": pd.Series(dtype="float"),
                "compression_ratio": pd.Series(dtype="float"),
                "no_speech_prob": pd.Series(dtype="float"),
            },
        )

    # Open the splits description file
    splits_desc_file = json.load(open(desc_filename))

    num_splits = len(splits_desc_file["splits"])
    print(f"Total splits: {num_splits}.")

    await transcribe_splits(
        run_state_df,
        desc_filename,
        num_splits,
        args.max_parallel_requests,
        args.sleep_between_batches_sec,
        args.checkpoint_every,
        partial_state_store_filename,
    )

    # Convert all state to dump

    results = []
    for sample_idx in np.sort(run_state_df["split_idx"].unique()):
        # Get all segments for this sample
        ith_sample_df = run_state_df[run_state_df["split_idx"] == sample_idx]
        segments = []
        for ith_seg_idx in np.sort(ith_sample_df["segment_idx"].unique()):
            segment_row = ith_sample_df[ith_sample_df["segment_idx"] == ith_seg_idx].iloc[0]
            segments.append(
                {
                    "id": int(ith_seg_idx),
                    **{
                        col: float(segment_row[col])
                        for col in [
                            "seek",
                            "start",
                            "end",
                            "avg_logprob",
                            "compression_ratio",
                            "no_speech_prob",
                        ]
                    },
                    "text": segment_row["text"],
                }
            )

        results.append({"split_idx": int(sample_idx), "segments": segments})

    json.dump(
        {"source": source, "episode": episode, "transcripts": results},
        open(json_fn, "w"),
        ensure_ascii=False,
    )

    # TODO: Decide if this is a neccesary step or keep using parquet e2e
    # Clear the state file since it's "done"
    # partial_state_store_fn.unlink()

    print("Done transcribing.")


def transcribe(args):
    # Iterate over each root directory
    descs = utils.find_files(args.root_dir, args.skip_dir, [".json"])

    if len(descs) == 0:
        print("Found no audio split folders to process (Looking for JSON files).")

    for idx, _desc in enumerate(descs):
        print(f"Transcribing episode {idx}/{len(descs)}, {_desc}.")
        desc = pathlib.Path(_desc)
        asyncio.run(transcribe_audio_source(desc, args))


if __name__ == "__main__":
    # Define an argument parser
    parser = argparse.ArgumentParser(description="Transcribe a set of audio snippets generated using process.py files.")

    # Add the arguments
    parser.add_argument(
        "--root-dir",
        action="append",
        required=True,
        help="Root directory to start search from. Can be passed multiple times.",
    )
    parser.add_argument(
        "--skip-dir",
        action="append",
        required=False,
        default=[],
        help="Directories to skip. Can be passed multiple times.",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        required=True,
        help="The directory where splitted audios will be stored.",
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Force retranscription of all files.",
    )
    parser.add_argument(
        "--ignore-partial-runs",
        action="store_true",
        help="Do not resume partial runs if found, instead overwrite and start from scratch",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        required=False,
        default=10,
        help="Store state every N processed splits.",
    )
    parser.add_argument("--server-url", type=str, required=True, help="Transcription server URL.")
    parser.add_argument(
        "--max-parallel-requests",
        type=int,
        required=False,
        default=10,
        help="Maximum number of parallel workers to use.",
    )
    parser.add_argument(
        "--sleep-between-batches-sec",
        type=int,
        required=False,
        default=0,
        help="Sleep between each batch to reduce server load",
    )

    # Parse the arguments
    args = parser.parse_args()

    transcribe(args)
