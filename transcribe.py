#!/usr/bin/python3

import argparse
import json
import pathlib
import asyncio
from openai import AsyncOpenAI
from openai.types.audio import Transcription

from utils import utils


def map_transcription_response_to_result(transcription: Transcription):
    """This function maps the output of the transcription server to the expected output format.
    Specifically, the whisper.cpp implementation does not provide for each segment:
    - seek
    - compression_ratio
    - no_speech_prob
    And thus, we compensate with filled 0.0 values instead.

    Args:
        transcription (Transcription): Server Transcription response

    Returns:
        dict: Segment in expected output format
    """
    return {
        "segments": [
            {
                "id": segment["id"],
                "seek": segment.get("seek") or 0.0,
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
                "avg_logprob": segment["avg_logprob"],
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
    desc_filename: pathlib.Path,
    # Batching
    num_splits: int,
    max_parallel_requests: int,
    sleep_between_batches_sec: int,
):
    # OAI client for this batch
    client = AsyncOpenAI(
        # This is the default and can be omitted
        api_key="none",
        base_url=args.server_url,
    )

    results = []
    for split_base in range(0, num_splits, max_parallel_requests):
        tasks = []
        for split_idx in range(split_base, min(split_base + max_parallel_requests, num_splits)):
            split_audio_path = desc_filename.parent / f"{split_idx}.mp3"
            if not split_audio_path.exists():
                raise Exception(f"Unable to find split {split_idx}.")

            tasks.append(fetch(client, split_audio_path, split_idx))

        responses = await asyncio.gather(*tasks)

        for r in responses:
            # Per split: id, seek, start, end, text, avg_logprob, compression_ratio, no_speech_prob
            # gather all results
            segments = []
            for ith_seg_idx, split_seg_data in enumerate(r["result"]["segments"]):
                segments.append(
                    {
                        "id": int(ith_seg_idx),
                        "seek": split_seg_data["seek"],
                        "start": split_seg_data["start"],
                        "end": split_seg_data["end"],
                        "avg_logprob": split_seg_data["avg_logprob"],
                        "compression_ratio": split_seg_data["compression_ratio"],
                        "no_speech_prob": split_seg_data["no_speech_prob"],
                        "text": split_seg_data["text"],
                    }
                )
            results.append({"split_idx": int(r["split_idx"]), "segments": segments})

        print(f"Done {split_idx + 1}/{num_splits}.")

        if sleep_between_batches_sec > 0:
            await asyncio.sleep(sleep_between_batches_sec)

    # Properly close the AIO OpenAI client
    await client.close()

    return results


async def transcribe_audio_source(desc_filename, args):
    source = desc_filename.parent.parent.name
    episode = desc_filename.parent.name
    target_dir = pathlib.Path(args.target_dir) / source / episode
    pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)

    json_fn = target_dir / f"transcripts.json"

    if not args.force_reprocess:
        try:
            desc_filename = json.load(open(json_fn, "r"))
            print("Already transcribed, skipping.")
            return
        except:
            pass

    # Open the splits description file
    splits_desc_file = json.load(open(desc_filename))

    num_splits = len(splits_desc_file["splits"])
    print(f"Total splits: {num_splits}.")

    results = await transcribe_splits(
        desc_filename,
        num_splits,
        args.max_parallel_requests,
        args.sleep_between_batches_sec,
    )

    json.dump(
        {"source": source, "episode": episode, "transcripts": results},
        open(json_fn, "w"),
        ensure_ascii=False,
    )

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
        help="Force re-transcription of all files.",
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
