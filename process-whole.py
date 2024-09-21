#!/usr/bin/python3

import argparse
import json
import pathlib
import asyncio
from typing import List, Dict
from openai import AsyncOpenAI
from openai.types.audio import Transcription

from vbs import load_and_split, save_segments
from utils.utils import find_files

def map_transcription_response_to_result(transcription: Transcription):
    return {
        "segments": [
            {
                "id": segment["id"],
                "seek": segment.get("seek", 0.0),
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
                "avg_logprob": segment["avg_logprob"],
                "compression_ratio": segment.get("compression_ratio", 0.0),
                "no_speech_prob": segment.get("no_speech_prob", 0.0),
            }
            for segment in transcription.segments
        ],
    }

async def fetch(client: AsyncOpenAI, file_path: str, split_idx: int):
    with open(file_path, "rb") as audio_file:
        transcription = await client.audio.transcriptions.create(
            file=audio_file,
            model="local",
            response_format="verbose_json",
            temperature=0.0,
            language="he",
        )

    return {
        "split_idx": split_idx,
        "result": map_transcription_response_to_result(transcription),
    }

async def transcribe_splits(
    output_dir: pathlib.Path,
    num_splits: int,
    max_parallel_requests: int,
    sleep_between_batches_sec: int,
    server_url: str,
):
    client = AsyncOpenAI(api_key="none", base_url=server_url)

    results = []
    for split_base in range(0, num_splits, max_parallel_requests):
        tasks = []
        for split_idx in range(split_base, min(split_base + max_parallel_requests, num_splits)):
            split_audio_path = output_dir / f"{split_idx}.mp3"
            if not split_audio_path.exists():
                raise Exception(f"Unable to find split {split_idx}.")

            tasks.append(fetch(client, str(split_audio_path), split_idx))

        responses = await asyncio.gather(*tasks)

        for r in responses:
            segments = [
                {
                    "id": int(idx),
                    "seek": seg["seek"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "avg_logprob": seg["avg_logprob"],
                    "compression_ratio": seg["compression_ratio"],
                    "no_speech_prob": seg["no_speech_prob"],
                    "text": seg["text"],
                }
                for idx, seg in enumerate(r["result"]["segments"])
            ]
            results.append({"split_idx": int(r["split_idx"]), "segments": segments})

        print(f"Done {split_base + len(tasks)}/{num_splits}.")

        if sleep_between_batches_sec > 0:
            await asyncio.sleep(sleep_between_batches_sec)

    await client.close()
    return results

async def process_audio_file(input_file: pathlib.Path, output_dir: pathlib.Path, args):
    relative_path = input_file.relative_to(pathlib.Path(args.root_dir[0]))
    target_dir = output_dir / relative_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    desc_file = target_dir / "desc.json"

    if not args.force_reprocess and desc_file.exists():
        try:
            with open(desc_file, 'r') as f:
                json.load(f)
            print(f"Skipping {input_file} as desc.json already exists and is valid.")
            return
        except json.JSONDecodeError:
            print(f"Invalid desc.json found for {input_file}. Reprocessing.")

    # Step 2: Perform volume-based splitting
    print(f"Splitting {input_file}")
    segments = load_and_split(str(input_file))
    save_segments(str(input_file), segments, target_dir)

    # Save desc.json with split information
    with open(desc_file, 'w') as f:
        json.dump({"splits": [{"start": s[0], "end": s[1]} for s in segments]}, f, indent=2)

    # Step 3 & 4: Perform transcription and save results
    print(f"Transcribing splits for {input_file}")
    num_splits = len(segments)
    results = await transcribe_splits(
        target_dir,
        num_splits,
        args.max_parallel_requests,
        args.sleep_between_batches_sec,
        args.server_url,
    )

    # Combine split information with transcription results
    combined_results = []
    for split, result in zip(segments, results):
        combined_results.append({
            "start": split[0],
            "end": split[1],
            "transcription": result
        })

    # Save combined results
    with open(target_dir / "transcripts.json", 'w', encoding='utf-8') as f:
        json.dump(
            {
                "source": str(input_file),
                "splits": combined_results
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Done processing {input_file}")

async def process_directory(args):
    root_dir = pathlib.Path(args.root_dir[0])
    target_dir = pathlib.Path(args.target_dir)

    audio_files = find_files(args.root_dir, args.skip_dir, ['.mp3', '.wav', '.flac'])

    for audio_file in audio_files:
        await process_audio_file(pathlib.Path(audio_file), target_dir, args)

def main():
    parser = argparse.ArgumentParser(description="Process and transcribe audio files")
    parser.add_argument("--root-dir", action="append", required=True, help="Root directory containing audio files")
    parser.add_argument("--skip-dir", action="append", default=[], help="Directories to skip")
    parser.add_argument("--target-dir", required=True, help="Target directory for output")
    parser.add_argument("--force-reprocess", action="store_true", help="Force re-processing of all files")
    parser.add_argument("--server-url", required=True, help="Transcription server URL")
    parser.add_argument("--max-parallel-requests", type=int, default=10, help="Maximum number of parallel requests")
    parser.add_argument("--sleep-between-batches-sec", type=int, default=0, help="Sleep between batches in seconds")
    args = parser.parse_args()

    asyncio.run(process_directory(args))

if __name__ == "__main__":
    main()
