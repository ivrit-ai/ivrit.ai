#!/usr/bin/python3

import argparse
import json

import datasets


def initialize_dataset(args, transcripts, split):
    uuids = []
    sources = []
    orig_texts = []
    texts = []
    stats = []

    print(f"Collecting {split} entries...")
    for t in transcripts:
        payload = t["data"]["payload"]

        if payload["skipped"]:
            continue

        if "attributes" in payload and "retranscribe" in payload["attributes"]:
            continue

        if t["source"] == "IdanEretzYoutube":
            continue

        is_test_source = t["source"] in ["Mindset", "SmallBigHistory"]
        if (split == "test") != is_test_source:
            continue

        uuid = t["data"]["uuid"]
        stat = payload.get("stats", {})

        uuids.append(uuid)
        sources.append(f"{args.audio_dir}/{uuid}.mp3")
        texts.append(payload["text"])
        orig_texts.append(payload["orig_text"])
        stats.append(stat)

    print("Preparing dataset...")
    ds = datasets.Dataset.from_dict(
        {
            "uuid": uuids,
            "audio": sources,
            "orig_text": orig_texts,
            "text": texts,
            "stats": stats,
        }
    )

    print("Loading audio files...")
    ds = ds.cast_column("audio", datasets.Audio())

    return ds


if __name__ == "__main__":
    # Define an argument parser
    parser = argparse.ArgumentParser(description="Create a training dataset and upload to Huggingface.")

    # Add the arguments
    parser.add_argument("--transcripts", type=str, required=True, help="Transcripts DB as a JSON file.")
    parser.add_argument("--audio-dir", type=str, required=True, help="Root audio directory.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of dataset to upload.")
    parser.add_argument("--branch", type=str, required=True, help="Branch to push to.")
    parser.add_argument("--hf-token", type=str, required=False, help="The HuggingFace token.")

    # Parse the arguments
    args = parser.parse_args()

    transcripts = json.load(open(args.transcripts, "r"))

    train_ds = initialize_dataset(args, transcripts, "train")
    test_ds = initialize_dataset(args, transcripts, "test")

    ds = datasets.DatasetDict({"train": train_ds, "test": test_ds})

    ds.push_to_hub(repo_id=args.dataset, token=args.hf_token)
