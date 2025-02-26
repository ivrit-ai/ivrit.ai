#!/usr/bin/python3

import argparse
import json
import pathlib

import datasets
from utils import utils

import mutagen.mp3
from sources.upload import upload_dataset


def create_transcripts_dataset(args):
    files = utils.find_files(args.root_dir, args.skip_dir, [".json"])

    ds = datasets.Dataset.from_dict({})

    ds_list = []

    for file_idx, file in enumerate(files):
        print(f"{file_idx}/{len(files)}")
        t_desc = json.load(open(file))

        source = t_desc["source"]
        episode = t_desc["episode"]

        sources = []
        episodes = []
        uuids = []
        texts = []
        attrs = []

        for seg_idx, seg in enumerate(t_desc["transcripts"]):
            uuid = f"{source}/{episode}/{seg_idx}"

            seg_texts = []
            for sub_segment in seg["segments"]:
                seg_texts.append(sub_segment["text"])

            text = " ".join(seg_texts)

            sources.append(source)
            episodes.append(episode)
            uuids.append(uuid)
            texts.append(text)
            attrs.append(seg)

        temp_ds = datasets.Dataset.from_dict(
            {"source": sources, "episode": episodes, "uuid": uuids, "text": texts, "attrs": attrs}
        )
        ds_list.append(temp_ds)

    ds = datasets.concatenate_datasets(ds_list)

    return ds


def create_eval_dataset(args):
    files = utils.find_files(args.root_dir, args.skip_dir, [".mp3"])

    uuids = []
    texts = []

    for file in files:
        file = pathlib.Path(file)

        episode = file.parent.name
        source = file.parent.parent.name
        segment = int(file.stem)

        uuid = f"{source}/{episode}/{segment}"

        text = file.with_suffix(".txt").read_text()

        uuids.append(uuid)
        texts.append(text)

    ds = datasets.Dataset.from_dict({"audio": files, "uuid": uuids, "text": texts})
    ds = ds.cast_column("audio", datasets.Audio())

    ds = datasets.DatasetDict({"test": ds})

    return ds


def create_audio_dataset(args):
    files = utils.find_files(args.root_dir, args.skip_dir, [".mp3", ".mp4"])

    total_duration = 0

    episodes = []
    sources = []
    uuids = []
    attrs = []

    cached_desc = None
    cached_uuid = None

    for idx, file in enumerate(files):
        print(f"{idx}/{len(files)}")

        file = pathlib.Path(file)

        if args.splits:
            episode = file.parent.name
            source = file.parent.parent.name
            segment = int(file.stem)
            uuid = f"{source}/{episode}/{segment}"

            if not cached_uuid == uuid:
                cached_desc = json.load(open(file.parent / "desc.json"))
                cached_uuid = uuid

            start = cached_desc["splits"][segment][0]
            end = cached_desc["splits"][segment][1]
            duration = end - start
        else:
            episode = file.stem
            source = file.parent.name
            uuid = f"{source}/{episode}"
            duration = mutagen.mp3.MP3(file).info.length

        total_duration += duration

        attr = {"license": "v1", "duration": duration}
        if args.splits:
            attr["segment"] = segment
            attr["start"] = start
            attr["end"] = end

        episodes.append(episode)
        sources.append(source)
        uuids.append(uuid)
        attrs.append(attr)

    print(f"Total dataset size is {total_duration:.2f} seconds.")

    ds = datasets.Dataset.from_dict(
        {"audio": files, "episode": episodes, "source": sources, "uuid": uuids, "attrs": attrs}
    )

    ds = ds.cast_column("audio", datasets.Audio())

    return ds


def upload_normalized_dataset(args):
    if args.skip_dir:
        raise NotImplementedError("Skipping directories not supported for normalized datasets")
    if len(args.root_dir) != 1:
        raise ValueError("Only one root directory is supported for normalized datasets")

    upload_dataset(
        args.root_dir[0],
        args.dataset,
        create_as_private=not args.create_as_public,
        hf_token=args.hf_token,
    )


def initialize_dataset(args):
    # Iterate over each root directory
    if args.transcripts:
        ds = create_transcripts_dataset(args)
    elif args.eval:
        ds = create_eval_dataset(args)
    elif args.splits:
        ds = create_audio_dataset(args)
    else:
        # upload the folder - not as a "Dataset" object, simply
        # list of files to maintain flexibility in content structure
        upload_normalized_dataset(args)
        # No further processing required
        return

    # For now, uploading a dataset of >200GB tends to fail, so we generate a local DB
    # and upload it using git lfs.
    #
    # Yair, July 2023

    print("Uploading repo...")
    ds.push_to_hub(repo_id=args.dataset, token=args.hf_token)
    # ds.save_to_disk(args.dataset, num_shards=2000)


if __name__ == "__main__":
    # Define an argument parser
    parser = argparse.ArgumentParser(description="Create a dataset and upload to Huggingface.")

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
        "--create_as_public", action="store_true", help="Create as a public dataset if it does not exist."
    )
    parser.add_argument("--dataset", type=str, required=True, help="The dataset name to upload.")
    parser.add_argument("--hf-token", type=str, required=False, help="The HuggingFace token.")
    parser.add_argument("--splits", action="store_true", help="This is a splits dataset.")
    parser.add_argument("--transcripts", action="store_true", help="This is a transcripts dataset.")
    parser.add_argument("--eval", action="store_true", help="This is a evaluation dataset.")
    parser.add_argument("--normalized", action="store_true", help="This is a normalized source dataset.")

    # Parse the arguments
    args = parser.parse_args()

    initialize_dataset(args)
