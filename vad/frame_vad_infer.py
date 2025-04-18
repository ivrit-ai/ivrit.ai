import json
import os
import shutil
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor

import torch
from nemo.collections.asr.parts.utils.vad_utils import init_frame_vad_model, prepare_manifest
from omegaconf import DictConfig

from vad.vad_io import get_frame_vad_probs_filename
from vad.definitions import SPEECH_PROB_FRAME_DURATION
from vad.nemo_patched_logic import generate_vad_frame_pred
from utils.audio import transcode_to_mono_16k

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_output_file_path(out_dir: str, source: str, episode: str):
    return get_frame_vad_probs_filename(out_dir, source, episode)


def parallel_audio_file_adapt(from_to_tuples: list[tuple[str, str]], max_parallel_workers=1):
    with ThreadPoolExecutor(max_workers=max_parallel_workers) as executor:
        executor.map(lambda fromto: transcode_to_mono_16k(fromto[0], fromto[1]), from_to_tuples)


def exclude_already_predicted(audio_files: list[str], final_output_fir: str):
    pruned_audio_files = []
    for audio_file in audio_files:
        source = Path(audio_file).parent.name
        episode = Path(audio_file).stem
        target_file_name = get_output_file_path(final_output_fir, source, episode)
        if not os.path.exists(target_file_name):
            pruned_audio_files.append(audio_file)
    return pruned_audio_files


def generate_frame_vad_predictions(audio_files: list[str], final_output_fir: str, config: dict = {}) -> None:
    temp_processing_dir = os.path.join(final_output_fir, "vad_temp_processing")
    temp_input_source_dir = os.path.join(temp_processing_dir, "audio_files")

    cfg = DictConfig(
        {
            "input_manifest": os.path.join(temp_processing_dir, "input_manifest.json"),
            "output_dir": os.path.join(temp_processing_dir, "out"),
            "split_duration": config.get(
                "nemo_vad_presplit_duration", 400
            ),  # will work on most GPUs, no need to change
            "sample_rate": 16000,
            "num_workers": 0,  # with >0 the pickling of the nemo classes fail - keep at 0
            "evaluate": False,
            "model_path": "vad_multilingual_frame_marblenet",
            "shift_length_in_sec": SPEECH_PROB_FRAME_DURATION,  # Can't change this - this model was trained on this frame size
        }
    )

    # prune files which already have prediction from previous runs
    # unless configured to reprocess
    if not config["force_reprocess"]:
        audio_files = exclude_already_predicted(audio_files, final_output_fir)

    if len(audio_files) == 0:
        logging.warning("No new audio files to process. Exiting.")
        return

    # Create it if needed - clear if it exists
    if os.path.exists(temp_processing_dir):
        shutil.rmtree(temp_processing_dir)
    os.makedirs(temp_input_source_dir)

    # Create the input manifest from the audio files list
    # Need to adapt the incoming file path structure (source/episode) to a unified
    # filename source__episode - since NeMo tooling looks only at the filename and
    # also requires it to be unique.
    # additionally the dataset loader would choke on non-mono input files.
    # so we will:
    # transcode to mono 16k wav since transcoding is done, at least we can offload this too ffmpeg
    # in one go

    input_file_to_flat_input_file_map = {
        input_file: f"{Path(input_file).parent.name}__{Path(input_file).stem}.wav" for input_file in audio_files
    }
    flat_input_file_map_to_input_file = {v: k for k, v in input_file_to_flat_input_file_map.items()}

    # Create optimized audio inputs files in the tmp folder from the original input files
    # Use the flat file names - these are the files nemo will process
    temp_input_file_list_relative = []
    audio_files_to_adapt_from_to_tuples = []
    for input_file, flat_input_file in input_file_to_flat_input_file_map.items():
        flat_input_file = os.path.join(temp_input_source_dir, flat_input_file)
        temp_input_file_list_relative.append(flat_input_file)
        input_file = os.path.abspath(input_file)
        flat_input_file = os.path.abspath(flat_input_file)
        audio_files_to_adapt_from_to_tuples.append((input_file, flat_input_file))

    print(f"pre-trasncoding {len(audio_files_to_adapt_from_to_tuples)} audio files")
    parallel_audio_file_adapt(
        audio_files_to_adapt_from_to_tuples, max_parallel_workers=config["nemo_vad_pretranscode_workers"]
    )

    input_manifest_audio_entries = [{"audio_filepath": af} for af in temp_input_file_list_relative]
    with open(cfg.input_manifest, "w", encoding="utf-8") as fout:
        for entry in input_manifest_audio_entries:
            json.dump(entry, fout)
            fout.write("\n")
            fout.flush()

    # Create the output dir - where vad results and intermediate steps output will be stored
    output_dir = cfg.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cfg.frame_out_dir = os.path.join(output_dir, "frame_preds")

    # Split inputs to segments that will fit in cuda memory
    logging.info("Split long audio file to avoid CUDA memory issue")
    logging.debug("Try smaller split_duration if you still have CUDA memory issue")
    segment_prepare_config = {
        "input": cfg.input_manifest,
        "window_length_in_sec": 0,
        "split_duration": cfg.split_duration,
        # Could be increased if CPU cores are available - each worker
        # loads the audio file, and determines the duration to be able to
        # decide on offset+duration to present to the vad model
        # each file length is served as an antire batch to the GPU
        # but they run serially on a single worker
        "num_workers": config.get("nemo_vad_presplit_workers", 1),
        "prepared_manifest_vad_input": os.path.join(output_dir, "segmented_input_manifest.json"),
        "out_dir": output_dir,
    }
    manifest_vad_input = prepare_manifest(segment_prepare_config)

    torch.set_grad_enabled(False)
    vad_model = init_frame_vad_model(cfg.model_path)

    # setup_test_data
    vad_model.setup_test_data(
        test_data_config={
            "batch_size": 1,
            "sample_rate": 16000,
            "manifest_filepath": manifest_vad_input,
            "labels": ["infer"],
            "num_workers": cfg.num_workers,
            "shuffle": False,
            "normalize_audio_db": None,
        }
    )

    vad_model = vad_model.to(device)
    vad_model.eval()

    os.mkdir(cfg.frame_out_dir)

    logging.info("Generating frame-level prediction ")
    generate_vad_frame_pred(
        vad_model=vad_model,
        window_length_in_sec=0,
        shift_length_in_sec=cfg.shift_length_in_sec,
        manifest_vad_input=manifest_vad_input,
        out_dir=cfg.frame_out_dir,
    )
    logging.info(f"Finish generating VAD frame level prediction.")

    # Go over each input audio file - and copy the frame level predictions output
    # from the temp output folder to the target output folder
    frame_level_temp_result_file_names = list(Path(cfg.frame_out_dir).glob("*.frame"))
    logging.info(f"Copying results to output folder")
    for temp_frame_result_file_name in frame_level_temp_result_file_names:
        base_file_name = temp_frame_result_file_name.stem  # without .frame
        # If there is not "stem" before the frame - NeMo stipped the "wav" suffix
        # (which is a very non general way of handling datasets, so we have to accommodate it)
        if base_file_name == Path(base_file_name).stem:
            base_file_name += ".wav"

        # recover the original input audio filew name + path
        input_audio_file = flat_input_file_map_to_input_file[base_file_name]

        # get source + episode parts from the path - into final results output file
        source = Path(input_audio_file).parent.name
        episode = Path(input_audio_file).stem
        target_file_name = get_output_file_path(final_output_fir, source, episode)

        # ensure output dir exists
        Path(target_file_name).parent.mkdir(parents=True, exist_ok=True)

        # move the frames result file to final target
        # Ensure the target dir exists
        Path(target_file_name).parent.mkdir(exist_ok=True, parents=True)
        os.replace(temp_frame_result_file_name, target_file_name)

    logging.info(f"Removing temporary processing directory")
    # Remove the temp processing folder
    shutil.rmtree(temp_processing_dir)

    logging.info("Done!")
