import json
import os
import shutil
from pathlib import Path

import torch
from nemo.collections.asr.parts.utils.vad_utils import generate_vad_frame_pred, init_frame_vad_model, prepare_manifest
from nemo.utils import logging
from omegaconf import DictConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_frame_vad_predictions(audio_files: list[str], final_output_fir: str, config: dict = {}) -> None:
    temp_processing_dir = os.path.join(final_output_fir, "vad_temp_processing")

    # Create it if needed - clear if it exists
    if os.path.exists(temp_processing_dir):
        shutil.rmtree(temp_processing_dir)
    temp_input_source_dir = os.path.join(temp_processing_dir, "audio_files")
    os.makedirs(temp_input_source_dir)

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
            "shift_length_in_sec": 0.02,  # Can't change this - this model was trained on this frame size
        }
    )

    # Create the input manifest from the audio files list
    # Need to adapt the incoming file path structure (source/episode) to a unified
    # filename source__episode - since NeMo tooling looks only at the filename and
    # also requires it to be unique.

    input_file_to_flat_input_file_map = {
        input_file: f"{Path(input_file).parent.name}__{Path(input_file).name}" for input_file in audio_files
    }
    flat_input_file_map_to_input_file = {v: k for k, v in input_file_to_flat_input_file_map.items()}

    # Create synlinks in the tmp folder from the original input files
    # to the flat file names - these are the files nemo will process
    temp_input_file_list_relative = []
    for input_file, flat_input_file in input_file_to_flat_input_file_map.items():
        flat_input_file = os.path.join(temp_input_source_dir, flat_input_file)
        temp_input_file_list_relative.append(flat_input_file)
        input_file = os.path.abspath(input_file)
        flat_input_file = os.path.abspath(flat_input_file)
        os.symlink(input_file, flat_input_file)

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
            "normalize_audio_db": False,
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
        target_file_name = os.path.join(final_output_fir, source, episode + ".speech_probs")

        # ensure output dir exists
        Path(target_file_name).parent.mkdir(parents=True, exist_ok=True)

        # move the frames result file to final target
        shutil.move(temp_frame_result_file_name, target_file_name)

    logging.info(f"Removing temporary processing directory")
    # Remove the temp processing folder
    shutil.rmtree(temp_processing_dir)

    logging.info("Done!")
