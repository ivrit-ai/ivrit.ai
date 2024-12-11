# This code was copied and adapted nemo/collections/asr/parts/utils/vad_utils.py
# Due to the autocast bug we wanted to avoid.
# The generate_vad_frame_pred() function now will not use autocast at all
# around inference code
import json
import os

import torch
from nemo.collections.asr.parts.utils.vad_utils import get_vad_stream_status
from nemo.utils import logging
from tqdm import tqdm


def generate_vad_frame_pred(
    vad_model,
    window_length_in_sec: float,
    shift_length_in_sec: float,
    manifest_vad_input: str,
    out_dir: str,
    use_feat: bool = False,
) -> str:
    """
    Generate VAD frame level prediction and write to out_dir
    """
    time_unit = int(window_length_in_sec / shift_length_in_sec)
    trunc = int(time_unit / 2)
    trunc_l = time_unit - trunc
    all_len = 0

    data = []
    with open(manifest_vad_input, "r", encoding="utf-8") as f:
        for line in f:
            file = json.loads(line)["audio_filepath"].split("/")[-1]
            data.append(file.split(".wav")[0])
    logging.info(f"Inference on {len(data)} audio files/json lines!")

    status = get_vad_stream_status(data)
    for i, test_batch in enumerate(tqdm(vad_model.test_dataloader(), total=len(vad_model.test_dataloader()))):
        test_batch = [x.to(vad_model.device) for x in test_batch]
        with torch.amp.autocast(vad_model.device.type, dtype=torch.float32):
            if use_feat:
                log_probs = vad_model(processed_signal=test_batch[0], processed_signal_length=test_batch[1])
            else:
                log_probs = vad_model(input_signal=test_batch[0], input_signal_length=test_batch[1])
            probs = torch.softmax(log_probs, dim=-1)
            if len(probs.shape) == 3 and probs.shape[0] == 1:
                # squeeze the batch dimension, since batch size is 1 for frame-VAD
                probs = probs.squeeze(0)  # [1,T,C] -> [T,C]
            pred = probs[:, 1]

            if window_length_in_sec == 0:
                to_save = pred
            elif status[i] == "start":
                to_save = pred[:-trunc]
            elif status[i] == "next":
                to_save = pred[trunc:-trunc_l]
            elif status[i] == "end":
                to_save = pred[trunc_l:]
            else:
                to_save = pred

            to_save = to_save.cpu().tolist()
            all_len += len(to_save)
            outpath = os.path.join(out_dir, data[i] + ".frame")
            with open(outpath, "a", encoding="utf-8") as fout:
                for f in range(len(to_save)):
                    fout.write("{0:0.4f}\n".format(to_save[f]))

        del test_batch
        if status[i] == "end" or status[i] == "single":
            logging.debug(f"Overall length of prediction of {data[i]} is {all_len}!")
            all_len = 0
    return out_dir
