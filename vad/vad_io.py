import os

import numpy as np

from vad.definitions import VAD_SPEECH_PROBS_FILENAME, VAD_SPEECH_PROBS_NP_CACHE_FILENAME


def get_frame_vad_probs_filename(root_dir: str, source: str, episode: str) -> str:
    return os.path.join(root_dir, source, episode, VAD_SPEECH_PROBS_FILENAME)


def load_frame_vad_probs(filename: str) -> list[float]:
    np_cache_file_name = filename.replace(VAD_SPEECH_PROBS_FILENAME, VAD_SPEECH_PROBS_NP_CACHE_FILENAME)
    if os.path.exists(np_cache_file_name):
        return np.load(np_cache_file_name)

    with open(filename, "r") as f:
        speech_probs_per_frame = np.array([float(v) for v in f.readlines() if v])
        np.save(np_cache_file_name, speech_probs_per_frame)
        return speech_probs_per_frame
