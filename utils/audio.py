from subprocess import CalledProcessError, run

import numpy as np

WHISPER_EXPECTED_SAMPLE_RATE = 16000


def load_audio_in_whisper_format(file: str, sr: int = WHISPER_EXPECTED_SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary
    Shamelessly stolen from https://github.com/openai/whisper/blob/main/whisper/audio.py
    Thanks OpenAI :)

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
