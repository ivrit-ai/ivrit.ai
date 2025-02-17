import json
from subprocess import CalledProcessError, run, PIPE, DEVNULL

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


def parse_audio_info(audio_info):
    if audio_info is not None and "streams" in audio_info:
        for stream in audio_info["streams"]:
            if stream["codec_type"] == "audio":
                sample_rate = int(stream["sample_rate"])
                channels = int(stream["channels"])
                duration = float(stream["duration"])
                return {"sample_rate": sample_rate, "channels": channels, "duration": duration}
    return None


def get_audio_info(input_file):
    # Run ffprobe to get audio properties in JSON format
    cmd = ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-select_streams", "a", input_file]
    result = run(cmd, stderr=PIPE, stdout=PIPE, text=True)
    output = result.stdout

    # Parse the JSON output
    info = None
    try:
        raw_info = json.loads(output)
        info = parse_audio_info(raw_info)
    except:
        print("Warning - Unable to probe input audio source properties...")
        pass

    return info


def is_mono_audio(audio_info):
    return audio_info is not None and audio_info["channels"] == 1  # Cannot detect audio info ? assume non mono


def transcode_to_mono_16k(input_file, output_file):
    cmd = ["ffmpeg", "-y", "-i", input_file, "-ac", "1", "-ar", "16000", output_file]
    run(cmd, stdout=DEVNULL, stderr=DEVNULL)
