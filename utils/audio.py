import json
from dataclasses import dataclass
from subprocess import CalledProcessError, run, PIPE, DEVNULL
from typing import Dict, Optional, Any, Union

import numpy as np


@dataclass
class AudioInfo:
    sample_rate: int
    channels: int
    duration: float
    codec_name: str
    format_name: str

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access for backward compatibility"""
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        """Mimic dict.get() method for backward compatibility"""
        try:
            return getattr(self, key)
        except AttributeError:
            return default


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


def parse_audio_info(audio_info: Dict) -> Optional[AudioInfo]:
    if audio_info is not None and "streams" in audio_info:
        for stream in audio_info["streams"]:
            if stream["codec_type"] == "audio":
                sample_rate = int(stream["sample_rate"])
                channels = int(stream["channels"])
                duration = float(stream["duration"])
                codec_name = stream.get("codec_name", "unknown")
                format_name = audio_info.get("format", {}).get("format_name", "unknown").split(",")[0]
                return AudioInfo(
                    sample_rate=sample_rate,
                    channels=channels,
                    duration=duration,
                    codec_name=codec_name,
                    format_name=format_name,
                )
    return None


def get_audio_info(input_file) -> Optional[AudioInfo]:
    # Run ffprobe to get audio properties and format information in JSON format
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        "-select_streams",
        "a",
        input_file,
    ]
    result = run(cmd, stderr=PIPE, stdout=PIPE, text=True)
    output = result.stdout

    # Parse the JSON output
    info = None
    try:
        raw_info = json.loads(output)
        info = parse_audio_info(raw_info)
    except Exception as e:
        print(e)
        print("Warning - Unable to probe input audio source properties...")
        pass

    return info


def is_mono_audio(audio_info: Optional[AudioInfo]) -> bool:
    return audio_info is not None and audio_info["channels"] == 1  # Cannot detect audio info ? assume non mono


def transcode_to_mono_16k(input_file, output_file):
    cmd = ["ffmpeg", "-y", "-i", input_file, "-ac", "1", "-ar", "16000", output_file]
    run(cmd, stdout=DEVNULL, stderr=DEVNULL)


# Map of known container to audio-only equivalents
AUDIO_ONLY_CONTAINERS = {
    "mp4": "m4a",
    "mkv": "mka",
    "webm": "webm",  # WebM supports audio-only
    "avi": "wav",  # AVI doesn't have a standard audio-only counterpart, WAV is a safe option
    "mov": "m4a",
    "flv": "mp3",  # FLV typically contains MP3 or AAC
}


def get_audio_only_extension(audio_info: Optional[AudioInfo]) -> str:
    """Detects the best audio-only container based on the input audio info."""
    if audio_info is None:
        return "mp3"  # Safe default if no audio info available

    try:
        # Match known container mappings
        format_name = audio_info.format_name
        if format_name in AUDIO_ONLY_CONTAINERS:
            return AUDIO_ONLY_CONTAINERS[format_name]

        # Otherwise, fallback based on audio codec
        audio_codec = audio_info.codec_name

        if audio_codec in ["aac"]:
            return "aac"
        elif audio_codec in ["opus"]:
            return "opus"
        elif audio_codec in ["mp3"]:
            return "mp3"
        elif audio_codec in ["flac"]:
            return "flac"
        elif audio_codec in ["wav", "pcm_s16le"]:
            return "wav"

        # Default fallback
        return "mp3"

    except Exception as e:
        print(f"Error determining audio format: {e}")
        return "mp3"  # Safe default


def extract_audio_from_media(input_file: str, output_audio_base_file: str) -> str:
    """
    Extract audio from media file, automatically determining the best output format.

    Args:
        input_file: Path to the input media file
        output_audio_base_file: Base path for the output audio file (without extension)

    Returns:
        The complete output filename including the extension
    """
    # Get audio info
    audio_info = get_audio_info(input_file)

    # Determine the best audio-only extension
    extension = get_audio_only_extension(audio_info)

    # Create the complete output filename
    output_audio_file = f"{output_audio_base_file}.{extension}"

    # Prepare base ffmpeg cmd (If transcoding is required, it will be to mono)
    cmd = ["ffmpeg", "-y", "-i", input_file, "-vn", "-ac", "1"]

    # Use copy codec when possible for better performance
    cmd += ["-acodec", "copy"]

    # Add the output file
    cmd += [output_audio_file]

    # Run the command
    run(cmd, stdout=DEVNULL, stderr=DEVNULL)

    return output_audio_file
