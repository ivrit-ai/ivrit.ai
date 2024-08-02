import argparse
import base64
from enum import Enum
from http import HTTPStatus
import io
import os
from threading import Semaphore
import threading

from faster_whisper import WhisperModel
from flask import Flask, request, jsonify
import tempfile

from server_models import (
    ResponseFormat,
    CreateTranscriptionRequest,
    segments_to_response,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 3 * 1024 * 1024  # 1 Megabyte


MAX_CONCURRENT_TASKS = 2
task_semaphore = Semaphore(MAX_CONCURRENT_TASKS)

MAX_IN_FLIGHT_TASKS = 5000
in_flight_semaphore = Semaphore(MAX_IN_FLIGHT_TASKS)

whisper_models = []


@app.route("/v1/audio/transcriptions", methods=["POST"])
@app.route("/audio/transcriptions", methods=["POST"])
def transcribe_audio():
    try:
        request_data = CreateTranscriptionRequest(
            **{
                "file": request.files["file"].stream.read(),
                "model": request.form.get("model", "whisper-1"),
                "response_format": request.form.get("response_format", "json"),
                "language": request.form.get("language"),
                "prompt": request.form.get("prompt"),
            }
        )

        if request.form.get("temperature") is not None:
            request_data.temperature = float(request.form.get("temperature"))
        if request.form.get("timestamp_granularities[]") is not None:
            request_data.timestamp_granularities = request.form.getlist("timestamp_granularities[]")
    except Exception as e:
        return str(e), HTTPStatus.BAD_REQUEST

    if not in_flight_semaphore.acquire(blocking=False):
        return (
            jsonify({"error": "Too many tasks in flight"}),
            HTTPStatus.TOO_MANY_REQUESTS,
        )

    try:
        # Process the task
        with task_semaphore:
            model = None
            try:
                # Model parameter is ignored - we use a predefined model at this point
                model = whisper_models.pop()

                segments, transcription_info = model.transcribe(
                    io.BytesIO(request_data.file),
                    task="transcribe",
                    language=request_data.language,
                    initial_prompt=request_data.prompt,
                    word_timestamps="word" in request_data.timestamp_granularities,
                    temperature=request_data.temperature,
                )

                # Format the response
                response = segments_to_response(segments, transcription_info, request_data.response_format)
                if request_data.response_format == ResponseFormat.TEXT:
                    return response, HTTPStatus.OK, {"Content-Type": "text/plain"}

                response_json = response.model_dump_json()
                return (
                    response_json,
                    HTTPStatus.OK,
                    {"Content-Type": "application/json"},
                )

            except Exception as e:
                return jsonify({"error": str(e)}), HTTPStatus.INTERNAL_SERVER_ERROR

            finally:
                if model is not None:
                    whisper_models.append(model)

    finally:
        in_flight_semaphore.release()


def process_task(task_type, data):
    # Implement your task processing logic here
    # For example, execute Python code based on task_type and data
    # Decode the base64-encoded MP3 data
    with tempfile.TemporaryDirectory() as d:
        mp3_bytes = base64.b64decode(data)
        open(f"{d}/audio.mp3", "wb").write(mp3_bytes)

        model = whisper_models.pop()

        ret = {"segments": []}

        segs, dummy = model.transcribe(f"{d}/audio.mp3", language="he")
        for s in segs:
            seg = {
                "id": s.id,
                "seek": s.seek,
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "avg_logprob": s.avg_logprob,
                "compression_ratio": s.compression_ratio,
                "no_speech_prob": s.no_speech_prob,
            }
            ret["segments"].append(seg)

        whisper_models.append(model)

        return ret


if __name__ == "__main__":
    # Configure args
    parser = argparse.ArgumentParser(
        description="Whisper transcription server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="large-v2",
        help="""Size of the model to use (tiny, tiny.en, base, base.en,
small, small.en, distil-small.en, medium, medium.en, distil-medium.en, large-v1,
large-v2, large-v3, large, distil-large-v2 or distil-large-v3), a path to a
converted model directory, or a CTranslate2-converted Whisper model ID from the HF Hub.
When a size or a model ID is configured, the converted model is downloaded
from the Hugging Face Hub.""",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for inference (default: cuda)",
        choices=["cuda", "cpu", "auto"],
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        default="int8",
        help="Type to use for computation. See https://opennmt.net/CTranslate2/quantization.html",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=4500,
        help="Port to listen on (default: 4500)",
    )
    parser.add_argument(
        "--max-concurrent-tasks",
        type=int,
        default=2,
        help="Maximum number of concurrent tasks (default: 1)",
    )
    args = parser.parse_args()

    print(f"Loading whisper model: {args.model}. pid={os.getpid()} tid={threading.current_thread().native_id}")

    whisper_models = []
    for i in range(args.max_concurrent_tasks):
        whisper_models.append(
            whisper_models.append(
                WhisperModel(
                    args.model,
                    device=args.device,
                    compute_type=args.compute_type,
                )
            )
        )

    app.run(host="0.0.0.0", port=args.port)
