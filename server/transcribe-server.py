import base64
from enum import Enum
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
            request_data.timestamp_granularities = request.form.getlist(
                "timestamp_granularities[]"
            )
    except Exception as e:
        return str(e), 400

    if not in_flight_semaphore.acquire(blocking=False):
        return (
            jsonify({"error": "Too many tasks in flight"}),
            429,
        )  # HTTP 429 Too Many Requests

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
                response = segments_to_response(
                    segments, transcription_info, request_data.response_format
                )
                if request_data.response_format == ResponseFormat.TEXT:
                    return response, 200, {"Content-Type": "text/plain"}

                response_json = response.model_dump_json()
                return response_json, 200, {"Content-Type": "application/json"}

            except Exception as e:
                return jsonify({"error": str(e)}), 500

            finally:
                if model is not None:
                    whisper_models.append(model)

    finally:
        in_flight_semaphore.release()


@app.route("/execute", methods=["POST"])
def execute_task():
    if not in_flight_semaphore.acquire(blocking=False):
        return (
            jsonify({"error": "Too many tasks in flight"}),
            429,
        )  # HTTP 429 Too Many Requests

    try:
        # Extract task details
        task = request.json
        task_type = task.get("type")
        data = task.get("data")
        token = task.get("token")

        # Verify token for security (optional, but recommended)

        # Process the task
        with task_semaphore:
            result = process_task(task_type, data)
    finally:
        in_flight_semaphore.release()

    # Return the result
    return jsonify({"result": result})


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
    print(
        f"Loading whisper models... pid={os.getpid()} tid={threading.current_thread().native_id}"
    )

    whisper_models = []
    for i in range(MAX_CONCURRENT_TASKS):
        whisper_models.append(
            whisper_models.append(
                WhisperModel("large-v2", device="cuda", compute_type="int8")
            )
        )

    app.run(host="0.0.0.0", port=4500)
