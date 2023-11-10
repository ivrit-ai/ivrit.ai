from flask import Flask, request, jsonify
import json

import base64
import numpy as np
import tempfile
import torch
import torchaudio
import io

import whisper

model = whisper.load_model('large-v3')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024  # 1 Megabyte

from threading import Semaphore

MAX_CONCURRENT_TASKS = 1
task_semaphore = Semaphore(MAX_CONCURRENT_TASKS)

MAX_IN_FLIGHT_TASKS = 10
in_flight_semaphore = Semaphore(MAX_IN_FLIGHT_TASKS)


@app.route('/execute', methods=['POST'])
def execute_task():
    if not in_flight_semaphore.acquire(blocking=False):
        return jsonify({'error': 'Too many tasks in flight'}), 429  # HTTP 429 Too Many Requests

    try:
        # Extract task details
        task = request.json
        task_type = task.get('type')
        data = task.get('data')
        token = task.get('token')

        # Verify token for security (optional, but recommended)

        # Process the task
        with task_semaphore:
            result = process_task(task_type, data)
    finally:
        in_flight_semaphore.release()

    # Return the result
    return jsonify({'result': result})

def process_task(task_type, data):
    # Implement your task processing logic here
    # For example, execute Python code based on task_type and data
    # Decode the base64-encoded MP3 data
    with tempfile.TemporaryDirectory() as d:
        mp3_bytes = base64.b64decode(data)
        open(f'{d}/audio.mp3', 'wb').write(mp3_bytes)

        return model.transcribe(f'{d}/audio.mp3', language='he')

if __name__ == '__main__':
    app.run(debug=True)

