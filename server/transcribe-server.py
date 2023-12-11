from flask import Flask, request, jsonify
import json

import base64
import numpy as np
import tempfile
import torch
import torchaudio
import io

import os
import threading

from faster_whisper import WhisperModel

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 3 * 1024 * 1024  # 1 Megabyte

from threading import Semaphore

MAX_CONCURRENT_TASKS = 2
task_semaphore = Semaphore(MAX_CONCURRENT_TASKS)

MAX_IN_FLIGHT_TASKS = 5000
in_flight_semaphore = Semaphore(MAX_IN_FLIGHT_TASKS)

whisper_models = [] 

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

        model = whisper_models.pop()

        ret = { 'segments' : [] }

        segs, dummy = model.transcribe(f'{d}/audio.mp3', language='he')
        for s in segs:
            seg = { 'id' : s.id, 'seek' : s.seek, 'start' : s.start, 'end' : s.end, 'text' : s.text, 'avg_logprob' : s.avg_logprob, 'compression_ratio' : s.compression_ratio, 'no_speech_prob' : s.no_speech_prob }
            ret['segments'].append(seg) 

        whisper_models.append(model)      

        return ret

if __name__ == '__main__':
    print(f'Loading whisper models... pid={os.getpid()} tid={threading.current_thread().native_id}')

    whisper_models = []
    for i in range(MAX_CONCURRENT_TASKS):
        whisper_models.append(WhisperModel('large-v2', device='cuda', compute_type='int8'))

    app.run(host='0.0.0.0', port=4500)

