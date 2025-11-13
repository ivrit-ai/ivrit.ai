"""
Example using faster-whisper with ctranslate2 backend for fast audio transcription.

Run with:
    pip install faster-whisper
    python with_faster_whisper.py
"""

import faster_whisper
model = faster_whisper.WhisperModel('ivrit-ai/whisper-large-v3-turbo-ct2')

segs, _ = model.transcribe('audio.wav', language='he')
text = ' '.join(s.text for s in segs)
print(f'Transcribed text: {text}')
