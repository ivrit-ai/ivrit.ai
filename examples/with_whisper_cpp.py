"""
Example of using whispercpp for fast and lightweight transcription.

Download the model from Hugging Face:
    wget https://huggingface.co/ivrit-ai/whisper-large-v3-turbo-ggml/resolve/main/ggml-model.bin

Run with:
    pip install pywhispercpp
    python with_whisper_cpp.py
"""

from pywhispercpp.model import Model

model = Model('./ggml-model.bin')
segs = model.transcribe('audio.wav', language='he')
text = ' '.join(segment.text for segment in segs)
print(f'Transcribed text: {text}')
