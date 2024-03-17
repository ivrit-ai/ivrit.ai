import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration

import datasets
import jiwer

import numpy

model = WhisperForConditionalGeneration.from_pretrained('/mnt/c/Users/yair/Desktop/whisper_v3_hebrew/') 
processor = WhisperProcessor.from_pretrained('/mnt/c/Users/yair/Desktop/whisper_v3_hebrew/')
test_ds = datasets.load_dataset('ivrit-ai/whisper-training')['test']

tf = jiwer.transforms.RemovePunctuation()

def transcribe(entry):
    audio_resample = librosa.resample(entry['audio']['array'], orig_sr=entry['audio']['sampling_rate'], target_sr=16000)
    input_features = processor(audio_resample, sampling_rate=16000, return_tensors="pt").input_features
    predicted_ids = model.generate(input_features, language='he', num_beams=5)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return transcription

def batch_compare():
    wers = []
    for i in range(8426):
        text = transcribe(test_ds[i])
        wer = jiwer.wer(tf(text), tf(test_ds[i]['text']))
        wers.append(wer)
        print(f'#{i}: wer={wer}, avg={numpy.average(wers)}')
