#!/usr/bin/env python3

import argparse

import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import faster_whisper

import torch

import io
import datasets
import jiwer
import json

import soundfile
import numpy
import pydub

import whisper
import whisper.normalizers

# Supported engines and models:
#
# 1. Engine: openai-whisper
#    - Models: large-v2, large-v3
# 2. Engine: transformers
#    - Models: openai/whisper-large-v2, openai/whisper-large-v3, user-trained models
# 3. Engine: faster-whisper
#    - Models: large-v2, large-v3, user-trained models


def clean_text(text):
    transformations = [
        jiwer.RemovePunctuation(),
        jiwer.RemoveEmptyStrings(),
        jiwer.ToLowerCase(),
        jiwer.RemoveWhiteSpace(replace_by_space=True),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(word_delimiter=" ")
    ]
    
    processed_text = text
    for transformation in transformations:
        processed_text = transformation(processed_text)
    
    return processed_text


def initialize_model(engine, model_path):
    if engine == 'openai-whisper':
        model = whisper.load_model(model_path)

        def transcribe(entry):
            return transcribe_openai_whisper(model, entry)

        return transcribe

    if engine == 'transformers':
        model = WhisperForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float32)
        model.to('cuda:0')

        processor = WhisperProcessor.from_pretrained(model_path)

        def transcribe(entry):
            return transcribe_transformers(model, processor, entry)

        return transcribe

    if engine == 'faster-whisper':
        model = faster_whisper.WhisperModel(model_path, device='cuda', compute_type='float32')

        def transcribe(entry):
            return transcribe_faster_whisper(model, entry)

        return transcribe

    raise Exception

def transcribe_transformers(model, processor, entry):
    audio_resample = librosa.resample(entry['audio']['array'], orig_sr=entry['audio']['sampling_rate'], target_sr=16000)
    input_features = processor(audio_resample, sampling_rate=16000, return_tensors="pt").input_features
    input_features = input_features.to('cuda:0')
    predicted_ids = model.generate(input_features, language='he', num_beams=5)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return transcription[0]

def transcribe_faster_whisper(model, entry):
    wav_buffer = io.BytesIO()
    soundfile.write(wav_buffer, entry['audio']['array'], entry['audio']['sampling_rate'], format='WAV')
    wav_buffer.seek(0)

    texts = []
    segs, dummy = model.transcribe(wav_buffer, language='he')
    for s in segs:
        texts.append(s.text)

    return ' '.join(texts)

def transcribe_openai_whisper(entry):
    wav_buffer = io.BytesIO()
    soundfile.write(wav_buffer, entry['audio']['array'], entry['audio']['sampling_rate'], format='WAV')
    wav_buffer.seek(0)  # Rewind to the start of the buffer

    audio = pydub.AudioSegment.from_file(wav_buffer, format="wav")
    audio.export('x.mp3', format="mp3")

    return model.transcribe('x.mp3', language='he', beam_size=5, best_of=5)['text']

def evaluate_model(transcribe_fn, ds, text_column,normalizer = clean_text): 

    ref_texts = []
    texts = []

    for i in range(len(ds)):
        ref_text = clean_text(ds[i][text_column])
        text = clean_text(transcribe_fn(ds[i]))

        ref_texts.append(ref_text)
        texts.append(text)

        if i > 0 and i % 10 == 0:
            print(f'Evaluated {i}/{len(ds)}, WER={jiwer.wer(texts, ref_texts)}')

    return jiwer.wer(texts, ref_texts)

if __name__ == '__main__':
    # Define an argument parser
    parser = argparse.ArgumentParser(description='Create a dataset and upload to Huggingface.')

    # Add the arguments
    parser.add_argument('--engine', type=str, required=True, choices={'openai-whisper', 'transformers', 'faster-whisper'}, help='Engine to use.')
    parser.add_argument('--model', type=str, required=True, help='Model to use. Can be remote (e.g. openai/whisper-large-v3) or local (full path).')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to evaluate.')
    parser.add_argument('--text-column', type=str, required=True, help='Name of reference transcription column in dataset.')
    

    # Parse the arguments
    args = parser.parse_args()

    print(f'Loading engine {args.engine} with model {args.model}...')
    transcribe_fn = initialize_model(args.engine, args.model)

    print(f'Loading dataset {args.dataset}...')
    ds = datasets.load_dataset(args.dataset)['test']
    ds_text_column = args.text_column

    print(f'Beginning evaluation.')
    evaluate_model(transcribe_fn, ds, ds_text_column)

