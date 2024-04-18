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

def initialize_model(model_path, tuned_model_path):
    model = whisper.load_model(model_path)

    if tuned_model_path:
        tuned_model = WhisperForConditionalGeneration.from_pretrained(tuned_model_path)
        model = copy_hf_model_weights_to_openai_model_weights(tuned_model, model)

    def transcribe(filename):
        return transcribe_openai_whisper(model, filename)

    return transcribe

def copy_hf_model_weights_to_openai_model_weights(tuned_model, model):
    dic_parameter_mapping = dict()

    for i in range(32):
        dic_parameter_mapping[f'decoder.blocks.{i}.attn.key.weight'] = f'model.decoder.layers.{i}.self_attn.k_proj.weight'
        dic_parameter_mapping[f'decoder.blocks.{i}.attn.out.bias'] = f'model.decoder.layers.{i}.self_attn.out_proj.bias'
        dic_parameter_mapping[f'decoder.blocks.{i}.attn.out.weight'] = f'model.decoder.layers.{i}.self_attn.out_proj.weight'
        dic_parameter_mapping[f'decoder.blocks.{i}.attn.query.bias'] = f'model.decoder.layers.{i}.self_attn.q_proj.bias'
        dic_parameter_mapping[f'decoder.blocks.{i}.attn.query.weight'] = f'model.decoder.layers.{i}.self_attn.q_proj.weight'
        dic_parameter_mapping[f'decoder.blocks.{i}.attn.value.bias'] = f'model.decoder.layers.{i}.self_attn.v_proj.bias'
        dic_parameter_mapping[f'decoder.blocks.{i}.attn.value.weight'] = f'model.decoder.layers.{i}.self_attn.v_proj.weight'
        dic_parameter_mapping[f'decoder.blocks.{i}.attn_ln.bias'] = f'model.decoder.layers.{i}.self_attn_layer_norm.bias'
        dic_parameter_mapping[f'decoder.blocks.{i}.attn_ln.weight'] = f'model.decoder.layers.{i}.self_attn_layer_norm.weight'
        dic_parameter_mapping[f'decoder.blocks.{i}.cross_attn.key.weight'] = f'model.decoder.layers.{i}.encoder_attn.k_proj.weight'
        dic_parameter_mapping[f'decoder.blocks.{i}.cross_attn.out.bias'] = f'model.decoder.layers.{i}.encoder_attn.out_proj.bias'
        dic_parameter_mapping[f'decoder.blocks.{i}.cross_attn.out.weight'] = f'model.decoder.layers.{i}.encoder_attn.out_proj.weight'
        dic_parameter_mapping[f'decoder.blocks.{i}.cross_attn.query.bias'] = f'model.decoder.layers.{i}.encoder_attn.q_proj.bias'
        dic_parameter_mapping[f'decoder.blocks.{i}.cross_attn.query.weight'] = f'model.decoder.layers.{i}.encoder_attn.q_proj.weight'
        dic_parameter_mapping[f'decoder.blocks.{i}.cross_attn.value.bias'] = f'model.decoder.layers.{i}.encoder_attn.v_proj.bias'
        dic_parameter_mapping[f'decoder.blocks.{i}.cross_attn.value.weight'] = f'model.decoder.layers.{i}.encoder_attn.v_proj.weight'
        dic_parameter_mapping[f'decoder.blocks.{i}.cross_attn_ln.bias'] = f'model.decoder.layers.{i}.encoder_attn_layer_norm.bias'
        dic_parameter_mapping[f'decoder.blocks.{i}.cross_attn_ln.weight'] = f'model.decoder.layers.{i}.encoder_attn_layer_norm.weight'
        dic_parameter_mapping[f'decoder.blocks.{i}.mlp.0.bias'] = f'model.decoder.layers.{i}.fc1.bias'
        dic_parameter_mapping[f'decoder.blocks.{i}.mlp.0.weight'] = f'model.decoder.layers.{i}.fc1.weight'
        dic_parameter_mapping[f'decoder.blocks.{i}.mlp.2.bias'] = f'model.decoder.layers.{i}.fc2.bias'
        dic_parameter_mapping[f'decoder.blocks.{i}.mlp.2.weight'] = f'model.decoder.layers.{i}.fc2.weight'
        dic_parameter_mapping[f'decoder.blocks.{i}.mlp_ln.bias'] = f'model.decoder.layers.{i}.final_layer_norm.bias'
        dic_parameter_mapping[f'decoder.blocks.{i}.mlp_ln.weight'] = f'model.decoder.layers.{i}.final_layer_norm.weight'

        dic_parameter_mapping[f'encoder.blocks.{i}.attn.key.weight'] = f'model.encoder.layers.{i}.self_attn.k_proj.weight'
        dic_parameter_mapping[f'encoder.blocks.{i}.attn.out.weight'] = f'model.encoder.layers.{i}.self_attn.out_proj.weight'
        dic_parameter_mapping[f'encoder.blocks.{i}.attn.query.weight'] = f'model.encoder.layers.{i}.self_attn.q_proj.weight'
        dic_parameter_mapping[f'encoder.blocks.{i}.attn.value.weight'] = f'model.encoder.layers.{i}.self_attn.v_proj.weight'
        dic_parameter_mapping[f'encoder.blocks.{i}.mlp.2.weight'] = f'model.encoder.layers.{i}.fc2.weight'
        dic_parameter_mapping[f'encoder.blocks.{i}.attn.out.bias'] = f'model.encoder.layers.{i}.self_attn.out_proj.bias'
        dic_parameter_mapping[f'encoder.blocks.{i}.attn.query.bias'] = f'model.encoder.layers.{i}.self_attn.q_proj.bias'
        dic_parameter_mapping[f'encoder.blocks.{i}.attn.value.bias'] = f'model.encoder.layers.{i}.self_attn.v_proj.bias'
        dic_parameter_mapping[f'encoder.blocks.{i}.attn_ln.bias'] = f'model.encoder.layers.{i}.self_attn_layer_norm.bias'
        dic_parameter_mapping[f'encoder.blocks.{i}.attn_ln.weight'] = f'model.encoder.layers.{i}.self_attn_layer_norm.weight'
        dic_parameter_mapping[f'encoder.blocks.{i}.mlp.2.bias'] = f'model.encoder.layers.{i}.fc2.bias'
        dic_parameter_mapping[f'encoder.blocks.{i}.mlp_ln.bias'] = f'model.encoder.layers.{i}.final_layer_norm.bias'
        dic_parameter_mapping[f'encoder.blocks.{i}.mlp_ln.weight'] = f'model.encoder.layers.{i}.final_layer_norm.weight'
        dic_parameter_mapping[f'encoder.blocks.{i}.mlp.0.weight'] = f'model.encoder.layers.{i}.fc1.weight'
        dic_parameter_mapping[f'encoder.blocks.{i}.mlp.0.bias'] = f'model.encoder.layers.{i}.fc1.bias'


    dic_parameter_mapping['encoder.conv1.bias'] = 'model.encoder.conv1.bias'
    dic_parameter_mapping['encoder.conv1.weight'] = 'model.encoder.conv1.weight'
    dic_parameter_mapping['encoder.conv2.bias'] = 'model.encoder.conv2.bias'
    dic_parameter_mapping['encoder.conv2.weight'] = 'model.encoder.conv2.weight'
    dic_parameter_mapping['encoder.ln_post.bias'] = 'model.encoder.layer_norm.bias'
    dic_parameter_mapping['encoder.ln_post.weight'] = 'model.encoder.layer_norm.weight'
    dic_parameter_mapping['encoder.positional_embedding'] = 'model.encoder.embed_positions.weight'

    dic_parameter_mapping['decoder.ln.bias'] = 'model.decoder.layer_norm.bias'
    dic_parameter_mapping['decoder.ln.weight'] = 'model.decoder.layer_norm.weight'
    dic_parameter_mapping['decoder.positional_embedding'] = 'model.decoder.embed_positions.weight'
    dic_parameter_mapping['decoder.token_embedding.weight'] = 'model.decoder.embed_tokens.weight'

    model_state_dict = model.state_dict()
    tuned_model_state_dict = tuned_model.state_dict()

    for source_param, target_param in dic_parameter_mapping.items():
        model_state_dict[source_param] = tuned_model_state_dict[target_param]

    model.load_state_dict(model_state_dict)

    return model

def transcribe_openai_whisper(model, filename):
    return model.transcribe(filename, language='he', beam_size=5, best_of=5)['text']

if __name__ == '__main__':
    # Define an argument parser
    parser = argparse.ArgumentParser(description='Create a dataset and upload to Huggingface.')

    # Add the arguments
    parser.add_argument('--model', type=str, required=True, help='Model to use, (large-v2, large-v3 etc).')
    parser.add_argument('--tuned-model', type=str, required=False, help='Huggingface-style tuned model. Can be hosted or local.')
    parser.add_argument('--file', type=str, required=True, help='File to transcribe.')

    # Parse the arguments
    args = parser.parse_args()

    print(f'Loading model {args.model}, tuned model is {args.tuned_model}...')
    transcribe_fn = initialize_model(args.model, args.tuned_model)

    print(f'Transcribing {args.file}...')
    transcribe_fn(args.file)

