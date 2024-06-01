#!/usr/bin/env python
# coding: utf-8

import evaluate
import numpy as np
import pandas as pd
import torch


from datasets import Audio, load_dataset, load_from_disk
from functools import partial

from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE
from transformers import WhisperProcessor
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer

from dataclasses import dataclass
from typing import Any, Dict, List, Union

#eval_dataset = load_dataset("ivrit-ai/audio-labeled", token='')
dataset = load_from_disk('/home/ubuntu/data/ivrit-13-20240601')

processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2", language="hebrew", task="transcribe")

sampling_rate = processor.feature_extractor.sampling_rate
dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))


def prepare_dataset(example, text_column_name):

    try:
        audio = example["audio"]

        example = processor(
            audio=audio["array"],
            sampling_rate=audio["sampling_rate"],
            text=example[text_column_name],
        )

        # compute input length of audio sample in seconds
        example["input_length"] = len(audio["array"]) / audio["sampling_rate"]

        return example
    except Exception as e:
        print('Exception')
        return None

def prepare_dataset_sentence(example):
    return prepare_dataset(example, 'sentence')

def prepare_dataset_text(example):
    return prepare_dataset(example, 'text')


train_set = dataset 
#eval_set = eval_dataset['test'] 

train_set = train_set.map(prepare_dataset_sentence, remove_columns=train_set.column_names, num_proc=1)
#eval_set = eval_set.map(prepare_dataset_text, remove_columns=eval_set.column_names, num_proc=1)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"][0]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

metric = evaluate.load("wer")
normalizer = BasicTextNormalizer()


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # compute orthographic wer
    wer_ortho = metric.compute(predictions=pred_str, references=label_str)

    # compute normalised WER
    pred_str_norm = [normalizer(pred) for pred in pred_str]
    label_str_norm = [normalizer(label) for label in label_str]
    # filtering step to only evaluate the samples that correspond to non-zero references:
    pred_str_norm = [
        pred_str_norm[i] for i in range(len(pred_str_norm)) if len(label_str_norm[i]) > 0
    ]
    label_str_norm = [
        label_str_norm[i]
        for i in range(len(label_str_norm))
        if len(label_str_norm[i]) > 0
    ]

    wer = metric.compute(predictions=pred_str_norm, references=label_str_norm)

    return {"wer_ortho": wer_ortho, "wer": wer}



model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
model.config.use_cache = False
model.generate = partial(model.generate, language="hebrew", task="transcribe", use_cache=True)




output_dir_trainer = 'whisper-large-yair'



training_args = Seq2SeqTrainingArguments(
    output_dir = output_dir_trainer,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    lr_scheduler_type="constant_with_warmup",
    warmup_ratio=0.1,
#     warmup_steps=100,
#     max_steps=2000,  # increase to 4000 if you have your own GPU or a Colab paid plan
#     gradient_checkpointing=True,
#     fp16=True,
#     fp16_full_eval=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=10,
    weight_decay=0.05,
    per_device_eval_batch_size=16,
    predict_with_generate=True,
    generation_max_length=225,
#     save_steps=250,
#     eval_steps=250,
    logging_strategy="epoch",
    report_to='none',
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)



trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_set,
    #eval_dataset=eval_set,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)

print('Start training!')
trainer.train()


kwargs = {
    "dataset_tags": "ivrit-ai/ivrit-13-20240601",
    "finetuned_from": "openai/whisper-large-v2",
    "tasks": "automatic-speech-recognition",
    "model_name": "ivrit-ai/large-v2-ivrit-13-20240601",
}

