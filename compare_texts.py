#!/usr/bin/env python3

import jiwer

import argparse
import re
import string


def text_cannonization(text):
    translator = str.maketrans("", "", string.punctuation.replace("-", ""))
    translator[ord("-")] = " "  # Map the hyphen to a space
    text = text.translate(translator)
    text = re.sub(r"\s+", " ", text)
    return text


parser = argparse.ArgumentParser(description="Create a dataset and upload to Huggingface.")

# Add the arguments
parser.add_argument("--reference", type=str, required=True, help="Reference text file.")
parser.add_argument("--test", type=str, required=True, help="Test text file.")

# Parse the arguments
args = parser.parse_args()

ref_text = text_cannonization(open(args.reference, "r").read())
test_text = text_cannonization(open(args.test, "r").read())

res = jiwer.process_words(ref_text, test_text)

print(f"Done comparing {args.reference} and {args.test}. WER={res.wer}, WIL={res.wil}")
