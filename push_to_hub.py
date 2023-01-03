import argparse
import os
import torch
from transformers import AutoModelForSequenceClassification

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="gpt-neo-125M-apps")
parser.add_argument("--model_path", type=str, default="checkpoints/gpt-neo-125M-apps/final_checkpoint")
args = parser.parse_args()

model = AutoModelForSequenceClassification.from_pretrained(args.model_path)

model.push_to_hub(args.model_name)
