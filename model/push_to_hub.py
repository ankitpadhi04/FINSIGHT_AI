# model/push_to_hub.py

from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast
)
from huggingface_hub import login
import os
from dotenv import load_dotenv

load_dotenv()

login(token=os.getenv("HF_TOKEN"))

REPO_NAME = "ankitpadhi04/finsight-categorizer"  
LOCAL_DIR = "model/finsight-categorizer"

print("Loading model...")
model     = DistilBertForSequenceClassification.from_pretrained(LOCAL_DIR)
tokenizer = DistilBertTokenizerFast.from_pretrained(LOCAL_DIR)

print(f"Pushing to {REPO_NAME}...")
model.push_to_hub(REPO_NAME)
tokenizer.push_to_hub(REPO_NAME)

print(f"\nDone. Model live at: https://huggingface.co/{REPO_NAME}")