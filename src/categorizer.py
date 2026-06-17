# src/categorizer.py

import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

HF_MODEL_REPO        = "ankitpadhi04/finsight-categorizer"  # update this
CONFIDENCE_THRESHOLD = 0.75
CATEGORIES = [
    "Food", "Travel", "Shopping", "Entertainment",
    "Health", "Bills", "Groceries", "Education",
    "Transfer", "Investment"
]

_model     = None
_tokenizer = None
_groq      = None


def load_model():
    global _model, _tokenizer
    if _model is None:
        print("Loading DistilBERT from HuggingFace Hub...")
        _tokenizer = DistilBertTokenizerFast.from_pretrained(HF_MODEL_REPO)
        _model     = DistilBertForSequenceClassification.from_pretrained(HF_MODEL_REPO)
        _model.eval()
    return _model, _tokenizer


def get_groq():
    global _groq
    if _groq is None:
        _groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq


def distilbert_predict(description: str):
    model, tokenizer = load_model()
    inputs = tokenizer(
        description,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=32
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs      = torch.softmax(logits, dim=-1)[0]
    confidence = probs.max().item()
    label_id   = probs.argmax().item()
    category   = model.config.id2label[label_id]
    return category, confidence


def groq_predict(description: str) -> str:
    client = get_groq()
    prompt = f"""You are a financial transaction categorizer for Indian UPI payments.
Categorize this transaction into exactly one of these categories:
Food, Travel, Shopping, Entertainment, Health, Bills, Groceries, Education, Transfer, Investment

Transaction: "{description}"

Rules:
- Reply with ONLY the category name, nothing else, no punctuation
- Personal names (Ramesh, Priya, etc.) = Transfer
- Local shop names = guess from context
- Petrol/fuel/cab = Travel
- Medical/pharmacy = Health

Category:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0
    )
    result = response.choices[0].message.content.strip()

    for cat in CATEGORIES:
        if cat.lower() in result.lower():
            return cat
    return "Transfer"


def categorize(description: str) -> dict:
    category, confidence = distilbert_predict(description)

    if confidence >= CONFIDENCE_THRESHOLD:
        return {
            "category":   category,
            "confidence": round(confidence, 4),
            "method":     "distilbert"
        }
    else:
        groq_category = groq_predict(description)
        return {
            "category":   groq_category,
            "confidence": round(confidence, 4),
            "method":     "groq"
        }


def categorize_batch(descriptions: list) -> list:
    return [categorize(desc) for desc in descriptions]