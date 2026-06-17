import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import torch
import os

MODEL_NAME    = "distilbert-base-uncased"
OUTPUT_DIR    = "model/finsight-categorizer"
DATA_PATH     = "data/transactions_augmented.csv"
MAX_LENGTH    = 32
BATCH_SIZE    = 16
EPOCHS        = 5
LEARNING_RATE = 2e-5

CATEGORIES = [
    "Food", "Travel", "Shopping", "Entertainment",
    "Health", "Bills", "Groceries", "Education",
    "Transfer", "Investment"
]
label2id = {cat: i for i, cat in enumerate(CATEGORIES)}
id2label = {i: cat for i, cat in enumerate(CATEGORIES)}


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df[df["category"].isin(CATEGORIES)].copy()
    df["label"] = df["category"].map(label2id).astype(int)
    df = df.dropna(subset=["label"])
    print(f"Total samples: {len(df)}")
    print(df["category"].value_counts())
    return df


def tokenize(batch, tokenizer):
    return tokenizer(
        batch["description"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }


def train():
    print("Loading data...")
    df = load_data()

    train_df, val_df = train_test_split(
        df, test_size=0.15, random_state=42, stratify=df["label"]
    )
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    train_ds = Dataset.from_pandas(train_df[["description", "label"]].reset_index(drop=True))
    val_ds   = Dataset.from_pandas(val_df[["description", "label"]].reset_index(drop=True))

    train_ds = train_ds.map(lambda b: tokenize(b, tokenizer), batched=True)
    val_ds   = val_ds.map(lambda b: tokenize(b, tokenizer), batched=True)

    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_ds.set_format("torch",   columns=["input_ids", "attention_mask", "label"])

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(CATEGORIES),
        id2label=id2label,
        label2id=label2id
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir="model/logs",
        logging_steps=20,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics
    )

    print("\nStarting training...")
    trainer.train()

    print("\nSaving model...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")

    results = trainer.evaluate()
    print(f"\nFinal accuracy: {results['eval_accuracy']:.4f}")
    print(f"Final F1:       {results['eval_f1']:.4f}")


if __name__ == "__main__":
    train()