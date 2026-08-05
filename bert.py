from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
 
import os
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
 
TARGET_COL = "Company response to consumer"
TEXT_COL = "Consumer complaint narrative"
OUTPUT_DIR = "models_results/BERT"
MODEL_NAME = "distilbert-base-uncased"
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
 
def load_and_encode_labels():
    train_df = pd.read_csv("train.csv")
    val_df = pd.read_csv("validation.csv")
    test_df = pd.read_csv("test.csv")
 
    label_encoder = LabelEncoder()
    train_df["label"] = label_encoder.fit_transform(train_df[TARGET_COL])
    val_df["label"] = label_encoder.transform(val_df[TARGET_COL])
    test_df["label"] = label_encoder.transform(test_df[TARGET_COL])
 
    return train_df, val_df, test_df, label_encoder
 
 
def build_datasets(train_df, val_df, test_df, tokenizer):
    def tokenize_function(examples):
        return tokenizer(
            examples[TEXT_COL],
            truncation=True,
            padding="max_length",
            max_length=512,
        )
 
    # keep only the columns the Trainer actually needs, to avoid dtype issues
    train_ds = Dataset.from_pandas(train_df[[TEXT_COL, "label"]])
    val_ds = Dataset.from_pandas(val_df[[TEXT_COL, "label"]])
    test_ds = Dataset.from_pandas(test_df[[TEXT_COL, "label"]])
 
    train_ds = train_ds.map(tokenize_function, batched=True)
    val_ds = val_ds.map(tokenize_function, batched=True)
    test_ds = test_ds.map(tokenize_function, batched=True)
 
    return train_ds, val_ds, test_ds
 
 
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, predictions, average="macro")
    return {"macro_f1": macro_f1}
 
 
class WeightedLossTrainer(Trainer):
    """Custom Trainer that applies class weights to the loss, since BERT has no
    built-in class_weight parameter the way sklearn/XGBoost do -- this is the
    transformer-world equivalent of the sample_weight we used for XGBoost.
    """
 
    def __init__(self, class_weights, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
 
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss
 
 
if __name__ == "__main__":
    train_df, val_df, test_df, label_encoder = load_and_encode_labels()
    num_labels = len(label_encoder.classes_)
 
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels
    )
 
    train_ds, val_ds, test_ds = build_datasets(train_df, val_df, test_df, tokenizer)
 
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_df["label"]),
        y=train_df["label"],
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)
 
    training_args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_steps=50,
    )
 
    trainer = WeightedLossTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
 
    print("Fine-tuning DistilBERT...")
    trainer.train()
 
    print("Evaluating on test set...")
    predictions_output = trainer.predict(test_ds)
    predictions = np.argmax(predictions_output.predictions, axis=-1)
 
    report = classification_report(
        test_df["label"], predictions, target_names=label_encoder.classes_
    )
    macro_f1 = f1_score(test_df["label"], predictions, average="macro")
 
    print(report)
    print(f"Macro-F1: {macro_f1:.4f}")
 
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
        f.write(report)
        f.write(f"\nMacro-F1: {macro_f1:.4f}\n")
 
    model.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
 
    print(f"\nSaved model and results to {OUTPUT_DIR}/")