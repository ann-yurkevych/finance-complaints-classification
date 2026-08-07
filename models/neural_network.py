import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import sparse
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight

from features.preprocessing import extract_target
from features.text_features import clean_series

TARGET_COL = "Company response to consumer"
TEXT_COL = "Consumer complaint narrative"
OUTPUT_DIR = "models_results/NeuralNetwork"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_column_transformer():
    return ColumnTransformer(
        transformers=[
            ("text", Pipeline([
                ("clean", FunctionTransformer(clean_series, feature_names_out="one-to-one")),
                ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
            ]), TEXT_COL),

            ("categorical", OneHotEncoder(handle_unknown="ignore"),
             ["Product", "Sub-product", "State", "Company"]),

            ("numeric", StandardScaler(),
             ["word_count", "processing_days", "year"]),
        ],
        remainder="drop"
    )


def build_model(input_dim, num_classes):
    """Returns a plain nn.Sequential model -- no custom class definition needed."""
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, num_classes),
    )


def get_batches(X_sparse, y, batch_size, shuffle=True):
    """Yields (X_batch, y_batch) pairs, converting each batch to dense only
    when it's produced -- keeps memory use low, same purpose the SparseDataset
    class served before, just as a plain generator function instead of a class.
    """
    X_sparse = sparse.csr_matrix(X_sparse)
    n = X_sparse.shape[0]
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]
        X_batch = torch.tensor(X_sparse[batch_idx].toarray(), dtype=torch.float32)
        y_batch = torch.tensor(y[batch_idx], dtype=torch.long)
        yield X_batch, y_batch


def count_batches(n_rows, batch_size):
    return (n_rows + batch_size - 1) // batch_size


def evaluate(model, X_sparse, y, batch_size, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in get_batches(X_sparse, y, batch_size, shuffle=False):
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())
    return np.array(all_labels), np.array(all_preds)


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    train_df = pd.read_csv("train.csv")
    val_df = pd.read_csv("validation.csv")
    test_df = pd.read_csv("test.csv")

    X_train_raw, y_train_raw = extract_target(train_df, TARGET_COL)
    X_val_raw, y_val_raw = extract_target(val_df, TARGET_COL)
    X_test_raw, y_test_raw = extract_target(test_df, TARGET_COL)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    y_val = label_encoder.transform(y_val_raw)
    y_test = label_encoder.transform(y_test_raw)
    num_classes = len(label_encoder.classes_)

    print("Fitting ColumnTransformer on train, transforming train/val/test...")
    column_transformer = build_column_transformer()
    X_train = column_transformer.fit_transform(X_train_raw, y_train)
    X_val = column_transformer.transform(X_val_raw)
    X_test = column_transformer.transform(X_test_raw)

    input_dim = X_train.shape[1]
    print(f"Input feature dimension: {input_dim}")

    BATCH_SIZE_TRAIN = 64
    BATCH_SIZE_EVAL = 128

    # class-weighted loss -- same imbalance handling as sample_weight/class_weight
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    model = build_model(input_dim=input_dim, num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    NUM_EPOCHS = 25
    best_val_macro_f1 = -1
    best_model_state = None

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        num_batches = 0

        for X_batch, y_batch in get_batches(X_train, y_train, BATCH_SIZE_TRAIN, shuffle=True):
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        val_labels, val_preds = evaluate(model, X_val, y_val, BATCH_SIZE_EVAL, DEVICE)
        val_macro_f1 = f1_score(val_labels, val_preds, average="macro")

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - train loss: {total_loss / num_batches:.4f} "
              f"- val macro-F1: {val_macro_f1:.4f}")

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_model_state = model.state_dict()

    print(f"\nBest validation macro-F1: {best_val_macro_f1:.4f}")
    model.load_state_dict(best_model_state)

    print("Evaluating on test set...")
    test_labels, test_preds = evaluate(model, X_test, y_test, BATCH_SIZE_EVAL, DEVICE)

    report_dict = classification_report(test_labels, test_preds, target_names=label_encoder.classes_, output_dict=True)
    report_text = classification_report(test_labels, test_preds, target_names=label_encoder.classes_)
    macro_f1 = f1_score(test_labels, test_preds, average="macro")

    print(report_text)
    print(f"Macro-F1: {macro_f1:.4f}")

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model_state.pt"))
    joblib.dump(column_transformer, os.path.join(OUTPUT_DIR, "column_transformer.joblib"))
    joblib.dump(label_encoder, os.path.join(OUTPUT_DIR, "label_encoder.joblib"))

    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_excel(os.path.join(OUTPUT_DIR, "Feed_forward_neural_network.xlsx"))

    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
        f.write(report_text)
        f.write(f"\nMacro-F1: {macro_f1:.4f}\n")

    print(f"\nSaved model and results to {OUTPUT_DIR}/")