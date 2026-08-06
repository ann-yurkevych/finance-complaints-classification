import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import resample
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

from preprocessing import extract_target
from text_features import clean_series

TARGET_COL = "Company response to consumer"
TEXT_COL = "Consumer complaint narrative"
OUTPUT_DIR = "models_results/XGBoost"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TUNING_SAMPLE_SIZE = 20000  # subsample size used ONLY for tuning, not the final fit


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


def tune_xgb(X_train, y_train, n_iter=10, cv=2):
    """RandomizedSearchCV over XGBoost hyperparameters, scored on macro-F1."""

    clf = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=len(np.unique(y_train)),
        eval_metric="mlogloss",
        device="cpu",
        random_state=42,
    )

    param_distributions = {
        "n_estimators": randint(50, 150),
        "learning_rate": uniform(0.05, 0.25),  
        "max_depth": randint(3, 7),              
        "min_child_weight": randint(1, 11),       
        "subsample": uniform(0.6, 0.4),           
        "colsample_bytree": uniform(0.6, 0.4),     
        "gamma": uniform(0, 0.5),                  
        "reg_alpha": uniform(0, 1),                
        "reg_lambda": uniform(0, 2), 
    }

    search = RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=0,  
    )

    total_fits = n_iter * cv
    with tqdm_joblib(tqdm(desc="Hyperparameter search", total=total_fits)):
        search.fit(X_train, y_train)

    return search.best_params_, search.best_score_


if __name__ == "__main__":
    train_df = pd.read_csv("train.csv")

    X_train_raw, y_train_raw = extract_target(train_df, TARGET_COL)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)

    print(f"Subsampling {TUNING_SAMPLE_SIZE} rows for tuning "
          f"(out of {len(X_train_raw)} total training rows)...")
    X_train_sub, y_train_sub = resample(
        X_train_raw, y_train,
        n_samples=TUNING_SAMPLE_SIZE,
        random_state=42,
        stratify=y_train,
    )

    print("Fitting ColumnTransformer on the tuning subsample...")
    column_transformer = build_column_transformer()
    X_train_transformed = column_transformer.fit_transform(X_train_sub, y_train_sub)

    print("Starting hyperparameter search...")
    best_params, best_macro_f1 = tune_xgb(X_train_transformed, y_train_sub, n_iter=10, cv=2)

    print(f"\nBest macro-F1 (CV on {TUNING_SAMPLE_SIZE}-row subsample): {best_macro_f1:.4f}")
    print("Best params:", best_params)

    with open(os.path.join(OUTPUT_DIR, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=2)

    print(f"\nSaved best_params.json to {OUTPUT_DIR}/")
    print("Next: run XGBoost_classifier.py's tuned-model step, which will refit "
          "these hyperparameters on the FULL training set with sample_weight applied.")