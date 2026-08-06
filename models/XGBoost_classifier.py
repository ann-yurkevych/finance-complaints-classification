from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.feature_extraction.text import TfidfVectorizer
from text_features import (
    preprocess_text
)
import pandas as pd
import os
from preprocessing import (
    extract_target,
    prepare_dataset
)
from sklearn.metrics import classification_report, f1_score
from text_features import clean_series # preprocess_text() includes tokenization, stopwords removal, lemmatization
import joblib
import json

"""Preprocessing steps:
Sample from raw file.
0. Load Train/test/validation (to split run the preprocessing.py). 
1. Deduplication.
2. Delete rows with missing complaints.
3. Dropping columns: Tags, Submitted via, Complaint ID, Company public response.  + two datetime columns
4. Replace missing values with "Unknown" category. 
5. All rare Companies values convert into "Others" category. 
6. Convert two time columns to datetime type from object type.
7. Convert text feature (Consumer narrative complaint) to vectors.
8. Create two additional features: processing time and year. 
9. Encode categorical features. 
10. Scale numerical features. 

"""
target_column = 'Company response to consumer'

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
val_df = pd.read_csv("validation.csv")

X_train, y_train = extract_target(train_df, target_column)
X_test, y_test = extract_target(test_df, target_column)
X_val, y_val = extract_target(val_df, target_column)

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)
y_val = label_encoder.transform(y_val)

column_transformer = ColumnTransformer(
    transformers=[
        ("text", Pipeline([
            ("clean", FunctionTransformer(clean_series, feature_names_out="one-to-one")),
            ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ]), 'Consumer complaint narrative'),

        ("categorical", OneHotEncoder(handle_unknown='ignore'),
         ['Product', 'Sub-product', 'State', 'Company']),

        ("numeric", StandardScaler(),
         ['word_count', 'processing_days', 'year']),
    ],
    remainder='drop'
)

pipeline = Pipeline([
    ("preprocessing", column_transformer),
    ("clf", XGBClassifier(objective="multi:softmax", num_class=len(label_encoder.classes_), eval_metric="mlogloss", random_state=42))
])

sample_weights = compute_sample_weight(class_weight="balanced", y=y_train) # imbalance target classes handling

pipeline.fit(X_train, y_train, clf__sample_weight=sample_weights)
predictions = pipeline.predict(X_test)

output_dir = "models_results/XGBoost"
os.makedirs(output_dir, exist_ok=True)
tuned_output_dir = "models_results/XGBoost_finetuned"
os.makedirs(tuned_output_dir, exist_ok=True)

report_dict = classification_report(y_test, predictions, target_names=label_encoder.classes_, output_dict=True)
macro_f1 = f1_score(y_test, predictions, average='macro')

print(classification_report(y_test, predictions, target_names=label_encoder.classes_))
print(f"Macro-F1: {f1_score(y_test, predictions, average='macro'):.4f}")

report_df = pd.DataFrame(report_dict).transpose()
report_df.to_excel(os.path.join(output_dir, "classification_report.xlsx"))

joblib.dump(pipeline, "models_results/XGBoost/pipeline.joblib") # saved in joblib for shap and feature importance analysis
joblib.dump(label_encoder, "models_results/XGBoost/label_encoder.joblib")

# fine-tuned model
with open("models_results/XGBoost/best_params.json") as f:
    best_params = json.load(f)

pipeline = Pipeline([
    ("preprocessing", column_transformer),
    ("clf", XGBClassifier(
        **best_params,
        objective="multi:softmax",
        num_class=len(label_encoder.classes_),
        eval_metric="mlogloss",
        random_state=42
    ))
])
sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
pipeline.fit(X_train, y_train, clf__sample_weight=sample_weights)

joblib.dump(pipeline, os.path.join(tuned_output_dir, "pipeline_tuned.joblib"))
joblib.dump(label_encoder, os.path.join(tuned_output_dir, "label_encoder.joblib"))

predictions = pipeline.predict(X_test)
report_dict = classification_report(y_test, predictions, target_names=label_encoder.classes_, output_dict=True)
macro_f1 = f1_score(y_test, predictions, average='macro')

print(f"Macro-F1 (tuned): {macro_f1:.4f}")

report_df = pd.DataFrame(report_dict).transpose()
report_df.to_excel(os.path.join(tuned_output_dir, "classification_report.xlsx"))

with open(os.path.join(tuned_output_dir, "macro_f1.txt"), "w") as f:
    f.write(f"Macro-F1: {macro_f1:.4f}\n")

