from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from text_features import preprocess_text
import pandas as pd
from preprocessing import (
    extract_target,
    prepare_dataset
)

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

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
val_df = pd.read_csv("validation.csv")

X_train, y_train = extract_target(train_df, 'Company response to consumer')
X_test, y_test = extract_target(test_df, 'Company response to consumer')
X_val, y_val = extract_target(val_df, 'Company response to consumer')

def clean_series(text_series):

    return text_series.apply(preprocess_text) # preprocess_text() includes tokenization, stopwords removal, lemmatization


pipeline = Pipeline([

    ("clean", FunctionTransformer(clean_series)),
    ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))

    # preprocess from prepare_dataset() 
    # convert text to vectors
    # scale numerical features
])

pipeline.fit(X_train['Consumer complaint narrative'], y_train)
predictions = pipeline.predict(X_test['Consumer complaint narrative'])