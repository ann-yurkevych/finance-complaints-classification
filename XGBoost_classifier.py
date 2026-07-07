from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from text_features import preprocess_text
import pandas as pd

def clean_series(text_series):
    
    return text_series.apply(preprocess_text)


pipeline = Pipeline([
    ("clean", FunctionTransformer(clean_series)),
    ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))
])

pipeline.fit(X_train['Consumer complaint narrative'], y_train)
predictions = pipeline.predict(X_test['Consumer complaint narrative'])