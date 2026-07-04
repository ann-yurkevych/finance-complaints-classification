import pandas as pd
from data_loading import *
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize

# empty text field handling
def remove_missing_text_rows(df: pd.DataFrame, col: str = 'Consumer complaint narrative'):
    df = df[df[col].notna()]
    df = df[df[col].str.strip() != ""]
    return df

# handle duplication of rows 
def deduplicate(df: pd.DataFrame, col: str = 'Consumer complaint narrative'):
   return df.drop_duplicates(subset=col)

# stratified sampling to go from 10 millions rows to 100.000 - 200.000, range is custom
def stratify_sample(df: pd.DataFrame, target_col: str="Company response to consumer", sample_size: int=None, random_state: int = 42):
    if sample_size is None:
        return df
    sampling_fraction = sample_size / len(df)
    sampled_groups = []
    for label_value, group in df.groupby(target_col):
        group_sample = group.sample(frac=sampling_fraction, random_state=random_state)
        sampled_groups.append(group_sample)

    return pd.concat(sampled_groups, ignore_index=True)

# splitting into train, test, validation: 70/15/15
def split_train_test(df: pd.DataFrame, target: str, test_size: float=0.3, random_state: int=42):

    X_train, X_test = train_test_split(
        df,
        test_size=test_size,  # 70% train, 30% test
        stratify=df[target],
        random_state=random_state
    )

    X_test, X_validation = train_test_split(
        X_test,
        test_size=0.5,  # split the 30% of X_test into 15% test, 15% validation
        stratify=X_test[target],
        random_state=random_state
    )
    return X_train, X_test, X_validation

def drop_rare_classes(df: pd.DataFrame, target: str, min_count: int = 20):

    counts = df[target].value_counts()
    rare_classes = counts[counts < min_count].index.tolist()
    if rare_classes:
        print(f"Dropping rare classes: {min_count}: {rare_classes}")
        df = df[~df[target].isin(rare_classes)]

    return df

def prepare_dataset(df: pd.DataFrame, sample_size=None):
    df = remove_missing_text_rows(df)
    df = deduplicate(df)
    df = stratify_sample(df, sample_size=sample_size)
    df = drop_rare_classes(df, 'Company response to consumer')

    return df
"""
Steps for text preprocessing:
1. Tokenization: word-level tokenization. 
2. Stopwords removal.
3. Lemmatization. 
4. Vectorization. 
"""

# tokenization

# handle class imnbalance


if __name__ == "__main__":

    RAW_PATH = "cfpb_complaints.csv.zip"
    dataset = load_dataset(RAW_PATH)

    SAMPLE_SIZE = 100000
    raw_df = prepare_dataset(dataset, sample_size=SAMPLE_SIZE)
    raw_df.to_csv("raw_df_sample.csv", index=False)

    print(f"Saved {raw_df.shape[0]} rows to raw_df_sample.csv")
    print("\nClass proportions in sample:")
    print(raw_df["Company response to consumer"].value_counts(normalize=True))

    X_train, X_test, X_validation = split_train_test(raw_df, 'Company response to consumer')
    print(f"\nTrain shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"Validation shape: {X_validation.shape}")

    X_train.to_csv("train.csv", index=False)
    X_test.to_csv("test.csv", index=False)
    X_validation.to_csv("validation.csv", index=False)
