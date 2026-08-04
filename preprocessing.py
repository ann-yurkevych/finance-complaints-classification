import pandas as pd
from data_loading import *
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
import re
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

def add_word_count_feature(df: pd.DataFrame, text_col: str = 'Consumer complaint narrative') -> pd.DataFrame:
    df = df.copy()
    df['word_count'] = df[text_col].str.split().str.len()
    return df

def remove_redaction_tokens(text: str) -> str:
    text = re.sub(r'\bx{2,}\b', '', text, flags=re.IGNORECASE)
    return text

def drop_cols(df: pd.DataFrame, columns_to_drop: list):
    return df.drop(columns=columns_to_drop, errors="ignore")

# Replace missing values with "Unknown" category. 
def fill_categorical_missing(df: pd.DataFrame):
    df = df.copy()
    df['Sub-product'] = df['Sub-product'].fillna('Not applicable')
    df['Sub-issue'] = df['Sub-issue'].fillna('Not applicable')
    df['State'] = df['State'].fillna('Unknown')
    return df

# All rare Companies values convert into "Others" category. 
def group_rare_companies(df: pd.DataFrame, company_col: str = 'Company', top_n: int = 20):
    df = df.copy()
    top_companies = df[company_col].value_counts().head(top_n).index
    df[company_col] = df[company_col].where(df[company_col].isin(top_companies), 'Other')
    return df

# Convert two time columns to datetime type from object type.
def convert_date_columns(df: pd.DataFrame):
    """Converts the two date columns from object/string to datetime type."""
    df = df.copy()
    df['Date received'] = pd.to_datetime(df['Date received'], format='mixed', utc=True)
    df['Date sent to company'] = pd.to_datetime(df['Date sent to company'], format='mixed', utc=True)
    return df

# Create two additional features: processing time and year. 
def add_processing_days_feature(df: pd.DataFrame):
    """Derives processing_days: gap between Date received and Date sent to company."""
    df = df.copy()
    df['processing_days'] = (df['Date sent to company'] - df['Date received']).dt.days
    return df

def add_year_feature(df: pd.DataFrame):
    """Derives year from Date received."""
    df = df.copy()
    df['year'] = df['Date received'].dt.year
    return df

# prepare_dataset() contains all preprocessing steps from earlier defined function, in the Pipeline() will not be called
def prepare_dataset(df: pd.DataFrame, sample_size=None, text_col='Consumer complaint narrative'): 
    df = drop_cols(df, ['Tags', 'Submitted via', 'Complaint ID', 'Company public response'])
    df = remove_missing_text_rows(df)
    df[text_col] = df[text_col].apply(remove_redaction_tokens) # strip XXXX and XX 
    df = deduplicate(df)
    df = stratify_sample(df, sample_size=sample_size)
    df = drop_rare_classes(df, 'Company response to consumer') # rare target values: initially there was 7 of them, but only 5 important
    df = add_word_count_feature(df, text_col)
    df = fill_categorical_missing(df)
    df = group_rare_companies(df)
    df = convert_date_columns(df)
    df = add_processing_days_feature(df)
    df = add_year_feature(df)
    return df

def extract_target(df: pd.DataFrame, target_col: str): 
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
    return X, y

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
