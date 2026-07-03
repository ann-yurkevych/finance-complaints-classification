import pandas as pd
from data_loading import *

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

def prepare_dataset(df: pd.DataFrame, sample_size=None):
    df = remove_missing_text_rows(df)
    df = deduplicate(df)
    df = stratify_sample(df, sample_size=sample_size)
    return df

if __name__ == "__main__":

    RAW_PATH = "cfpb_complaints.csv.zip"
    dataset = load_dataset(RAW_PATH)

    SAMPLE_SIZE = 100000
    raw_df = prepare_dataset(dataset, sample_size=SAMPLE_SIZE)
    raw_df.to_csv("raw_df_sample.csv", index=False)

    print(f"Saved {raw_df.shape[0]} rows to raw_df_sample.csv")
    print("\nClass proportions in sample:")
    print(raw_df["Company response to consumer"].value_counts(normalize=True))