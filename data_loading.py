import pandas as pd
import requests
from tqdm import tqdm

url = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"

response = requests.get(url, stream=True)
total_size = int(response.headers.get("content-length", 0))

with open("cfpb_complaints.csv.zip", "wb") as f, tqdm(total=total_size, unit="B", unit_scale=True) as bar:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)
        bar.update(len(chunk))

dataset = pd.read_csv("cfpb_complaints.csv.zip", compression="zip", low_memory=False)
print(dataset.shape)

# empty text field handling
def remove_missing_text_rows(df: pd.DataFrame, col: str = 'Consumer complaint narrative'):
    df = df[df[col].notna()]
    df = df[df[col].str.strip() != ""]
    return df

# handle duplication of rows 
def deduplicate(df: pd.DataFrame, col: str = 'Consumer complaint narrative'):
   return df.drop_duplicates(subset=col)

# stratified sampling to go from 10 millions rows to 100.000 - 200.000
def stratify_sample(df: pd.DataFrame, target_col: str="Company response to consumer", sample_size: int=None, random_state: int = 42):
    if sample_size is None:
        return df
    sampling_fraction = sample_size / len(df)
    sampled_groups = []
    for label_value, group in df.groupby(target_col):
        group_sample = group.sample(frac=sampling_fraction, random_state=random_state)
        sampled_groups.append(group_sample)

    return pd.concat(sampled_groups, ignore_index=True)

if __name__ == "__main__":

    raw_df = remove_missing_text_rows(dataset)

    raw_df = deduplicate(raw_df)

    SHORTEN = True
    SAMPLE_SIZE = 10000 if SHORTEN else None
    raw_df = stratify_sample(raw_df, sample_size=SAMPLE_SIZE)

    print("Final sample shape:", raw_df.shape)
    print("\nClass proportions in sample:")
    print(raw_df["Company response to consumer"].value_counts(normalize=True))

    