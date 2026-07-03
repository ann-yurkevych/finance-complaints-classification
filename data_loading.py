import pandas as pd
import requests
from tqdm import tqdm

def load_dataset():
    url = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    with open("cfpb_complaints.csv.zip", "wb") as f, tqdm(total=total_size, unit="B", unit_scale=True) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    dataset = pd.read_csv("cfpb_complaints.csv.zip", compression="zip", low_memory=False)
    print(dataset.shape)
    print(dataset.head())
    return dataset


if __name__ == "__main__":

    dataset = load_dataset()

    SHORTEN = True
    SAMPLE_SIZE = 10000 if SHORTEN else None
    print("\nClass proportions in sample:")
    print(dataset["Company response to consumer"].value_counts(normalize=True))

 