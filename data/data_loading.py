import pandas as pd
import requests
from tqdm import tqdm
import os

def load_dataset(save_path: str="cfpb_complaints.csv.zip"):

    url = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"

    if not os.path.exists(save_path):
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))

        with open(save_path, "wb") as f, tqdm(total=total_size, unit="B", unit_scale=True) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
    else:
        print(f"{save_path} already exists")

    dataset = pd.read_csv(save_path, compression="zip", low_memory=False)
    print(dataset.shape)
    print(dataset.head())
    return dataset


if __name__ == "__main__":

    dataset = load_dataset()
    

 