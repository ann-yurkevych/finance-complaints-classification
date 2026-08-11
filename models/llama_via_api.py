 
import os
import time
import requests
import pandas as pd
from sklearn.metrics import classification_report, f1_score
import json 

TEXT_COLUMN = "Consumer complaint narrative"
TARGET_COLUMN = "Company response to consumer"
SAMPLE_SIZE = 300
 
RESULTS_FOLDER = "models_results/LLM"
os.makedirs(RESULTS_FOLDER, exist_ok=True)
 
CATEGORIES = [
    "Closed with explanation",
    "Closed with non-monetary relief",
    "Closed with monetary relief",
    "Untimely response",
]
 
API_URL = "https://router.huggingface.co/v1/chat/completions"

def load_hf_token(creds_path="creds.json"):
    with open(creds_path) as f:
        creds = json.load(f)
    return creds["huggingface"]["api_key"]

HF_TOKEN = load_hf_token()
 
 
def classify_complaint(text, max_attempts=3):
    """Sends one complaint to the LLM and returns a matching category, or
    'UNKNOWN' if the model's answer couldn't be matched or every attempt failed.
    """
    prompt = (
        "Classify this consumer financial complaint into exactly one category:\n"
        + "\n".join(f"- {c}" for c in CATEGORIES)
        + f'\n\nComplaint:\n"""{text}"""\n\n'
        "Respond with ONLY the exact category name, nothing else."
    )
 
    payload = {
        "model": "meta-llama/Llama-3.3-70B-Instruct",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 30,
        "temperature": 0.0,
    }
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
 
    for attempt in range(max_attempts):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            answer = response.json()["choices"][0]["message"]["content"].strip().lower()
 
            for category in CATEGORIES:
                if category.lower() in answer:
                    return category
            return "UNKNOWN" 
 
        except Exception as error:
            print(f"  attempt {attempt + 1} failed ({error}), retrying...")
            time.sleep(3)
 
    return "UNKNOWN" 
 
 
if __name__ == "__main__":
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN not set. Run: export HF_TOKEN=\"your_token\"")
 
    test_df = pd.read_csv("test.csv")
    sample_df = test_df.sample(SAMPLE_SIZE, random_state=42).copy()
 
    predictions = []
    for i, text in enumerate(sample_df[TEXT_COLUMN], start=1):
        pred = classify_complaint(text)
        predictions.append(pred)
        print(f"[{i}/{SAMPLE_SIZE}] {pred}")
        time.sleep(1)
 
    sample_df["llm_prediction"] = predictions
    sample_df.to_csv(os.path.join(RESULTS_FOLDER, "llm_predictions.csv"), index=False)
 
    scored_df = sample_df[sample_df["llm_prediction"] != "UNKNOWN"]
    unmatched = len(sample_df) - len(scored_df)
 
    macro_f1 = f1_score(scored_df[TARGET_COLUMN], scored_df["llm_prediction"], average="macro")
    report = classification_report(scored_df[TARGET_COLUMN], scored_df["llm_prediction"])
 
    print(f"\nUnmatched: {unmatched}/{len(sample_df)}")
    print(report)
    print(f"Macro-F1: {macro_f1:.4f}")
 
    with open(os.path.join(RESULTS_FOLDER, "classification_report.txt"), "w") as f:
        f.write(f"Unmatched: {unmatched}/{len(sample_df)}\n\n{report}\nMacro-F1: {macro_f1:.4f}\n")
 
    print(f"\nSaved to {RESULTS_FOLDER}/")