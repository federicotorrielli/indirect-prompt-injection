import os
from urllib.request import urlretrieve

import pandas as pd
from tqdm import tqdm  # type: ignore

# Create pdfs directory if it doesn't exist
print("Creating pdfs directory...")
os.makedirs("pdfs", exist_ok=True)
print("✓ pdfs directory ready")

# Load the local CSV dataset
print("Loading dataset from dataset/selected_100_papers.csv...")
dataset = pd.read_csv("dataset/selected_100_papers.csv")
print(f"✓ Loaded dataset with {len(dataset)} papers")

# Filter papers with valid OpenReview submission IDs
valid_papers = dataset[dataset["openreview_submission_id"].notna()]
print(f"✓ Found {len(valid_papers)} papers with valid OpenReview submission IDs")

print("\nStarting PDF downloads...")
for _, sample in tqdm(
    valid_papers.iterrows(), total=len(valid_papers), desc="Downloading PDFs"
):
    idd = sample["openreview_submission_id"]
    url = f"https://openreview.net/pdf?id={idd}"
    filename = f"pdfs/{idd}.pdf"

    try:
        urlretrieve(url, filename)
        tqdm.write(f"✓ Downloaded: {idd}.pdf")
    except Exception as e:
        tqdm.write(f"✗ Failed to download {idd}.pdf: {str(e)}")

print("\n✓ PDF download process completed!")
