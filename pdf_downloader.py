from urllib.request import urlretrieve
from datasets import load_dataset
import os


# Create pdfs directory if it doesn't exist
os.makedirs("pdfs", exist_ok=True)

dataset = load_dataset("nhop/OpenReview")

for sample in dataset["train"]:
  if sample["openreview_submission_id"] is not None:
      idd = sample["openreview_submission_id"]
      url = f"https://openreview.net/pdf?id={idd}"
      urlretrieve(url, f"pdfs/{idd}.pdf")
