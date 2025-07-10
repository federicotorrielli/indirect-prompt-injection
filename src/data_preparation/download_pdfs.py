import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import urlretrieve

import fitz  # PyMuPDF
import pandas as pd
from tqdm import tqdm  # type: ignore


def download_pdf(paper_data: tuple) -> tuple[str, str, bool]:
    """Download a single PDF and return status."""
    idd, url, filename = paper_data
    try:
        urlretrieve(url, filename)
        return idd, filename, True
    except Exception as e:
        return idd, str(e), False


def redact_conference_info(pdf_path: str) -> None:
    """Redact conference submission and publication information from PDF."""
    # Optimized patterns - longest first, case-insensitive matching
    redaction_terms = [
        # Long specific patterns
        "international conference on learning representations",
        "conference on neural information processing systems",
        "international conference on machine learning",
        "published as a conference paper at",
        "under review as a conference paper at",
        "anonymous submission",
        "conference paper",
        "workshop paper",
        "under review",
        "submitted to",
        "proceedings of",
        "accepted at",
        # Conference names with years
        "neurips 2024",
        "neurips 2023",
        "neurips 2022",
        "neurips 2021",
        "neurips 2020",
        "icml 2024",
        "icml 2023",
        "icml 2022",
        "icml 2021",
        "icml 2020",
        "iclr 2024",
        "iclr 2023",
        "iclr 2022",
        "iclr 2021",
        "iclr 2020",
        "aaai 2024",
        "aaai 2023",
        "aaai 2022",
        "aaai 2021",
        "aaai 2020",
        "ijcai 2024",
        "ijcai 2023",
        "ijcai 2022",
        "ijcai 2021",
        "ijcai 2020",
        "nips 2024",
        "nips 2023",
        "nips 2022",
        "nips 2021",
        "nips 2020",
        # Conference acronyms
        "neurips",
        "icml",
        "iclr",
        "aaai",
        "ijcai",
        "nips",
        # Copyright patterns
        "© 2024",
        "© 2023",
        "© 2022",
        "© 2021",
        "© 2020",
    ]

    try:
        doc = fitz.open(pdf_path)
        redactions_made = False

        max_pages = doc.page_count

        for page_num in range(max_pages):
            page = doc.load_page(page_num)
            page_redactions = False

            for term in redaction_terms:
                # Case-insensitive search for each term
                matches = page.search_for(term)
                for rect in matches:
                    page.add_redact_annot(rect)
                    page_redactions = True
                    redactions_made = True

                # Also search for uppercase version
                matches_upper = page.search_for(term.upper())
                for rect in matches_upper:
                    page.add_redact_annot(rect)
                    page_redactions = True
                    redactions_made = True

                # And title case version
                matches_title = page.search_for(term.title())
                for rect in matches_title:
                    page.add_redact_annot(rect)
                    page_redactions = True
                    redactions_made = True

            # Apply redactions for this page if any were made
            if page_redactions:
                page.apply_redactions()

        # Save to temporary file first, then replace original
        if redactions_made:
            temp_path = pdf_path + ".tmp"
            doc.save(temp_path, garbage=3, deflate=True)
            doc.close()

            # Replace original with redacted version
            os.replace(temp_path, pdf_path)
        else:
            doc.close()

    except Exception as e:
        tqdm.write(f"⚠ Failed to redact {pdf_path}: {str(e)}")


# Create pdfs directory if it doesn't exist
print("Creating pdfs directory...")
os.makedirs("data/raw_pdfs", exist_ok=True)
print("✓ pdfs directory ready")

# Load the local CSV dataset
print("Loading dataset from dataset/selected_100_papers.csv...")
dataset = pd.read_csv("data/analysis/selected_100_papers.csv")
print(f"✓ Loaded dataset with {len(dataset)} papers")

# Filter papers with valid OpenReview submission IDs
valid_papers = dataset[dataset["openreview_submission_id"].notna()]
print(f"✓ Found {len(valid_papers)} papers with valid OpenReview submission IDs")

print("\nStarting parallel PDF downloads and redaction...")

# Prepare download tasks
download_tasks = []
for _, sample in valid_papers.iterrows():
    idd = sample["openreview_submission_id"]
    url = f"https://openreview.net/pdf?id={idd}"
    filename = f"data/raw_pdfs/{idd}.pdf"
    download_tasks.append((idd, url, filename))

# Use ThreadPoolExecutor for parallel downloads and processing
with (
    ThreadPoolExecutor(max_workers=4) as download_executor,
    ThreadPoolExecutor(max_workers=2) as redact_executor,
):
    # Submit download tasks
    download_futures = {
        download_executor.submit(download_pdf, task): task for task in download_tasks
    }

    redact_futures = []
    completed_count = 0

    # Progress bar for overall progress
    with tqdm(total=len(download_tasks), desc="Processing PDFs") as pbar:
        # Process downloads as they complete
        for future in as_completed(download_futures):
            task = download_futures[future]
            idd, result, success = future.result()

            if success:
                filename = result
                tqdm.write(f"✓ Downloaded: {idd}.pdf")

                # Submit redaction task immediately after download
                redact_future = redact_executor.submit(redact_conference_info, filename)
                redact_futures.append((redact_future, idd))

            else:
                error_msg = result
                tqdm.write(f"✗ Failed to download {idd}.pdf: {error_msg}")

            completed_count += 1
            pbar.update(1)

    # Wait for all redaction tasks to complete
    tqdm.write("\nWaiting for redaction tasks to complete...")
    for redact_future, idd in redact_futures:
        try:
            redact_future.result()  # Wait for completion
            tqdm.write(f"✓ Redacted: {idd}.pdf")
        except Exception as e:
            tqdm.write(f"⚠ Failed to redact {idd}.pdf: {str(e)}")

print("\n✓ PDF download and redaction process completed!")
