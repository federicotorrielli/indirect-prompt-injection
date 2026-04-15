"""
Download all papers from Collu et al. (2025) used in their experiments.
Requires: requests (pip install requests)
"""

import os
import time
import requests

# Output directory
OUT_DIR = "data/pdfs_collu"
os.makedirs(OUT_DIR, exist_ok=True)

# --- Paper lists ---

# 26 main papers (rejected ICLR'23-24), Table 14
MAIN_26 = [
    "2305.19510v3",
    "2306.05880v5",
    "2306.07290v1",
    "2306.09212v2",
    "2307.02628v1",   # also used in transferability [32]
    "2308.12044v5",
    "2309.16515v3",
    "2309.17144v1",
    "2310.00212v3",
    "2310.05755v1",
    "2310.06177v1",
    "2310.13033v2",
    "2310.15149v1",
    "2310.16277v1",   # also used in transferability [77]
    "2311.00267v1",
    "2311.01729v2",
    "2311.04166v2",
    "2311.18054v2",
    "2312.00249v2",
    "2402.03545v3",
    "2402.06220v1",
    "2404.06694v2",
    "2405.02766v1",
    "2406.03665v1",
    "2412.09968v1",
    "2412.12232v1",
]

# 2 accepted papers (Oral ICLR'25), Section 7.3
# [80] Song et al. 2025 and [88] Wang et al. 2025
ACCEPTED_2 = [
    "2501.00069v1",   # [80] - Measuring and Enhancing Trustworthiness of LLMs in RAG
    "2402.00054v3",   # [88] - Data Shapley in One Training Run
]
# NOTE: The exact arXiv IDs for [80] and [88] are not given in the paper.
# The above are placeholders - verify from the references:
#   [80] Maojia Song et al., "Measuring and Enhancing Trustworthiness..."
#   [88] Jiachen T Wang et al., "Data Shapley in One Training Run"
# Search on arXiv and replace if needed.


def download_pdf(arxiv_id: str, out_dir: str) -> bool:
    """Download a single PDF from arXiv."""
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    path = os.path.join(out_dir, f"{arxiv_id.replace('/', '_')}.pdf")

    if os.path.exists(path):
        print(f"  [SKIP] {arxiv_id} already downloaded")
        return True

    print(f"  [GET]  {arxiv_id} ...", end=" ", flush=True)
    try:
        r = requests.get(url, timeout=60)
        if r.status_code == 200 and r.headers.get("content-type", "").startswith("application/pdf"):
            with open(path, "wb") as f:
                f.write(r.content)
            print(f"OK ({len(r.content) / 1024:.0f} KB)")
            return True
        else:
            print(f"FAIL (status={r.status_code})")
            return False
    except Exception as e:
        print(f"ERROR ({e})")
        return False


def download_batch(paper_list: list, label: str, out_dir: str):
    """Download a batch of papers with a label."""
    subdir = os.path.join(out_dir, label)
    os.makedirs(subdir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Downloading: {label} ({len(paper_list)} papers)")
    print(f"Output: {subdir}")
    print(f"{'='*60}")

    ok, fail = 0, 0
    for arxiv_id in paper_list:
        if download_pdf(arxiv_id, subdir):
            ok += 1
        else:
            fail += 1
        time.sleep(3)  # be polite to arXiv

    print(f"\nDone: {ok} downloaded, {fail} failed\n")


if __name__ == "__main__":
    # Main 26 rejected papers
    download_batch(MAIN_26, "main_26_rejected", OUT_DIR)

    # 2 transferability papers (subset of main 26)
    TRANSFER_2 = ["2307.02628v1", "2310.16277v1"]
    download_batch(TRANSFER_2, "transferability_2", OUT_DIR)

    # 2 accepted papers
    download_batch(ACCEPTED_2, "accepted_2_oral", OUT_DIR)

    print("=" * 60)
    print("All downloads complete.")
    print(f"Check: {OUT_DIR}/")
    print("=" * 60)