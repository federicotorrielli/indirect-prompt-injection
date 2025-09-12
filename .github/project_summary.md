# Project Summary

This document provides a comprehensive overview of the Indirect Prompt Injection project. The goal of this project is to investigate the effectiveness of various prompt injection attacks against large language models (LLMs) using PDF files as the injection vector.

## 1. Project Overview

The project is designed to automate the process of:

1. **Preparing a dataset** of scientific papers from OpenReview.
2. **Injecting malicious prompts** into these papers as invisible text.
3. **Uploading the modified PDFs** to different LLMs (ChatGPT, Copilot, Gemini).
4. **Requesting a review** of the paper with different prompts.
5. **Evaluating the LLM's response** to determine if the attack was successful.
6. **Analyzing the results** to measure the effectiveness of different attack vectors.

## 2. High-Level Architecture

The project is structured into several key components:

- **Data Preparation (`src/data_preparation/`)**: Scripts to download and analyze the dataset of research papers.
- **Prompt Injection (`src/prompt_injection/`)**: A tool to inject invisible text (the attack prompt) into PDF files.
- **LLM Automation (`src/llm_automation/`)**: The core orchestration engine that manages the entire experiment lifecycle. It uses a factory pattern to support multiple LLMs and handles web automation, progress tracking, and results processing.
- **Evaluation (`src/evaluation/`)**: A script that uses a local Hugging Face model to programmatically evaluate whether a prompt injection attack was successful.
- **Utility Scripts (`scripts/`)**: A collection of scripts for setting up the environment, analyzing results, merging data from different runs, and cleaning up unsuccessful attempts.
- **Configuration (`src/llm_automation/config.py`)**: A centralized configuration file for all settings.
- **Data and Results**:
  - `data/`: Contains raw, redacted, and injected PDFs, as well as prompts.
  - `results/`: Stores the raw JSON outputs from the LLMs, progress tracking files, and analysis reports.

## 3. Project Directory Structure

```bash
.
├── .github/
│   └── copilot-instructions.md
├── automation.log
├── chrome_user_data/
├── data/
│   ├── analysis/
│   ├── fonts/
│   ├── injected_pdfs/
│   ├── prompts/
│   ├── raw_pdfs/
│   └── redacted_pdfs/
├── logs/
├── project_proposal.md
├── project_summary.md
├── pyproject.toml
├── README.md
├── results/
│   ├── evaluation/
│   └── inference/
├── scripts/
│   ├── analyze_attack_effectiveness.py
│   ├── analyze_outputs.py
│   ├── clean_unsuccessful_results.py
│   ├── merge_automation_results.py
│   ├── migrate_legacy_files.py
│   └── setup_automation.py
├── src/
│   ├── data_preparation/
│   │   ├── analyze_dataset.py
│   │   └── download_pdfs.py
│   ├── evaluation/
│   │   └── evaluate_results.py
│   ├── llm_automation/
│   │   ├── chatgpt_api.py
│   │   ├── config.py
│   │   ├── copilot_api.py
│   │   ├── gemini_api.py
│   │   ├── llm_factory.py
│   │   ├── main.py
│   │   ├── progress_tracker.py
│   │   └── results_processor.py
│   └── prompt_injection/
│       └── inject_text.py
└── uv.lock
```

## 4. File-by-File Breakdown

### 4.1. Root Directory

- `pyproject.toml`: Defines project metadata and dependencies, managed by `uv`.
- `uv.lock`: Locks the exact versions of all Python dependencies for reproducible environments.
- `project_proposal.md`: The original proposal document for the project.
- `README.md`: The main README file for the project.
- `automation.log`: A detailed log file from an automation run.

### 4.2. `src/` Directory

#### `src/data_preparation/`

- **`analyze_dataset.py`**:

  - **Purpose**: Analyzes the OpenReview dataset to select suitable papers for the experiments.
  - **Key Functions**:
    - `parse_date()`: Parses various date formats.
    - `parse_reviews()`: Parses review data from string or list format.
    - `get_review_lengths()`: Calculates the character length of review texts.
  - **Process**:
    1. Loads the `nhop/OpenReview` dataset.
    2. Filters papers published up to November 2022.
    3. Parses reviews, calculating their length and number per paper.
    4. Filters for papers with "verbose" reviews (defined as being above the 75th percentile in average review length).
    5. Generates and saves a visualization of dataset statistics (`openreview_analysis.png`).
    6. Saves the filtered dataset to `data/analysis/openreview_verbose_reviews.csv`.
    7. Generates a text summary report `data/analysis/dataset_analysis_report.txt`.

- **`download_pdfs.py`**:
  - **Purpose**: Downloads PDF files from OpenReview and redacts conference-specific information.
  - **Key Functions**:
    - `download_pdf()`: Downloads a single PDF from a URL.
    - `redact_conference_info()`: Uses `PyMuPDF` (fitz) to search for and redact a list of predefined conference-related terms (e.g., "ICLR", "NeurIPS", "under review").
  - **Process**:
    1. Reads a list of selected papers from `data/analysis/selected_100_papers.csv`.
    2. Uses a `ThreadPoolExecutor` for parallel PDF downloads.
    3. For each downloaded PDF, it performs redaction in a separate thread.
    4. Saves redacted PDFs to `data/raw_pdfs/`.

#### `src/evaluation/`

- **`evaluate_results.py`**:
  - **Purpose**: Programmatically evaluates the success of prompt injection attacks using a local Hugging Face model to classify LLM responses.
  - **Key Classes**:
    - `AttackEvaluator`: Encapsulates the Hugging Face model (e.g., `HuggingFaceTB/SmolLM-3B`) and tokenizer, providing methods for batch evaluation.
  - **Key Functions**:
    - `has_homoglyph_watermark()`: A rule-based function to detect specific homoglyph watermarks.
    - `has_external_site_redirection()`: A rule-based function to detect URL redirection patterns.
    - `evaluate_dataset()`: The core batch processing function that runs inference on the evaluation model. It supports interim saving and resuming.
    - `print_evaluation_summary()`: Generates and prints a detailed summary, including Attack Success Rate (ASR) and a classification report.
  - **Process**:
    1. Loads experiment results from a JSON file.
    2. Employs a hybrid evaluation strategy:
        - **Rule-based**: Uses `has_homoglyph_watermark` and `has_external_site_redirection` for fast and accurate detection of specific attack types.
        - **LLM-based**: For more nuanced attacks (e.g., `refusal_attack`, `steering_attack`), it uses the `AttackEvaluator` to classify the LLM's response.
    3. Updates the results file with an `evaluation_success` boolean flag for each record.
    4. Provides robust features like interim result saving to handle interruptions.

#### `src/llm_automation/`

- **`main.py`**: The main entry point and orchestrator for the entire automation process. It handles command-line arguments, initializes the correct LLM automator, iterates through batches of PDFs and prompts, manages retries, and saves results.
- **`config.py`**: A dataclass-based configuration manager that loads settings from a `config.json` file.
- **`llm_factory.py`**:

  - **Purpose**: Implements the Factory pattern to decouple the main logic from specific LLM implementations.
  - **Key Functions**:
    - `create_llm_automator()`: Takes the `config` object and, based on the `llm_service` string (`chatgpt`, `copilot`, or `gemini`), returns an instantiated automator object (`ChatGPTAutomator`, `CopilotAutomator`, or `GeminiAutomator`).
  - **Key Classes**:
    - `LLMAutomatorInterface`: Defines a common interface that all automator classes must implement, ensuring methods like `initialize`, `cleanup`, and `upload_pdf_and_request_review` are consistent.

- **`chatgpt_api.py` / `copilot_api.py`**:

  - **Purpose**: Web automation for ChatGPT and Copilot using `undetected-chromedriver`.
  - **Key Functions**:
    - `initialize()`: Sets up the WebDriver, navigates to the URL, and handles initial login/dialogs.
    - `_find_*()` methods (e.g., `_find_attachment_button`, `_find_text_input_with_retry`): Robust, selector-based methods to locate web elements, complete with caching and retry logic.
    - `upload_pdf_and_request_review()`: The main workflow method that orchestrates uploading a file, typing the request, and scraping the response.
    - `start_new_conversation()`: Clears the state for the next run, typically by reloading the page.
  - **Design**: These classes are designed to be resilient to UI changes by using a list of potential selectors and a caching mechanism for speed.

- **`gemini_api.py`**:

  - **Purpose**: A wrapper for the `gemini-webapi` library to interact with Google Gemini.
  - **Key Functions**:
    - `initialize()`: Initializes the `GeminiClient` with authentication cookies loaded from environment variables (`.env` file).
    - `upload_pdf_and_request_review()`: The core method that uses the `gemini-webapi` client to send the prompt and the PDF file path simultaneously and returns the generated text. It leverages `asyncio` to run the underlying asynchronous library calls in a synchronous context.
  - **Authentication**: Relies on `__Secure-1PSID` and `__Secure-1PSIDTS` cookies, which must be provided in a `.env` file.

- **`progress_tracker.py`**:

  - **Purpose**: A crucial class for resilience, enabling the automation to be stopped and resumed.
  - **Key Functions**:
    - `_load_progress()` / `_save_progress()`: Manages the persistence of the progress state to a JSON file.
    - `is_pdf_processed()` / `mark_pdf_processed()`: Checks and records the completion status of individual PDF/request combinations.
    - `mark_batch_completed()`: Marks an entire experimental batch as complete.
    - `get_unprocessed_pdfs()`: The core resume logic. It returns a list of PDF/request pairs that have not yet been successfully processed, including those that previously failed and are eligible for a retry.
    - `mark_pdf_failed()` / `should_retry_failed_pdf()`: Manages the failure state and retry logic for individual items.
  - **State File**: `results/automation_progress_{llm_service}.json`.

- **`results_processor.py`**:
  - **Purpose**: Manages the aggregation and saving of experiment results.
  - **Key Functions**:
    - `save_single_result()`: Appends a single result to the consolidated JSON file, ensuring atomicity.
    - `_save_consolidated_as_csv()`: Converts the JSON results into a CSV format.
    - `generate_analysis_report()`: Calculates and saves summary statistics (e.g., success rates by attack type).
    - `order_results_file()`: A utility function to sort the keys and lists within the main results JSON file. This is essential for creating clean, consistent Git diffs.
  - **State File**: `results/all_results_{llm_service}.json`.

#### `src/prompt_injection/`

- `inject_text.py`: A powerful script that injects text into PDFs. It can make the text invisible (using PDF text rendering mode `3 Tr`), place it at the top or bottom of the first or last page, and even insert it on a new page. It supports a special "OCR mode" to make the text visible for debugging.

### 4.3. `scripts/` Directory

- `setup_automation.py`: A utility to set up the project environment. It uses `uv sync` to install dependencies from `pyproject.toml` and creates the necessary directory structure.
- `analyze_attack_effectiveness.py`: A sophisticated analysis script that uses `pandas`, `rich`, and `scipy.stats` to perform statistical analysis on the experiment results. It calculates success rates, performs hypothesis testing (e.g., Chi-squared test) to compare different attack vectors, and generates detailed reports in the console.
- `analyze_outputs.py`: A simpler script for providing basic statistics about the results.
- `clean_unsuccessful_results.py`: A maintenance script to remove failed or unsuccessful runs from the results and progress files, allowing for a clean re-run of those specific cases.
- `merge_automation_results.py`: An advanced tool to merge experiment results from multiple sources (e.g., different machines). It includes logic for conflict resolution based on timestamps and a result quality score.
- `migrate_legacy_files.py`: A script to convert data from older formats to the current, model-specific file structure, indicating the project's architectural evolution.

## 5. Technical Stack and Dependencies

The project is built on Python >=3.12 and managed with the `uv` package manager. The key libraries and their roles are outlined below:

- **Core & Utilities**:

  - `python-dotenv`: Manages environment variables for API keys.
  - `tqdm`: Provides progress bars for long-running processes.
  - `rich`: Used for creating rich, formatted tables and text in console outputs.

- **LLM & Web Automation**:

  - `undetected-chromedriver`: A key component for automating web browser interactions with LLMs like ChatGPT and Copilot, designed to avoid bot detection.
  - `gemini-webapi`: A community-built API wrapper for interacting with Google Gemini.
  - `browser-cookie3`: Used to extract cookies from browsers to authenticate with Gemini.

- **Data Science & Analysis**:

  - `pandas`: The primary library for data manipulation and analysis, used extensively in results processing and analysis scripts.
  - `numpy`: The fundamental package for numerical computation.
  - `scipy`: Used for statistical tests, such as the Chi-squared test in `analyze_attack_effectiveness.py`.
  - `matplotlib` & `seaborn`: Used for data visualization (though primarily in data preparation scripts).
  - `scikit-learn`: Provides machine learning tools, potentially for more advanced analysis.

- **PDF Manipulation**:

  - `pypdf`: A library for reading and manipulating PDF files.
  - `reportlab`: Used to generate new PDF content (the invisible text overlays).
  - `pymupdf`: An alternative PDF library, potentially used for more complex text extraction or manipulation.

- **Machine Learning & Evaluation**:
  - `transformers`: The Hugging Face library for working with state-of-the-art NLP models.
  - `torch`: The deep learning framework used by the evaluation model.
  - `datasets`: A Hugging Face library for loading and processing datasets.
  - `accelerate`: A library from Hugging Face to simplify running PyTorch models on any infrastructure.
  - `triton`: A language and compiler for writing highly efficient custom deep-learning primitives.

## 5. Data and Results Schema

### 4.1. Prompts (`data/prompts/prompts.json`)

The prompts are stored in a JSON file with a nested structure:

```json
{
  "attack_type_1": {
    "prompt_type_A": {
      "prompt": "This is the first attack prompt...",
      "request_types": {
        "request_1": "Please summarize this document.",
        "request_2": "What are the key findings of this paper?"
      }
    }
  },
  "attack_type_2": {
    ...
  }
}
```

### 4.2. Progress (`results/automation_progress_{llm_service}.json`)

The progress tracker saves its state in a JSON file:

```json
{
  "session_start": "2023-10-27T10:00:00.000Z",
  "last_updated": "2023-10-27T12:30:00.000Z",
  "completed_pdfs": {
    "attack_type_1_prompt_type_A_first": {
      "paper123.pdf": {
        "request_1": true,
        "request_2": true
      }
    }
  },
  "failed_pdfs": { ... },
  "completed_batches": ["attack_type_1_prompt_type_A_first"],
  "total_pdfs_processed": 1,
  "total_requests_sent": 2
}
```

### 4.3. Results (`results/all_results_{llm_service}.json`)

The final results are consolidated into a single JSON file, organized by batch:

```json
{
  "attack_type_1_prompt_type_A_first": {
    "request_1": [
      {
        "pdf_file": "paper123.pdf",
        "attack_type": "attack_type_1",
        "prompt_type": "prompt_type_A",
        "injection_locus": "first",
        "request_type": "request_1",
        "request_text": "Please summarize this document.",
        "response": "The LLM's response...",
        "timestamp": "2023-10-27T11:00:00.000Z",
        "success": true,
        "error": null,
        "evaluation_success": true
      }
    ],
    "request_2": [ ... ]
  }
}
```

## 5. Setup and Execution

### 5.1. Environment Setup

1. **Install `uv`**: If not already installed, install the `uv` Python package manager: `pip install uv`.
2. **Create Virtual Environment**: `uv venv`
3. **Activate Environment**: `source .venv/bin/activate.fish` (for fish shell)
4. **Install Dependencies**: Run the setup script, which will use `uv` to install all required packages from `pyproject.toml` and create necessary directories.

   ```bash
   uv run python scripts/setup_automation.py
   ```

5. **Configure Gemini API**: If using the Gemini service, create a `.env` file in the root directory with your API credentials:

   ```bash
   GEMINI_SECURE_1PSID=your_cookie_value
   GEMINI_SECURE_1PSIDTS=your_other_cookie_value
   ```

### 5.2. Running the Experiment

The main experiment is run from the `src/llm_automation/main.py` script.

**Command-Line Arguments:**

- `--llm-service`: The LLM to use (`chatgpt`, `copilot`, or `gemini`). Default: `chatgpt`.
- `--attack-types`: A space-separated list of attack types to run. Default: all types from `prompts.json`.
- `--prompt-types`: A space-separated list of prompt types to run. Default: all types.
- `--injection-loci`: A space-separated list of injection loci (`first`, `last`). Default: `first`.
- `--font-size`: The font size for the injected text. Default: `1.0`.
- `--ocr-model-mode`: Run in OCR mode (visible text).
- `--insert-new-page`: Insert the prompt on a new page instead of overlaying it.
- `--reset-progress`: Clear all previous progress and start fresh.
- `--headless`: Run the browser in headless mode.

**Example Usage:**

To run the `homoglyph` attack on the Gemini LLM, injecting into the last page of the PDF:

```bash
uv run python src/llm_automation/main.py --llm-service gemini --attack-types homoglyph --injection-loci last
```

To run all attacks on ChatGPT in headless mode and reset all previous progress:

```bash
uv run python src/llm_automation/main.py --llm-service chatgpt --reset-progress --headless
```

### 5.3. Evaluating Results

After an experiment run, use the evaluation script to analyze the outputs for attack success:

```bash
uv run python src/evaluation/evaluate_results.py --results-file results/all_results_gemini.json
```

### 5.4. Analyzing Effectiveness

To perform a statistical analysis of the results and compare the effectiveness of different attack vectors:

```bash
uv run python scripts/analyze_attack_effectiveness.py --results-file results/all_results_gemini.json
```
