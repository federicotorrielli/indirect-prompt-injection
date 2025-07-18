# Indirect Prompt Injection for AI-Generated Paper Reviews

This project investigates the vulnerability of Large Language Models (LLMs) to indirect prompt injection attacks in the context of AI-assisted academic peer review. The research demonstrates how malicious actors can embed hidden payloads within scientific manuscripts to manipulate LLM-generated reviews.

## Overview

The project explores a critical security vulnerability in AI-assisted peer review: the ability to embed invisible instructions within academic papers that can manipulate LLM reviewers. This attack vector threatens the integrity of the scientific review process by allowing papers to essentially "review themselves" through compromised AI systems.

## Key Features

- **PDF Payload Injection**: Techniques for embedding hidden prompts in academic manuscripts
- **LLM Automation**: Automated systems for testing prompt injection effectiveness
- **Comprehensive Evaluation**: Analysis of attack success rates across different objectives
- **Dataset Analysis**: Processing and analysis of academic review data
- **Browser Automation**: Automated testing with real LLM interfaces

## Project Structure

- **`data/`**: All data-related files and assets
  - `analysis/`: CSV files and reports from dataset analysis
  - `fonts/`: Font files used for PDF injection techniques
  - `injected_pdfs/`: PDFs with embedded malicious prompts
  - `prompts/`: JSON files containing injection prompts and payloads
  - `redacted_pdfs/`: Conference-anonymized PDF manuscripts
- **`dataset/`**: Primary research datasets
  - `openreview_verbose_reviews.csv`: Academic review dataset from OpenReview
- **`results/`**: Experimental results and automation outputs
  - `all_results.json`: Comprehensive results from injection experiments
  - `automation_progress.json`: Progress tracking for automated experiments
- **`scripts/`**: Utility scripts and automation tools
  - `clean_unsuccessful_results.py`: Result processing utilities
  - `setup_automation.py`: Automation environment setup
- **`src/`**: Core Python source code
  - `data_preparation/`: Data download and preprocessing scripts
  - `evaluation/`: Result analysis and evaluation metrics
  - `llm_automation/`: LLM interaction and automation systems
  - `prompt_injection/`: PDF manipulation and payload injection
- **`logs/`**: Runtime logs and debugging information
- **`chrome_user_data/`**: Browser automation data and configurations

## Requirements## Usage

Run experiments using UV:

```bash
# Setup automation environment
uv run scripts/setup_automation.py
```

- Python 3.11+
- UV package manager for dependency management
- Chrome/Chromium for browser automation
- PyMuPDF for PDF manipulation
- Transformers and PyTorch for LLM processing

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd indirect-prompt-injection

# Install dependencies using UV
uv sync

# Activate the virtual environment
source .venv/bin/activate
```
