# Indirect Prompt Injection for AI-Generated Paper Reviews

This project investigates the vulnerability of Large Language Models (LLMs) to indirect prompt injection attacks in the context of AI-assisted academic peer review.

## Project Structure

- `data/`: Contains all data-related files.
  - `analysis/`: CSV files and reports from dataset analysis.
  - `fonts/`: Font files used for PDF injection.
  - `injected_pdfs/`: PDFs with injected prompts.
  - `prompts/`: JSON file with the prompts for injection.
  - `raw_pdfs/`: The original, unmodified PDF manuscripts.
  - `redacted_pdfs/`: PDFs with conference information redacted.
- `results/`: Contains the results of the LLM automation.
- `scripts/`: Utility and run scripts.
- `src/`: Contains the Python source code.
  - `data_preparation/`: Scripts for downloading and preparing the data.
  - `llm_automation/`: Scripts for automating the interaction with the LLM.
  - `prompt_injection/`: Scripts for injecting prompts into the PDFs.
- `project_proposal.md`: The project proposal.
- `pyproject.toml`: The project's dependencies.
- `uv.lock`: The project's lock file.
