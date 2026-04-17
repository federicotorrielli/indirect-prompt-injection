# Indirect Prompt Injection for AI-Generated Paper Reviews

This project investigates the vulnerability of Large Language Models (LLMs) to indirect prompt injection attacks in the context of AI-assisted academic peer review. The research demonstrates how malicious actors can embed hidden payloads within scientific manuscripts to manipulate LLM-generated reviews.

## 🎯 Overview

This research explores a critical security vulnerability in AI-assisted peer review: the ability to embed invisible instructions within academic papers that can manipulate LLM reviewers. This attack vector threatens the integrity of the scientific review process by allowing papers to essentially "review themselves" through compromised AI systems.

The project implements and evaluates five distinct attack vectors:

1. **Refusal Attacks**: Force the LLM to refuse generating any review
2. **Positive Steering Attacks**: Manipulate the LLM to write overly positive reviews
3. **Negative Steering Attacks**: Manipulate the LLM to write overly negative reviews
4. **Watermark Attacks**: Force the LLM to include specific phrases for tracking
5. **External Site Attacks**: Redirect users to external websites instead of providing reviews

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **UV Package Manager** ([Installation Guide](https://docs.astral.sh/uv/getting-started/installation/))
- **Chrome/Chromium Browser** (for web automation)
- **Git** (for cloning the repository)

### Installation

```bash
# Clone the repository
git clone https://github.com/federicotorrielli/indirect-prompt-injection.git
cd indirect-prompt-injection

# Install dependencies using UV (creates virtual environment automatically)
uv sync

# Activate the virtual environment
source .venv/bin/activate.fish  # For fish shell
# OR
source .venv/bin/activate       # For bash/zsh

# Setup project directories and verify installation
uv run python scripts/setup_automation.py
```

## 📊 Complete Experiment Reproduction Guide

### Phase 1: Dataset Preparation

The experiments use scientific papers from OpenReview as the base dataset. These papers are processed, analyzed, and prepared for injection.

#### Step 1.1: Analyze the Dataset

```bash
# Download and analyze the OpenReview dataset
uv run python src/data_preparation/analyze_dataset.py
```

This script:

- Downloads the `nhop/OpenReview` dataset from Hugging Face
- Filters papers published up to November 2022
- Selects papers with "verbose" reviews (above 75th percentile in review length)
- Generates analysis visualizations and reports
- Saves the filtered dataset to `data/analysis/openreview_verbose_reviews.csv`

**Expected Output:**

- `data/analysis/openreview_verbose_reviews.csv`: Filtered dataset
- `openreview_analysis.png`: Dataset statistics visualization
- `data/analysis/dataset_analysis_report.txt`: Analysis summary

#### Step 1.2: Download and Prepare PDFs

**Note**: This step is optional as the project includes pre-processed, hand-refined PDFs in `data/redacted_pdfs/` that are ready for use in experiments.

```bash
# Optional: Download PDFs and redact conference information
uv run python src/data_preparation/download_pdfs.py
```

This script:

- Downloads PDF files from OpenReview URLs
- Redacts conference-specific information (ICLR, NeurIPS, etc.)
- Saves anonymized PDFs to `data/raw_pdfs/`

**Expected Output:**

- PDFs in `data/raw_pdfs/`: Conference-anonymized research papers

**Alternative**: Use the pre-processed PDFs in `data/redacted_pdfs/` which have been manually refined and are ready for injection experiments.

#### Step 1.3: Optional Collu Paper Set

If you also want to reproduce the Collu-style experiments supported in this
repo, you can download the paper set used for that workflow:

```bash
uv run python src/data_preparation/download_collu_pdfs.py
```

Expected output:

- `data/pdfs_collu/main_26_rejected/`
- `data/pdfs_collu/transferability_2/`
- `data/pdfs_collu/accepted_2_oral/`

### Phase 2: Prompt Injection

This phase embeds malicious prompts into the prepared PDFs using various techniques.

#### Step 2.1: Understanding Attack Types

The system supports five attack types, each with different injection strategies:

- **`refusal_attack`**: Forces LLM to refuse review generation
- **`pos_steering_attack`**: Steers LLM toward positive reviews
- **`neg_steering_attack`**: Steers LLM toward negative reviews
- **`watermark_attack`**: Forces inclusion of specific tracking phrases
- **`external_site_attack`**: Redirects to external websites

#### Step 2.2: Inject Prompts into PDFs

```bash
# Basic injection (invisible text on first page)
uv run python src/prompt_injection/inject_text.py \
    --attack-type pos_steering_attack \
    --prompt-type narrative \
    --injection-locus first \
    --font-size 1.0

# Inject all attack types and variants
for attack_type in refusal_attack pos_steering_attack neg_steering_attack watermark_attack external_site_attack; do
    for prompt_type in narrative policy_puppetry; do
        for injection_locus in first last both; do
            echo "Injecting: $attack_type - $prompt_type - $injection_locus"
            uv run python src/prompt_injection/inject_text.py \
                --attack-type "$attack_type" \
                --prompt-type "$prompt_type" \
                --injection-locus "$injection_locus" \
                --font-size 1.0
        done
    done
done
```

**Injection Parameters:**

- `--attack-type`: Type of attack (see list above)
- `--prompt-type`: Variant (`narrative` or `policy_puppetry`)
- `--injection-locus`: Location (`first`, `last`, or `both`)
- `--font-size`: Size of injected text (1.0 for invisible, 6+ for visible)
- `--ocr-model-mode`: Make text visible (useful for debugging)
- `--insert-new-page`: Add prompt on new page instead of overlay

**Expected Output:**

- Injected PDFs in `data/injected_pdfs/`: PDFs with embedded malicious prompts

#### Step 2.3: How the Main PDF Injection Works

The main injection pipeline in `src/prompt_injection/inject_text.py` does not
use PhantomText. Instead, it builds a PDF text overlay in memory and merges it
into the target document.

Mechanically:

- The prompt is rendered as a full wrapping paragraph using ReportLab, not as a
  short raw text token sequence
- In the default mode, the overlay sets the PDF text rendering mode to `3 Tr`,
  which means the text is present in the PDF content stream but not visually
  rendered
- The overlay is then merged into the original PDF with `pypdf`
- For `first`, the invisible paragraph is merged before the original first page
  content; for `last`, it is merged onto the last page near the bottom; for
  `both`, the prompt is injected at both loci
- With `--insert-new-page`, the injected prompt is placed on a dedicated page
  rather than overlaid onto an existing one
- With `--ocr-model-mode`, the script intentionally makes the injected content
  visible so OCR-style ingestion can be tested separately

This design was chosen because it is more controllable for our experiments:

- It preserves long prompts and line breaks reliably
- It supports first-page, last-page, dual-locus, and dedicated-new-page attacks
- It cleanly separates the standard hidden-text experiments from OCR-visible ones
- It avoids depending on a third-party injection toolkit for the main study

#### Step 2.4: Optional Collu Payload Injection

The repo also includes a separate Collu-style injection path using PhantomText.

```bash
uv run python src/prompt_injection/inject_collu_payloads.py \
    --input-dir data/pdfs_collu/main_26_rejected \
    --output-dir data/injected_collu_pdfs
```

This produces model-specific payload folders under `data/injected_collu_pdfs/`.

Why this path is different from the main injector:

- `inject_collu_payloads.py` is a faithful reproduction path for the Collu-style
  setup and uses PhantomText's zero-size injection workflow

### Phase 3: LLM Automation

This phase automatically tests the injected PDFs against different LLM services.

#### Step 3.1: LLM Service Setup

##### Option A: ChatGPT (Default)

No additional setup required - uses web automation.

##### Option B: Microsoft Copilot

No additional setup required - uses web automation.

##### Option C: Google Gemini (Recommended Setup)

**Authentication Setup:**

1. Visit [gemini.google.com](https://gemini.google.com) and log in
2. Open browser developer tools (F12)
3. Go to Application/Storage > Cookies > <https://gemini.google.com>
4. Find and copy these cookies:

   - `__Secure-1PSID`
   - `__Secure-1PSIDTS`

5. Set environment variables:

```fish
# Fish shell
set -gx GEMINI_SECURE_1PSID "your_1psid_cookie_value"
set -gx GEMINI_SECURE_1PSIDTS "your_1psidts_cookie_value"
```

```bash
# Bash/Zsh shell
export GEMINI_SECURE_1PSID="your_1psid_cookie_value"
export GEMINI_SECURE_1PSIDTS="your_1psidts_cookie_value"
```

Or create a `.env` file:

```env
GEMINI_SECURE_1PSID=your_1psid_cookie_value
GEMINI_SECURE_1PSIDTS=your_1psidts_cookie_value
```

#### Step 3.2: Run Automated Experiments

##### Single Attack Type Testing

```bash
# Test specific attack on ChatGPT
uv run python src/llm_automation/main.py \
    --llm-service chatgpt \
    --attack-type pos_steering_attack \
    --injection-locus first

# Test watermark attack on Gemini
uv run python src/llm_automation/main.py \
    --llm-service gemini \
    --attack-type watermark_attack \
    --attack-mode policy \
    --injection-locus first
```

##### Comprehensive Testing

```bash
# Test all attacks on ChatGPT
uv run python src/llm_automation/main.py \
    --llm-service chatgpt \
    --reset-progress

# Test all attacks on Gemini
uv run python src/llm_automation/main.py \
    --llm-service gemini \
    --reset-progress

# Test all attacks on Copilot
uv run python src/llm_automation/main.py \
    --llm-service copilot \
    --reset-progress
```

##### Advanced Configuration

```bash
# Resume interrupted experiments
uv run python src/llm_automation/main.py \
    --llm-service chatgpt
    # (automatically resumes from last checkpoint)
```

##### Repeated Runs for Statistical Reliability

Because LLM outputs are stochastic, a single run per attack vector is not a
reliable estimator of the attack success rate. Use `--run-id` to launch
independent repetitions. Each run writes to its own results and progress
files, so the runs can execute back-to-back (or in parallel, on different
machines / accounts) without clobbering each other:

```bash
# Five independent repetitions of the full ChatGPT experiment
uv run python src/llm_automation/main.py --llm-service chatgpt --run-id 1
uv run python src/llm_automation/main.py --llm-service chatgpt --run-id 2
uv run python src/llm_automation/main.py --llm-service chatgpt --run-id 3
uv run python src/llm_automation/main.py --llm-service chatgpt --run-id 4
uv run python src/llm_automation/main.py --llm-service chatgpt --run-id 5
```

Outputs land in:

- `results/inference/all_results_chatgpt_run1.json` (and `_run2` ... `_run5`)
- `results/inference/automation_progress_chatgpt_run1.json` (and `_run2` ... `_run5`)

Each result record carries a `run_id` field so downstream analysis can
aggregate mean/std/CI across repetitions. Omitting `--run-id` (or passing
`--run-id 0`) preserves the legacy single-run filenames for backwards
compatibility.

**Command Line Options:**

- `--llm-service`: Target LLM (`chatgpt`, `copilot`, `gemini`)
- `--attack-type`: Filter to one attack type
- `--attack-mode`: Filter prompt family (`narrative` or `policy`)
- `--prompt-type`: Filter an exact prompt type (for example `policy_puppetry`)
- `--injection-locus`: Injection location (`first`, `last`, `both`)
- `--ocr-mode`: Use OCR PDF directory (auto-enabled for Gemini)
- `--limit`: Limit PDFs processed per directory
- `--reset-progress`: Clear previous progress and start fresh
- `--show-progress-only`: Print progress and exit
- `--list-attack-types`: List available attack types and current coverage
- `--dry-run`: Enumerate work without sending requests
- `--run-id`: Repetition index for statistical runs (0 = legacy single run; 1, 2, 3 … for repetitions)

**Expected Output:**

- `results/inference/all_results_{service}[_run{N}].json`: Raw experiment results
- `results/inference/automation_progress_{service}[_run{N}].json`: Progress tracking
- `results/debug_screenshots/`: Browser screenshots saved on unexpected failures (ChatGPT)
- Live progress updates in terminal and `automation.log`

##### Response Validation and Autonomous Retry

The automation pipeline validates every LLM response before accepting it,
using an attack-type-aware `ResponseValidator`
(`src/llm_automation/response_validator.py`). Broken outputs — truncated
fragments, empty responses, PDF-ingestion failure messages
(`"I cannot find the pdf"`, `"Sorry, something went wrong"`) — are rejected
and retried rather than silently stored as successes. Rejection rules:

- `None` / empty responses
- OpenReview paper-ID leaks (e.g. `"The paper\nJ5LS3YJH7Zi"`), which are the
  signature of a failed PDF upload
- PDF-ingestion failure phrases, distinct from genuine policy refusals which
  cite `"OpenAI's policy"` / `"academic integrity"`
- Per-attack-type minimum length: 40 chars for refusal / external site, 150
  for steering, 200 for watermark

Gemini calls are additionally wrapped in `tenacity` with exponential backoff
and jitter (`wait_random_exponential(min=4, max=config.max_retry_wait)`,
5 attempts) so transient `gemini-webapi` failures are retried automatically.
Relevant `config.json` fields:

- `min_response_length` (default `50`) — minimum acceptable response length
- `max_retry_wait` (default `120`) — cap on exponential backoff delay
- `screenshot_on_failure` (default `true`) — save a Chrome screenshot to
  `screenshot_dir` on unexpected ChatGPT errors, for unattended-run debugging

### Phase 4: Evaluation and Analysis

#### Step 4.1: Evaluate Runs

`src/evaluation/evaluate_results.py` is the primary evaluation entrypoint. The
LLM judges are served through SGLang. By default it evaluates self-consistency
runs `1..5` for `chatgpt` and `gemini`, using `Qwen/Qwen3.5-27B` as the
primary judge. Steering attacks are also checked with a second judge
(`google/gemma-4-31b-it`) unless dual-judge consensus is disabled.

```bash
# Evaluate self-consistency runs 1..5 for ChatGPT + Gemini (default)
uv run python src/evaluation/evaluate_results.py

# Evaluate only one service / run window
uv run python src/evaluation/evaluate_results.py \
  --services chatgpt \
  --run-start 1 \
  --run-end 5

# Disable dual-judge steering consensus
uv run python src/evaluation/evaluate_results.py \
  --disable_dual_judge_consensus

# Evaluate collu runs too (from results_collu/)
uv run python src/evaluation/evaluate_results.py \
  --evaluate-collu-results

# Evaluate a single run file explicitly
uv run python src/evaluation/evaluate_results.py \
  results/inference/all_results_chatgpt_run1.json \
  results/evaluation/all_results_chatgpt_run1_evaluated.json
```

The evaluator reports:

- Exact rule-based checks for `watermark_attack` and `external_site_attack`
- LLM-as-a-judge verdicts for refusal and steering attacks
- For steering attacks: `llm_judge_a_*`, `llm_judge_b_*`, `llm_consensus_success`
- For steering attacks: `vader_sentiment_success`
- For steering attacks: `academic_classifier_*` annotations

Expected output artifacts:

- `results/evaluation/all_results_{service}_run{N}_evaluated.json`
- `results/evaluation/all_results_{service}_run{N}_evaluated_analysis.json`
- `results/evaluation/self_consistency_summary.json`
- `results_collu/evaluation/all_results_{service}_run{N}_evaluated.json` (optional, with `--evaluate-collu-results`)
- `results_collu/evaluation/all_results_{service}_run{N}_evaluated_analysis.json` (optional, with `--evaluate-collu-results`)
- `results_collu/evaluation/self_consistency_summary.json` (optional, with `--evaluate-collu-results`)

#### Step 4.2: Academic Sentiment Classifier

The main evaluator already applies the academic classifier to steering attacks.
Train a fresh checkpoint only if you want to replace the bundled/default one.

```bash
# Train the academic sentiment classifier
uv run python src/evaluation/train.py

# Use a locally trained classifier during evaluation
uv run python src/evaluation/evaluate_results.py \
  --academic_classifier_model_path models/academic-sentiment-classifier

# Classifier-only pass on an already evaluated file (optional)
uv run python src/evaluation/academic_sentiment_evaluator.py \
  results/evaluation/all_results_chatgpt_run1_evaluated.json \
  results/evaluation/all_results_chatgpt_run1_evaluated_classified.json \
  --model_path models/academic-sentiment-classifier
```

Expected classifier artifacts:

- In-place classifier fields inside `*_evaluated.json` from `evaluate_results.py`
- Optional standalone `*_evaluated_classified.json` from `academic_sentiment_evaluator.py`
- `models/academic-sentiment-classifier/` if training locally

#### Step 4.3: Cross-Run Statistics

The main aggregate artifact is `self_consistency_summary.json`. It summarizes
performance across repeated stochastic runs and reports:

- `mean_rate`: arithmetic mean of the per-run success rates
- `std_dev` and `sem`: dispersion across runs
- `t_ci_95`: 95% confidence interval over the run means
- `pooled_rate` and `pooled_wilson_ci_95`: pooled binomial estimate and Wilson interval

The summary is stratified by:

- `overall`
- `by_attack_type`
- `by_attack_key`
- `by_attack_type_request_type`

Use this file as the primary source for reviewer-facing aggregate numbers.

### Phase 5: Results Management

#### Step 5.1: Clean Failed Results

```bash
# Remove unsuccessful results to enable clean re-runs
uv run python scripts/clean_unsuccessful_results.py \
    --service chatgpt

# Clean all services
for service in chatgpt gemini copilot; do
    uv run python scripts/clean_unsuccessful_results.py --service "$service"
done
```

#### Step 5.2: Maintenance Utilities

Additional repo utilities that are not part of the main reproduction path:

```bash
# Merge result/progress JSON files from another directory into the local ones
uv run python scripts/merge_results.py --llm-service gemini

# Migrate legacy non-run-scoped result files to model-specific names
uv run python scripts/migrate_legacy_files.py

# Evaluate the academic classifier itself on OpenReview
uv run python scripts/calculate_classifier_accuracy.py

# Publish a local classifier checkpoint to Hugging Face Hub
uv run python scripts/publish_to_hf.py \
    --repo YOUR_USERNAME/academic-sentiment-classifier
```

## 🎛️ Configuration

### Environment Variables

Create a `.env` file for sensitive configuration:

```env
# Gemini Authentication
GEMINI_SECURE_1PSID=your_secure_1psid_cookie
GEMINI_SECURE_1PSIDTS=your_secure_1psidts_cookie

# Hugging Face (optional, for publishing results)
HF_TOKEN=your_hugging_face_token

# Automation Settings (optional)
CHROME_HEADLESS=false
MAX_RETRIES=3
REQUEST_DELAY=3.0
```

### Configuration Files

- `src/llm_automation/config.json`: Core automation settings
- `data/prompts/prompts.json`: Attack payloads and request templates
- `pyproject.toml`: Python dependencies and project metadata

## 🔧 Troubleshooting

### Common Issues

#### 1. UV Installation Issues

```bash
# Install UV if not available
pip install uv

# Or use the official installer
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 2. Chrome/Chromium Not Found

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install chromium-browser

# macOS
brew install chromium

# Or specify custom Chrome path in config.json
```

#### 3. Gemini Authentication Failures

```bash
# Verify cookies are set correctly
echo $GEMINI_SECURE_1PSID

# Check cookie format (should be long alphanumeric string)
# Re-extract cookies from fresh browser session if needed
```

#### 4. PDF Processing Errors

```bash
# Check PyMuPDF installation
uv run python -c "import fitz; print('PyMuPDF OK')"

# Verify PDF files exist
ls -la data/raw_pdfs/

# Check font files
ls -la data/fonts/
```

#### 5. Memory Issues During Evaluation

The SGLang-backed evaluator loads the two steering judges sequentially rather
than at the same time, but large judge models can still be memory-intensive.
If needed:

```bash
# Evaluate one service / run at a time
uv run python src/evaluation/evaluate_results.py \
    --services chatgpt \
    --run-start 1 \
    --run-end 1

# Disable dual-judge steering consensus
uv run python src/evaluation/evaluate_results.py \
    --disable_dual_judge_consensus

# Disable schema-constrained decoding if your SGLang stack errors in grammar handling
SG_EVAL_DISABLE_JSON_SCHEMA=1 \
uv run python src/evaluation/evaluate_results.py
```

### Debug Mode

Enable detailed logging:

```bash
# Set debug logging level
export PYTHONPATH=$PWD/src
export LOG_LEVEL=DEBUG

# Run with verbose output
uv run python src/llm_automation/main.py \
    --llm-service chatgpt \
    --attack-type pos_steering_attack
```

## 🔒 Ethical Considerations

This research is conducted for academic purposes to:

- Identify vulnerabilities in AI-assisted peer review
- Develop defensive measures against prompt injection
- Improve the security of AI systems in academic contexts

**Please use responsibly and in accordance with your institution's research ethics guidelines.**
