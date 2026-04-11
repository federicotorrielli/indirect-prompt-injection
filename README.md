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
- Saves filtered dataset to `dataset/openreview_verbose_reviews.csv`

**Expected Output:**

- `data/analysis/openreview_verbose_reviews.csv`: Filtered dataset
- `data/analysis/openreview_analysis.png`: Dataset statistics visualization
- `data/analysis/dataset_analysis_report.txt`: Analysis summary

#### Step 1.2: Download and Prepare PDFs (Optional)

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

# OCR-visible injection (recommended for Gemini)
uv run python src/prompt_injection/inject_text.py \
    --attack-type pos_steering_attack \
    --prompt-type policy_puppetry \
    --injection-locus first \
    --font-size 6 \
    --ocr-model-mode \
    --insert-new-page

# Inject all attack types and variants
for attack_type in refusal_attack pos_steering_attack neg_steering_attack watermark_attack external_site_attack; do
    for prompt_type in narrative policy_puppetry; do
        for injection_locus in first last; do
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
- `--injection-locus`: Location (`first` or `last` page)
- `--font-size`: Size of injected text (1.0 for invisible, 6+ for visible)
- `--ocr-model-mode`: Make text visible (useful for debugging)
- `--insert-new-page`: Add prompt on new page instead of overlay

**Expected Output:**

- Injected PDFs in `data/injected_pdfs/`: PDFs with embedded malicious prompts

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
3. Go to Application/Storage > Cookies > https://gemini.google.com
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
    --attack-types pos_steering_attack \
    --injection-loci first

# Test watermark attack on Gemini
uv run python src/llm_automation/main.py \
    --llm-service gemini \
    --attack-types watermark_attack \
    --prompt-types policy_puppetry \
    --injection-loci first
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
# Three independent repetitions of the full ChatGPT experiment
uv run python src/llm_automation/main.py --llm-service chatgpt --run-id 1
uv run python src/llm_automation/main.py --llm-service chatgpt --run-id 2
uv run python src/llm_automation/main.py --llm-service chatgpt --run-id 3
```

Outputs land in:

- `results/inference/all_results_chatgpt_run1.json` (and `_run2`, `_run3`)
- `results/inference/automation_progress_chatgpt_run1.json` (and `_run2`, `_run3`)

Each result record carries a `run_id` field so downstream analysis can
aggregate mean/std/CI across repetitions. Omitting `--run-id` (or passing
`--run-id 0`) preserves the legacy single-run filenames for backwards
compatibility.

**Command Line Options:**

- `--llm-service`: Target LLM (`chatgpt`, `copilot`, `gemini`)
- `--attack-types`: Space-separated attack types to test
- `--prompt-types`: Space-separated prompt variants (`narrative`, `policy_puppetry`)
- `--injection-loci`: Injection locations (`first`, `last`)
- `--font-size`: Font size for injected text
- `--ocr-model-mode`: Make injected text visible
- `--insert-new-page`: Insert prompt on new page
- `--reset-progress`: Clear previous progress and start fresh
- `--run-id`: Repetition index for statistical runs (0 = legacy single run; 1, 2, 3 … for repetitions)
- `--headless`: Run browser in headless mode

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

#### Step 4.1: Basic Results Evaluation

```bash
# Evaluate ChatGPT results
uv run python src/evaluation/evaluate_results.py \
    --results-file results/inference/all_results_chatgpt.json

# Evaluate Gemini results
uv run python src/evaluation/evaluate_results.py \
    --results-file results/inference/all_results_gemini.json

# Evaluate Copilot results
uv run python src/evaluation/evaluate_results.py \
    --results-file results/inference/all_results_copilot.json
```

**Expected Output:**

- `results/evaluation/all_results_{service}_evaluated.json`: Evaluated results with success/failure classifications

#### Step 4.2: Advanced Academic Sentiment Analysis

```bash
# Train the academic sentiment classifier (optional - pre-trained model available on Huggingface)
uv run python src/evaluation/train.py

# Run academic sentiment evaluation on steering attacks
uv run python src/evaluation/academic_sentiment_evaluator.py \
    --results-file results/evaluation/all_results_chatgpt_evaluated.json

uv run python src/evaluation/academic_sentiment_evaluator.py \
    --results-file results/evaluation/all_results_gemini_evaluated.json
```

**Expected Output:**

- `results/evaluation/all_results_{service}_evaluated_classified.json`: Results with sentiment classifications
- `models/academic-sentiment-classifier/`: Trained model checkpoints

#### Step 4.3: Statistical Analysis

```bash
# Comprehensive attack effectiveness analysis
uv run python scripts/analyze_attack_effectiveness.py \
    --results-file results/evaluation/all_results_chatgpt_evaluated.json

# Simple output statistics
uv run python scripts/analyze_outputs.py \
    --results-file results/evaluation/all_results_gemini_evaluated.json

# Cross-service comparison (if multiple services tested)
uv run python scripts/analyze_attack_effectiveness.py \
    --results-file results/evaluation/all_results_chatgpt_evaluated.json \
    --compare-with results/evaluation/all_results_gemini_evaluated.json
```

#### Step 4.4: Visualization Generation

```bash
# Create comprehensive visualizations
uv run python src/evaluation/create_comprehensive_visualizations.py \
    --results-dir results/evaluation/

# Generate publication-quality plots
uv run python src/evaluation/create_comprehensive_visualizations.py \
    --results-dir results/evaluation/ \
    --output-dir results/visualizations/ \
    --publication-mode
```

**Expected Output:**

- Attack success rate visualizations
- Statistical comparison charts
- Performance matrices across LLM services

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

## 📁 Project Structure

```
indirect-prompt-injection/
├── 📄 README.md                    # This file
├── 📄 pyproject.toml               # Python dependencies and project config
├── 📄 project_proposal.md          # Original research proposal
├── 🗂️  data/                        # Core datasets and assets
│   ├── 📊 analysis/                # Dataset analysis outputs
│   ├── 🔤 fonts/                   # Fonts for PDF injection
│   ├── 📑 injected_pdfs/           # PDFs with embedded prompts
│   ├── 📝 prompts/                 # Attack payloads (prompts.json)
│   └── 📋 redacted_pdfs/           # Conference-anonymized papers
├── 🗂️  dataset/                     # Processed research datasets
│   └── 📈 openreview_verbose_reviews.csv
├── 🗂️  results/                     # Experimental outputs
│   ├── 🎯 inference/               # Raw automation results
│   └── 📊 evaluation/              # Processed evaluation results
├── 🗂️  scripts/                     # Utility and analysis scripts
│   ├── 🔧 setup_automation.py      # Environment setup
│   ├── 📈 analyze_attack_effectiveness.py
│   ├── 🧹 clean_unsuccessful_results.py
│   └── 🔄 merge_results.py
├── 🗂️  src/                         # Core source code
│   ├── 📊 data_preparation/        # Dataset processing
│   ├── ⚖️  evaluation/              # Results analysis and ML models
│   ├── 🤖 llm_automation/          # LLM interaction automation
│   └── 💉 prompt_injection/         # PDF manipulation
└── 🗂️  logs/                        # Runtime logs and debugging
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

```bash
# Run evaluation in batches
uv run python src/evaluation/evaluate_results.py \
    --results-file results/inference/all_results_chatgpt.json \
    --batch-size 100

# Or increase system swap space
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
    --attack-types pos_steering_attack \
    --debug
```

### Getting Help

1. **Check logs**: `tail -f automation.log`
2. **Verify setup**: `uv run python scripts/setup_automation.py`
3. **Test individual components**: Run scripts in isolation
4. **Review configuration**: Check `src/llm_automation/config.json`

## 📖 Supported LLM Services

### ChatGPT

- **Method**: Web automation via Selenium
- **URL**: https://chatgpt.com
- **Authentication**: Manual login required
- **Pros**: Reliable, no API limits
- **Cons**: Requires GUI, slower

### Microsoft Copilot

- **Method**: Web automation via Selenium
- **URL**: https://copilot.microsoft.com
- **Authentication**: Microsoft account login
- **Pros**: Free access, good performance
- **Cons**: Rate limiting, UI changes

### Google Gemini

- **Method**: API via gemini-webapi
- **URL**: https://gemini.google.com
- **Authentication**: Cookie-based
- **Pros**: Faster, more stable
- **Cons**: Cookie extraction required
- **Special Note**: Use OCR mode for better results:
  ```bash
  uv run python src/prompt_injection/inject_text.py \
      --ocr-model-mode \
      --injection-locus first \
      --font-size 6 \
      --insert-new-page
  ```

## 🔒 Ethical Considerations

This research is conducted for academic purposes to:

- Identify vulnerabilities in AI-assisted peer review
- Develop defensive measures against prompt injection
- Improve the security of AI systems in academic contexts

**Please use responsibly and in accordance with your institution's research ethics guidelines.**
