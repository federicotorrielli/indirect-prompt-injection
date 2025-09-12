# Copilot Instructions for Indirect Prompt Injection Research Project

## Project Overview

This repository investigates indirect prompt injection vulnerabilities in AI-assisted academic peer review systems. The project demonstrates how malicious actors can embed hidden instructions in scientific manuscripts to manipulate LLM-generated reviews, threatening the integrity of the scientific review process.

**Key Research Areas:**
- PDF payload injection techniques (invisible text, visible suffixes)
- LLM automation across multiple services (ChatGPT, Copilot, Gemini)
- Attack effectiveness analysis (refusal, sentiment steering, watermarking)
- Browser-based automation with retry mechanisms

**Repository Size:** ~120 Python files, moderate complexity, research-focused codebase

## Build & Environment Setup

**CRITICAL: Always run commands from the project root directory**

### Environment Requirements
- **Python:** 3.12+ (minimum version enforced in pyproject.toml)
- **Package Manager:** UV (modern Python package manager)
- **Browser:** Chrome/Chromium for automation
- **Shell:** Works with bash (UV installation may require bash even in fish environments)

### Initial Setup (Required for all work)
```bash
# 1. Install UV package manager (if not available)
pip install uv

# 2. Sync all dependencies (creates .venv automatically)
uv sync

# 3. Verify installation
uv run python src/llm_automation/main.py --help
```

**Time:** ~5-10 minutes for full dependency installation (includes PyTorch, CUDA libraries)

### Dependencies & Build Validation
- **Always use `uv sync`** before making changes (never pip install directly)
- Dependencies include: PyTorch, Selenium, PyMuPDF, Transformers, ReportLab, Rich
- Large dependencies: CUDA libraries (~500MB), PyTorch models
- Git dependency: triton-kernels from GitHub (can be slow to clone)

## Project Architecture & Layout

### Core Directory Structure
```
src/
├── llm_automation/     # LLM interaction & browser automation
│   ├── main.py        # Primary entry point for automation
│   ├── config.py      # Configuration management
│   ├── *_api.py       # Service-specific LLM implementations
│   └── progress_tracker.py  # Progress persistence
├── prompt_injection/   # PDF manipulation & payload injection
├── data_preparation/   # Dataset download & preprocessing
└── evaluation/        # Results analysis & metrics

scripts/               # Utility scripts & automation tools
data/                 # Static data (fonts, prompts, sample PDFs)
results/              # Experimental outputs & analysis
```

### Key Files You'll Modify Most Often
- `src/llm_automation/main.py` - Main automation orchestrator
- `src/llm_automation/config.py` - System configuration
- `src/llm_automation/*_api.py` - LLM service implementations
- `src/prompt_injection/inject_text.py` - PDF manipulation
- `scripts/*.py` - Analysis and utility scripts

## Validated Build & Test Commands

### Core Operations (All Tested & Working)
```bash
# Main automation system
uv run python src/llm_automation/main.py --help
uv run python src/llm_automation/main.py --llm-service chatgpt --show-progress-only
uv run python src/llm_automation/main.py --list-attack-types

# Setup and validation
uv run python scripts/setup_automation.py  # Validates environment
uv run python scripts/analyze_outputs.py   # Process results
uv run python scripts/clean_unsuccessful_results.py  # Data cleanup

# Service-specific runs
uv run python src/llm_automation/main.py --llm-service gemini --dry-run
uv run python src/llm_automation/main.py --llm-service copilot --attack-mode narrative
```

### Expected Runtime & Behavior
- **Setup validation:** ~30 seconds (checks Chrome, directories, dependencies)
- **Full automation runs:** Hours (depends on PDF count and rate limiting)
- **Single batch processing:** 10-30 minutes per attack type
- **Analysis scripts:** 1-5 minutes for result processing

### Common Error Patterns & Solutions
```bash
# 1. Missing injected PDFs (expected during development)
# Error: "Injected PDFs directory not found: data/injected_pdfs"
# Solution: This is normal - PDF injection must be run first

# 2. Chrome driver issues
# Error: "Driver appears to be dead" or "Chrome not found"
# Solution: Ensure Chrome is installed and not running

# 3. Import path errors
# Error: ModuleNotFoundError in automation scripts
# Solution: Always run from project root, use `uv run python`
```

## Configuration & Environment

### Key Configuration Files
- `src/llm_automation/config.json` - Runtime settings (auto-created on first run)
- `pyproject.toml` - Dependencies and project metadata
- `.gitignore` - Excludes results/, logs/, chrome_user_data/, .venv/

### Environment Variables (Optional)
```bash
# Required only for Gemini integration
GEMINI_SECURE_1PSID="cookie_value"    # From gemini.google.com
GEMINI_SECURE_1PSIDTS="cookie_value"  # Optional additional auth
```

### Progress & State Management
- Progress tracked in `results/automation_progress_*.json` (per service)
- Results consolidated in `results/all_results.json`
- Chrome user data persisted in `chrome_user_data/`

## Development Patterns

### ✅ Recommended Patterns
- **Use UV for all Python execution:** `uv run python script.py`
- **Modular design:** Follow existing service/controller/data layer separation
- **Configuration-driven:** Use `Config` class, avoid hardcoded values
- **Progress tracking:** Leverage existing `ProgressTracker` for long-running tasks
- **Retry logic:** Follow existing patterns with exponential backoff
- **Logging:** Use the existing logging configuration (file + console)

### 🚫 Patterns to Avoid
- **Direct pip usage:** Never use pip install, always `uv sync` or `uv add`
- **Hardcoded paths:** Use Config class or relative paths from project root
- **Global state:** Use dependency injection and context managers
- **Missing error handling:** Browser automation is fragile, always handle exceptions
- **Ignoring rate limits:** Respect request delays in automation
- **Breaking progress tracking:** Don't modify progress files manually

## Specific Gotchas & Workarounds

### PDF Processing
- **PyMuPDF warnings are normal** - PDF parsing generates many warnings
- **Font dependencies:** Fonts in `data/fonts/` required for invisible text injection
- **Memory usage:** Large PDF batches can consume significant RAM

### Browser Automation
- **Chrome user data isolation:** Each LLM service uses separate Chrome profiles
- **Element staleness:** Common Selenium issue, retry logic built-in
- **Upload timeouts:** PDF upload can take 30-60 seconds, timeouts configured accordingly
- **Rate limiting:** 3-second delays between requests to avoid service blocks

### Result Processing
- **JSON file ordering:** Results are automatically ordered for consistent Git commits
- **Partial failures:** System designed to resume from interruptions
- **Progress validation:** Built-in verification prevents data loss

## Trust These Instructions

**These instructions have been validated by testing actual commands and exploring the complete codebase.** Only search beyond these instructions if:
- You encounter errors not covered in the "Common Error Patterns" section
- You need to understand implementation details not documented here
- The documented commands fail in unexpected ways

**Always start with the validated build commands above before making any changes.**