#!/usr/bin/env python3
"""
Setup script for ChatGPT PDF Review Automation.
"""

import os
import subprocess
import sys
from pathlib import Path


def install_requirements():
    """Install Python requirements using the main project's uv environment."""
    print("Installing Python requirements in main project virtual environment...")

    # Change to main project directory
    main_project_dir = Path(__file__).parent.parent
    original_dir = os.getcwd()

    try:
        os.chdir(main_project_dir)

        # Check if pyproject.toml exists
        if not Path("pyproject.toml").exists():
            print("✗ pyproject.toml not found in main project directory")
            return False

        # Use uv to sync dependencies (this will install everything in pyproject.toml)
        try:
            subprocess.check_call(["uv", "sync"])
            print("✓ Dependencies synced successfully using uv")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"✗ Failed to sync dependencies with uv: {e}")
            print("  Make sure uv is installed: pip install uv")
            return False

    except Exception as e:
        print(f"✗ Failed to install requirements: {e}")
        return False
    finally:
        os.chdir(original_dir)


def setup_directories():
    """Create necessary directories."""
    print("Setting up directories...")

    directories = ["results", "logs", "chrome_user_data"]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

    return True


def check_chrome_installation():
    """Check if Chrome is installed."""
    print("Checking Chrome installation...")

    chrome_paths = [
        "/usr/bin/google-chrome",  # Linux
        "/usr/bin/chromium-browser",  # Linux alternative
        "/usr/bin/chromium",  # Linux alternative
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",  # macOS
        "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",  # Windows 64-bit
        "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",  # Windows 32-bit
    ]

    for path in chrome_paths:
        if os.path.exists(path):
            print(f"✓ Found Chrome at: {path}")
            return True

    print("✗ Chrome/Chromium not found. Please install Google Chrome.")
    print("  Download from: https://www.google.com/chrome/")
    return False


def validate_pdf_directories():
    """Validate that injected PDF directories exist."""
    print("Validating PDF directories...")

    injected_pdfs_dir = Path("../injected_pdfs")

    if not injected_pdfs_dir.exists():
        print(f"✗ Injected PDFs directory not found: {injected_pdfs_dir}")
        print("  Please run the PDF injection script first to generate injected PDFs.")
        return False

    # Count PDF directories
    pdf_dirs = [d for d in injected_pdfs_dir.iterdir() if d.is_dir()]
    pdf_count = 0

    for pdf_dir in pdf_dirs:
        pdfs = list(pdf_dir.glob("*.pdf"))
        pdf_count += len(pdfs)

    print(f"✓ Found {len(pdf_dirs)} PDF directories with {pdf_count} total PDFs")
    return True


def create_run_script():
    """Create a convenient run script."""
    print("Creating run script...")

    run_script_content = """#!/bin/bash
# ChatGPT PDF Review Automation Runner

echo "Starting ChatGPT PDF Review Automation..."
echo "Make sure Chrome is not running and you're logged into ChatGPT in a separate browser first."
echo ""

# Change to main project directory to use its .venv
cd "$(dirname "$0")/.."

# Run the automation using uv
echo "Running automation with uv..."
uv run python chatgpt_automation/main.py "$@"
"""

    with open("run.sh", "w") as f:
        f.write(run_script_content)

    # Make executable
    os.chmod("run.sh", 0o755)
    print("✓ Created run.sh script")

    return True


def main():
    """Main setup function."""
    print("=" * 60)
    print("ChatGPT PDF Review Automation Setup")
    print("=" * 60)
    print()

    success = True

    # Install requirements
    if not install_requirements():
        success = False
    print()

    # Setup directories
    if not setup_directories():
        success = False
    print()

    # Check Chrome
    if not check_chrome_installation():
        success = False
    print()

    # Validate PDFs
    if not validate_pdf_directories():
        success = False
    print()

    # Create run script
    if not create_run_script():
        success = False
    print()

    if success:
        print("=" * 60)
        print("✓ Setup completed successfully!")
        print("=" * 60)
        print()
        print("Next steps:")
        print("1. Make sure you're logged into ChatGPT in a browser")
        print("2. Close all Chrome windows")
        print("3. Run the automation:")
        print("   ./run.sh                              # Full automation")
        print("   uv run python main.py --help         # See all options")
        print(
            "   cd .. && uv run python chatgpt_automation/main.py --help  # From main project"
        )
        print()
        print("The automation will:")
        print("- Open Chrome and navigate to ChatGPT")
        print("- Upload each injected PDF")
        print("- Request reviews using different prompts")
        print("- Save all responses to the results/ directory")
        print()
    else:
        print("=" * 60)
        print("✗ Setup failed!")
        print("=" * 60)
        print("Please fix the issues above and run setup again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
