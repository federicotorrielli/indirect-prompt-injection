#!/bin/bash
# ChatGPT PDF Review Automation Runner

echo "Starting ChatGPT PDF Review Automation..."
echo "Make sure Chrome is not running and you're logged into ChatGPT in a separate browser first."
echo ""

# Change to main project directory to use its .venv
cd "$(dirname "$0")/.."

# Run the automation using uv
echo "Running automation with uv..."
uv run python chatgpt_automation/main.py "$@"
