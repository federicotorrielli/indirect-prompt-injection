"""
Configuration management for ChatGPT PDF Review Automation.
"""

import json
import os
from dataclasses import asdict, dataclass


@dataclass
class Config:
    """Configuration settings for the automation system."""

    # Directory paths
    injected_pdfs_dir: str = "data/injected_pdfs"
    results_dir: str = "results"
    logs_dir: str = "logs"

    # ChromeDriver settings
    chrome_user_data_dir: str = "chrome_user_data"
    chrome_headless: bool = False
    chrome_window_size: str = "1920,1080"
    page_load_timeout: int = 30
    implicit_wait: int = 5

    # Request settings
    request_delay: float = 3.0  # Seconds between requests
    upload_timeout: int = 60  # Seconds to wait for file upload
    response_timeout: int = 300  # Seconds to wait for response
    max_retries: int = 3

    # ChatGPT interaction settings
    chatgpt_url: str = "https://chatgpt.com/?model=gpt-4o&temporary-chat=true"
    max_response_wait: int = 180  # Maximum seconds to wait for response completion

    # Copilot interaction settings
    copilot_url: str = "https://copilot.microsoft.com/"

    # Gemini interaction settings
    gemini_timeout: int = 30
    gemini_auto_refresh: bool = True

    # LLM service selection ('chatgpt', 'copilot', or 'gemini')
    llm_service: str = "chatgpt"

    # Output settings
    save_intermediate_results: bool = True
    results_format: str = "json"  # json, csv, both

    @classmethod
    def load_from_file(cls, config_path: str = "config.json") -> "Config":
        """Load configuration from JSON file."""
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                data = json.load(f)
            return cls(**data)
        else:
            # Create default config
            config = cls()
            config.save_to_file(config_path)
            return config

    def save_to_file(self, config_path: str):
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    def validate(self) -> bool:
        """Validate configuration settings."""
        if not os.path.exists(self.injected_pdfs_dir):
            print(
                f"Warning: Injected PDFs directory not found: {self.injected_pdfs_dir}"
            )
            return False

        if self.request_delay < 1.0:
            print(
                "Warning: Request delay should be at least 1.0 seconds to avoid rate limiting"
            )

        if self.response_timeout < 60:
            print("Warning: Response timeout should be at least 60 seconds")

        return True

    def setup_directories(self):
        """Create necessary directories."""
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.chrome_user_data_dir, exist_ok=True)
