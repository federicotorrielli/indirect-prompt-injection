"""
Factory for creating LLM automators (ChatGPT or Copilot).
"""

import logging

from config import Config

logger = logging.getLogger(__name__)


def create_llm_automator(config: Config):
    """Create appropriate LLM automator based on configuration."""

    if config.llm_service.lower() == "copilot":
        try:
            from copilot_api import CopilotAutomator

            logger.info("Creating Copilot automator")
            return CopilotAutomator(config)
        except ImportError as e:
            logger.error(f"Failed to import CopilotAutomator: {e}")
            raise
    elif config.llm_service.lower() == "chatgpt":
        try:
            from chatgpt_api import ChatGPTAutomator

            logger.info("Creating ChatGPT automator")
            return ChatGPTAutomator(config)
        except ImportError as e:
            logger.error(f"Failed to import ChatGPTAutomator: {e}")
            raise
    else:
        raise ValueError(
            f"Unsupported LLM service: {config.llm_service}. Supported: 'chatgpt', 'copilot'"
        )


class LLMAutomatorInterface:
    """Interface for LLM automators to ensure consistent API."""

    def initialize(self) -> bool:
        """Initialize the automator."""
        raise NotImplementedError

    def cleanup(self):
        """Clean up resources."""
        raise NotImplementedError

    def upload_pdf_and_request_review(self, pdf_path: str, review_request: str) -> str:
        """Upload PDF and request review."""
        raise NotImplementedError

    def refresh_page(self):
        """Refresh the current page."""
        raise NotImplementedError

    def start_new_conversation(self):
        """Start a new conversation."""
        raise NotImplementedError
