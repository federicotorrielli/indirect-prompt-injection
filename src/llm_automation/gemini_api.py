#!/usr/bin/env python3
"""
Gemini API wrapper for automated PDF review generation.
Uses the gemini-webapi package to interact with Google Gemini web interface.
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

try:
    from gemini_webapi import GeminiClient
except ImportError as e:
    raise ImportError(
        "gemini-webapi package not found. Please install it with: pip install gemini-webapi"
    ) from e

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

sys.path.insert(0, str(Path(__file__).parent))  # Ensure src/ is in sys.path
from config import Config

logger = logging.getLogger(__name__)


class GeminiAutomator:
    """Handles automated interactions with Google Gemini web interface."""

    def __init__(self, config: Config):
        self.config = config
        self.client: Optional[GeminiClient] = None
        self.chat_session = None

        # Gemini-specific settings
        self.secure_1psid = os.getenv("GEMINI_SECURE_1PSID")
        self.secure_1psidts = os.getenv("GEMINI_SECURE_1PSIDTS")

        # If cookies are not present in the current environment, try loading from .env
        if not self.secure_1psid or not self.secure_1psidts:
            self._load_env_from_dotenv()
            # Re-read after attempting to load from .env (do not override existing values)
            self.secure_1psid = self.secure_1psid or os.getenv("GEMINI_SECURE_1PSID")
            self.secure_1psidts = self.secure_1psidts or os.getenv(
                "GEMINI_SECURE_1PSIDTS"
            )

    def initialize(self) -> bool:
        """Initialize the Gemini client."""
        try:
            logger.info("Initializing Gemini client...")

            # Check for required credentials
            if not self.secure_1psid:
                logger.error(
                    "GEMINI_SECURE_1PSID environment variable not set. "
                    "Please set it with your Gemini __Secure-1PSID cookie value."
                )
                return False

            # Create Gemini client
            self.client = GeminiClient(
                secure_1psid=self.secure_1psid, secure_1psidts=self.secure_1psidts
            )

            # Initialize the client asynchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                loop.run_until_complete(
                    self.client.init(
                        timeout=60, auto_close=False, auto_refresh=True, verbose=True
                    )
                )
                logger.info("Gemini client initialized successfully")
                return True
            finally:
                # Don't close the loop as we'll need it for requests
                pass

        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            return False

    def cleanup(self):
        """Clean up the Gemini client."""
        if self.client:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, schedule cleanup
                    asyncio.create_task(self.client.close())
                else:
                    # If loop is not running, run cleanup directly
                    loop.run_until_complete(self.client.close())
            except Exception as e:
                logger.warning(f"Error during cleanup: {e}")
            finally:
                self.client = None
                logger.info("Gemini client cleaned up")

    def start_new_conversation(self) -> bool:
        """Start a new conversation (Gemini handles this automatically)."""
        try:
            # Gemini API handles new conversations automatically
            # We just need to reset our chat session
            self.chat_session = None
            logger.info("Started new Gemini conversation")
            return True
        except Exception as e:
            logger.error(f"Error starting new conversation: {e}")
            return False

    def upload_pdf_and_request_review(
        self, pdf_path: str, request_text: str
    ) -> Optional[str]:
        """Upload a PDF and request a review.

        Args:
            pdf_path: Path to the PDF file
            request_text: Review request text

        Returns:
            Generated review response or None if failed
        """
        try:
            if not self.client:
                logger.error("Gemini client not initialized")
                return None

            if not os.path.exists(pdf_path):
                logger.error(f"PDF file not found: {pdf_path}")
                return None

            pdf_name = Path(pdf_path).name
            logger.info(f"Uploading PDF and requesting review: {pdf_name}")

            # Run the async operation
            loop = asyncio.get_event_loop()

            # Tenacity-wrapped inner call: exponential backoff with jitter on
            # any transient exception from the Gemini web API. Max wait is
            # capped by config.max_retry_wait so we never hang indefinitely.
            max_wait = getattr(self.config, "max_retry_wait", 120)

            @retry(
                retry=retry_if_exception_type(Exception),
                wait=wait_random_exponential(multiplier=2, min=4, max=max_wait),
                stop=stop_after_attempt(5),
                before_sleep=before_sleep_log(logger, logging.WARNING),
                reraise=True,
            )
            async def _process_pdf():
                response = await self.client.generate_content(
                    request_text, files=[Path(pdf_path)], model="gemini-3.0-flash"
                )
                if response and response.text:
                    return response.text.strip()
                # None/empty response from the API → treat as transient and retry
                raise RuntimeError("Empty response from Gemini")

            # Execute the async operation. Tenacity raises on final failure.
            try:
                response = loop.run_until_complete(_process_pdf())
            except Exception as inner_e:
                logger.error(f"Gemini retries exhausted: {inner_e}")
                return None

            if response:
                logger.info(
                    f"Successfully received review response ({len(response)} characters)"
                )
                return response
            else:
                logger.error("Failed to get response from Gemini")
                return None

        except Exception as e:
            logger.error(f"Error in upload_pdf_and_request_review: {e}")
            return None

    def send_pdf_review_request(
        self, pdf_path: str, request_text: str
    ) -> Optional[str]:
        """Send a PDF review request (alias for upload_pdf_and_request_review)."""
        return self.upload_pdf_and_request_review(pdf_path, request_text)

    def _handle_rate_limits(self) -> bool:
        """Handle rate limits by waiting.

        Note: the primary backoff path is now tenacity inside
        `upload_pdf_and_request_review`. This method remains for compatibility
        but uses an exponential schedule capped by config.max_retry_wait so
        a repeated call does not sit on a flat 60s sleep.
        """
        try:
            wait = min(30, getattr(self.config, "max_retry_wait", 120))
            logger.warning(f"Rate limit detected, waiting {wait}s before retry...")
            time.sleep(wait)
            return True
        except Exception as e:
            logger.error(f"Error handling rate limits: {e}")
            return False

    def _check_for_errors(self) -> Optional[str]:
        """Check for any error conditions."""
        # Gemini API handles most errors internally
        # This is mainly for consistency with other automators
        return None

    def _load_env_from_dotenv(self) -> None:
        """Attempt to load environment variables from a .env file.

        This is a no-op if python-dotenv is not installed or if no .env file is found.
        Existing process environment variables take precedence (no override).
        """
        try:
            # Lazy import to avoid hard dependency if env is already configured
            from dotenv import find_dotenv, load_dotenv  # type: ignore

            dotenv_path = find_dotenv(usecwd=True)
            if dotenv_path:
                loaded = load_dotenv(dotenv_path=dotenv_path, override=False)
                if loaded:
                    logger.info(
                        f"Loaded environment variables from .env at: {dotenv_path}"
                    )
                else:
                    logger.debug(
                        ".env file found but no variables were loaded (possibly already set)."
                    )
            else:
                logger.debug("No .env file found via find_dotenv; skipping .env load.")
        except Exception as e:
            # Keep failures non-fatal; just log for diagnostics
            logger.debug(f"Skipping .env loading due to: {e}")
