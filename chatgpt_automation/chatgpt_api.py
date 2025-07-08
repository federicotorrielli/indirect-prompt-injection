"""
ChatGPT automation API using Selenium WebDriver.
Handles PDF uploads and review requests.
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional

import undetected_chromedriver as uc
from config import Config
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

logger = logging.getLogger(__name__)


class ChatGPTAutomator:
    """Handles automated interactions with ChatGPT web interface."""

    def __init__(self, config: Config):
        self.config = config
        self.driver = None
        self.wait = None

    def initialize(self) -> bool:
        """Initialize the Chrome driver and navigate to ChatGPT."""
        try:
            logger.info("Setting up Chrome driver...")

            # Setup Chrome options
            options = uc.ChromeOptions()

            # Add arguments for stability
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument("--disable-software-rasterizer")
            options.add_argument("--disable-extensions")
            options.add_argument("--disable-notifications")
            options.add_argument("--disable-popup-blocking")
            options.add_argument("--ignore-certificate-errors")
            options.add_argument(f"--window-size={self.config.chrome_window_size}")
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_argument("--disable-logging")
            options.add_argument("--log-level=3")
            options.add_argument("--silent")

            # User data directory
            user_data_dir = os.path.abspath(self.config.chrome_user_data_dir)
            os.makedirs(user_data_dir, exist_ok=True)
            options.add_argument(f"--user-data-dir={user_data_dir}")

            if self.config.chrome_headless:
                options.add_argument("--headless")

            # Initialize driver
            self.driver = uc.Chrome(options=options)
            if self.driver:
                self.driver.set_page_load_timeout(self.config.page_load_timeout)
                self.driver.implicitly_wait(self.config.implicit_wait)

                self.wait = WebDriverWait(self.driver, self.config.implicit_wait)

                logger.info("Navigating to ChatGPT...")
                self.driver.get(self.config.chatgpt_url)

            # Wait for page to load
            time.sleep(3)

            # Check if we need to handle any initial dialogs
            self._handle_initial_dialogs()

            logger.info("ChatGPT automation initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize ChatGPT automator: {e}")
            if self.driver:
                try:
                    self.driver.quit()
                except:
                    pass
                self.driver = None
            return False

    def cleanup(self):
        """Clean up the driver and resources."""
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
            self.driver = None
            logger.info("ChatGPT automator cleaned up")

    def _handle_initial_dialogs(self):
        """Handle any initial dialogs or prompts."""
        try:
            if not self.driver or not self.wait:
                return

            # Look for "Stay logged out" button
            stay_logged_out_xpath = "//a[contains(text(), 'Stay logged out')]"
            try:
                stay_logged_out = self.wait.until(
                    EC.element_to_be_clickable((By.XPATH, stay_logged_out_xpath))
                )
                stay_logged_out.click()
                logger.info("Clicked 'Stay logged out'")
                time.sleep(2)
            except TimeoutException:
                logger.info("No 'Stay logged out' button found")

            # Handle any other common dialogs
            time.sleep(2)

        except Exception as e:
            logger.warning(f"Error handling initial dialogs: {e}")

    def _ensure_driver_alive(self) -> bool:
        """Check if driver is still alive and responsive."""
        try:
            if not self.driver:
                return False
            self.driver.current_url
            return True
        except Exception:
            logger.warning("Driver appears to be dead")
            return False

    def _find_attachment_button(self):
        """Find the attachment/upload button element."""
        try:
            if not self.driver:
                return None

            # Common attachment button selectors for ChatGPT
            attachment_selectors = [
                "button[aria-label*='attach']",
                "button[aria-label*='file']",
                "button[aria-label*='upload']",
                "[data-testid='attachment-button']",
                "[data-testid='file-upload-button']",
                "button:has(svg[data-icon='paperclip'])",
                "button:has(svg[data-icon='attachment'])",
                "button[class*='attach']",
                ".attachment-button",
                "button[title*='attach']",
                "button[title*='file']",
            ]

            for selector in attachment_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        return elements[0]
                except Exception:
                    continue

            return None

        except Exception as e:
            logger.error(f"Error finding attachment button: {e}")
            return None

    def _is_attachment_button_disabled(self) -> bool:
        """Check if the attachment button is disabled/greyed out."""
        try:
            attachment_btn = self._find_attachment_button()
            if not attachment_btn:
                logger.warning("Could not find attachment button")
                return True

            # Check various indicators that the button is disabled
            try:
                class_attr = attachment_btn.get_attribute("class") or ""
                style_attr = attachment_btn.get_attribute("style") or ""

                is_disabled = (
                    not attachment_btn.is_enabled()
                    or attachment_btn.get_attribute("disabled") is not None
                    or attachment_btn.get_attribute("aria-disabled") == "true"
                    or "disabled" in class_attr.lower()
                    or ("opacity" in style_attr and "0" in style_attr)
                )

                if is_disabled:
                    logger.warning("Attachment button is disabled/greyed out")
                    return True

                return False
            except Exception as e:
                logger.warning(f"Error checking button attributes: {e}")
                return True

        except Exception as e:
            logger.warning(f"Error checking attachment button state: {e}")
            return False

    def _wait_for_attachment_rate_limit(self) -> bool:
        """Wait for attachment rate limit to expire (31 minutes)."""
        try:
            logger.warning("Attachment button is rate-limited. Waiting 31 minutes...")

            # Wait in chunks to allow for graceful interruption
            wait_time = 31 * 60  # 31 minutes in seconds
            chunk_size = 60  # 1 minute chunks

            for i in range(0, wait_time, chunk_size):
                remaining = wait_time - i
                minutes_remaining = remaining // 60
                logger.info(
                    f"Rate limit wait: {minutes_remaining} minutes remaining..."
                )
                time.sleep(min(chunk_size, remaining))

            logger.info("Rate limit wait completed. Attempting upload again...")
            return True

        except Exception as e:
            logger.error(f"Error during rate limit wait: {e}")
            return False

    def _find_file_upload_input(self):
        """Find the file upload input element."""
        try:
            if not self.driver:
                return None

            # Check if attachment button is disabled first
            if self._is_attachment_button_disabled():
                logger.warning("Attachment button is disabled - likely rate limited")
                if not self._wait_for_attachment_rate_limit():
                    return None

            # Common file input selectors for ChatGPT
            file_selectors = [
                "input[type='file']",
                "input[accept*='pdf']",
                "input[accept*='.pdf']",
                "[data-testid='file-upload']",
                ".file-upload input",
                "#file-upload",
            ]

            # First try to find visible file inputs
            for selector in file_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        return elements[0]
                except Exception:
                    continue

            # Try to click attachment button to reveal file input
            attachment_btn = self._find_attachment_button()
            if attachment_btn:
                try:
                    if attachment_btn.is_enabled():
                        attachment_btn.click()
                        time.sleep(1)

                        # Now look for file input again
                        for selector in file_selectors:
                            try:
                                elements = self.driver.find_elements(
                                    By.CSS_SELECTOR, selector
                                )
                                if elements:
                                    return elements[0]
                            except Exception:
                                continue
                except Exception as e:
                    logger.warning(f"Failed to click attachment button: {e}")

            return None

        except Exception as e:
            logger.error(f"Error finding file upload input: {e}")
            return None

    def _upload_pdf_file(self, pdf_path: str) -> bool:
        """Upload a PDF file to ChatGPT."""
        try:
            if not os.path.exists(pdf_path):
                logger.error(f"PDF file not found: {pdf_path}")
                return False

            logger.info(f"Attempting to upload PDF: {Path(pdf_path).name}")

            # Find and use file upload input
            file_input = self._find_file_upload_input()
            if not file_input:
                logger.error("Could not find file upload input")
                return False

            # Upload the file
            file_input.send_keys(os.path.abspath(pdf_path))

            # Wait for upload to complete
            logger.info("Waiting for file upload to complete...")
            time.sleep(self.config.upload_timeout)

            # Look for upload confirmation or file name in UI
            uploaded = self._wait_for_upload_confirmation(pdf_path)
            if uploaded:
                logger.info("PDF uploaded successfully")
                return True
            else:
                logger.warning("Could not confirm PDF upload")
                return True  # Continue anyway

        except Exception as e:
            logger.error(f"Failed to upload PDF: {e}")
            return False

    def _wait_for_upload_confirmation(self, pdf_path: str) -> bool:
        """Wait for confirmation that the file was uploaded."""
        try:
            pdf_name = Path(pdf_path).name

            # Look for the filename in the UI
            confirmation_selectors = [
                f"//*[contains(text(), '{pdf_name}')]",
                "//*[contains(@class, 'file') and contains(@class, 'uploaded')]",
                "//*[contains(@class, 'attachment')]",
            ]

            for selector in confirmation_selectors:
                try:
                    WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.XPATH, selector))
                    )
                    return True
                except TimeoutException:
                    continue

            return False

        except Exception as e:
            logger.warning(f"Error waiting for upload confirmation: {e}")
            return False

    def _send_message(self, message: str) -> bool:
        """Send a message to ChatGPT."""
        try:
            if not self.driver:
                logger.error("Driver not available")
                return False

            # Find text input area
            text_input_selectors = [
                "#prompt-textarea",
                "[data-testid='prompt-textarea']",
                "textarea[placeholder*='message']",
                "textarea[placeholder*='type']",
                ".prompt-textarea",
                "div[contenteditable='true']",
            ]

            text_input = None
            for selector in text_input_selectors:
                try:
                    text_input = self.driver.find_element(By.CSS_SELECTOR, selector)
                    if text_input:
                        break
                except Exception:
                    continue

            if not text_input:
                logger.error("Could not find text input area")
                return False

            # Clear and type message
            text_input.clear()
            text_input.send_keys(message)
            time.sleep(1)

            # Find and click send button
            send_button_selectors = [
                "[data-testid='send-button']",
                "button[aria-label*='send']",
                "button[type='submit']",
                ".send-button",
                "button:has(svg[class*='send'])",
            ]

            send_button = None
            for selector in send_button_selectors:
                try:
                    send_button = self.driver.find_element(By.CSS_SELECTOR, selector)
                    if send_button and send_button.is_enabled():
                        break
                except Exception:
                    continue

            if not send_button:
                # Try pressing Enter
                text_input.send_keys(Keys.RETURN)
            else:
                send_button.click()

            logger.info("Message sent successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False

    def _wait_for_response_completion(self) -> str:
        """Wait for ChatGPT response to complete and extract it."""
        try:
            if not self.driver:
                logger.error("Driver not available")
                return ""

            logger.info("Waiting for response...")

            # Wait for response container to appear
            response_container_selectors = [
                ".markdown.prose",
                "[data-testid='conversation-turn-response']",
                ".response-container",
                ".message-content",
                ".prose",
            ]

            response_element = None
            for selector in response_container_selectors:
                try:
                    response_element = WebDriverWait(self.driver, 30).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                    )
                    break
                except TimeoutException:
                    continue

            if not response_element:
                logger.error("Could not find response container")
                return ""

            # Wait for response to complete
            stable_count = 0
            last_length = 0
            start_time = time.time()

            while time.time() - start_time < self.config.max_response_wait:
                try:
                    # Get all response elements (in case there are multiple)
                    all_responses = self.driver.find_elements(
                        By.CSS_SELECTOR, response_container_selectors[0]
                    )
                    if all_responses:
                        current_text = all_responses[-1].text
                        current_length = len(current_text)

                        if current_length > 0 and current_length == last_length:
                            stable_count += 1
                            if stable_count >= 3:  # Response stable for 3 checks
                                logger.info("Response appears complete")
                                time.sleep(2)  # Final wait
                                return all_responses[-1].text
                        else:
                            stable_count = 0

                        last_length = current_length

                    time.sleep(2)

                except Exception as e:
                    logger.warning(f"Error checking response: {e}")
                    time.sleep(2)

            # Timeout reached, return what we have
            logger.warning("Response timeout reached")
            try:
                all_responses = self.driver.find_elements(
                    By.CSS_SELECTOR, response_container_selectors[0]
                )
                if all_responses:
                    return all_responses[-1].text
            except Exception:
                pass

            return ""

        except Exception as e:
            logger.error(f"Error waiting for response: {e}")
            return ""

    def send_pdf_review_request(
        self, pdf_path: str, request_text: str
    ) -> Optional[str]:
        """Upload a PDF and request a review."""
        try:
            if not self._ensure_driver_alive():
                logger.error("Driver is not alive")
                return None

            logger.info(f"Processing PDF review request for: {Path(pdf_path).name}")

            # Upload PDF
            if not self._upload_pdf_file(pdf_path):
                logger.error("Failed to upload PDF")
                return None

            # Send request message
            if not self._send_message(request_text):
                logger.error("Failed to send request message")
                return None

            # Wait for and extract response
            response = self._wait_for_response_completion()

            # After getting the response, start a new conversation
            if response:
                logger.info(f"Received response ({len(response)} characters)")

                # Start new conversation (this will navigate away from current chat)
                logger.info("Starting new conversation...")
                if self.start_new_conversation():
                    logger.info("Successfully started new conversation")
                else:
                    logger.warning(
                        "Failed to start new conversation, but continuing..."
                    )

                return response
            else:
                logger.error("No response received")
                # Still try to start new conversation to clean up
                self.start_new_conversation()
                return None

        except Exception as e:
            logger.error(f"Error in PDF review request: {e}")
            # Try to start new conversation to clean up
            try:
                self.start_new_conversation()
            except Exception:
                pass
            return None

    def start_new_conversation(self) -> bool:
        """Start a new conversation in ChatGPT by reloading the page."""
        try:
            if not self.driver:
                logger.error("Driver not available")
                return False

            logger.info("Starting new conversation by reloading page...")

            # Simply reload the ChatGPT URL to start a fresh conversation
            self.driver.get(self.config.chatgpt_url)
            time.sleep(3)  # Wait for page to load

            # Handle any initial dialogs that might appear
            self._handle_initial_dialogs()

            logger.info("Successfully started new conversation")
            return True

        except Exception as e:
            logger.warning(f"Error starting new conversation: {e}")
            return False
