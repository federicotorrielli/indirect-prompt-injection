"""
ChatGPT automation API using Selenium WebDriver.
Handles PDF uploads and review requests.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import undetected_chromedriver as uc
from config import Config
from selenium.common.exceptions import (
    ElementNotInteractableException,
    TimeoutException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

logger = logging.getLogger(__name__)


class ChatGPTAutomator:
    """Handles automated interactions with ChatGPT web interface."""

    def __init__(self, config: Config):
        self.config = config
        self.driver: Optional[WebDriver] = None
        self.selector_cache: Dict[str, Any] = {}  # Cache for successful selectors

    def initialize(self) -> bool:
        """Initialize the WebDriver and navigate to ChatGPT."""
        try:
            logger.info("Setting up Chrome driver...")

            # Setup Chrome options
            options = uc.ChromeOptions()

            options.add_argument(
                "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            )

            # Add arguments for stability
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
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

            options.add_argument("--disable-background-timer-throttling")
            options.add_argument("--disable-renderer-backgrounding")
            options.add_argument("--disable-backgrounding-occluded-windows")
            options.add_argument(
                "--disable-features=CalculateNativeWinOcclusion,IntensiveWakeUpThrottling"
            )

            # User data directory
            user_data_dir = os.path.abspath(self.config.chrome_user_data_dir)
            os.makedirs(user_data_dir, exist_ok=True)
            options.add_argument(f"--user-data-dir={user_data_dir}")
            print(f"User data directory: {user_data_dir}")

            if self.config.chrome_headless:
                options.add_argument("--headless")

            # Initialize driver
            self.driver = uc.Chrome(options=options, use_subprocess=True)
            if self.driver:
                self.driver.set_page_load_timeout(self.config.page_load_timeout)
                self.driver.implicitly_wait(self.config.implicit_wait)

                try:
                    self.driver.execute_cdp_cmd(
                        "Emulation.setFocusEmulationEnabled", {"enabled": True}
                    )
                except Exception as e:
                    logger.debug(f"Could not enable focus emulation: {e}")
                try:
                    self.driver.execute_cdp_cmd(
                        "Page.setWebLifecycleState", {"state": "active"}
                    )
                except Exception as e:
                    logger.debug(f"Could not pin page lifecycle state: {e}")

                self.wait = WebDriverWait(self.driver, self.config.implicit_wait)

                logger.info("Navigating to ChatGPT...")
                self.driver.get(self.config.chatgpt_url)

            # Wait for page to load
            time.sleep(3)

            # Check for usage limits immediately after loading
            usage_error = self._check_for_usage_limits()
            if usage_error:
                self._handle_usage_limit_error(usage_error)
                # This won't return since _handle_usage_limit_error calls exit()

            # Check if we need to handle any initial dialogs
            self._handle_initial_dialogs()

            logger.info("ChatGPT automation initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize ChatGPT automator: {e}")
            if self.driver:
                try:
                    self.driver.quit()
                except Exception:
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
        """Handle any initial dialogs or prompts with optimized detection."""
        try:
            if not self.driver:
                return

            logger.info("Checking for initial dialogs...")

            # First, do a quick check if we're already at the chat interface
            if self._is_chat_interface_ready():
                logger.info("Chat interface already ready - no dialogs to handle")
                return

            # Quick check for login requirements (2 second timeout)
            login_required = self._check_for_login_requirement()

            if login_required:
                logger.info("Login required - waiting for user to complete login")
                self._wait_for_login_completion()
                return

            # Quick check for "Stay logged out" option (2 second timeout)
            if self._handle_stay_logged_out():
                logger.info("Handled 'Stay logged out' option")

            # Final short wait for any remaining page elements to load
            time.sleep(1)

        except Exception as e:
            logger.warning(f"Error handling initial dialogs: {e}")

    def _is_chat_interface_ready(self) -> bool:
        """Quick check if the chat interface is already loaded and ready."""
        try:
            if not self.driver:
                return False

            chat_indicators = [
                "#prompt-textarea",
                "[data-testid='prompt-textarea']",
                "textarea[placeholder*='message']",
                "div[contenteditable='true']",
            ]

            for selector in chat_indicators:
                try:
                    element = WebDriverWait(self.driver, 1).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                    )
                    if element and element.is_displayed():
                        return True
                except TimeoutException:
                    continue

            return False

        except Exception:
            return False

    def _check_for_login_requirement(self) -> bool:
        """Quick check if login is required (2 second timeout)."""
        try:
            if not self.driver:
                return False

            login_selectors = [
                ("[data-testid='login-button']", By.CSS_SELECTOR),
                ("//button[contains(text(), 'Log in')]", By.XPATH),
                ("//a[contains(text(), 'Log in')]", By.XPATH),
                ("//a[contains(@href, 'login')]", By.XPATH),
            ]

            # Try cached selector first
            cached = self.selector_cache.get("login_button")
            if cached:
                cached_selector, by = cached
                try:
                    element = self.driver.find_element(by, cached_selector)
                    if element and element.is_displayed():
                        logger.debug(
                            f"Login required (cached selector: {cached_selector})"
                        )
                        return True
                except Exception:
                    logger.debug("Cached login selector failed, trying all")
                    del self.selector_cache["login_button"]  # Remove failed selector

            for selector, by in login_selectors:
                try:
                    element = self.driver.find_element(by, selector)
                    if element and element.is_displayed():
                        logger.debug(f"Login required (new selector: {selector})")
                        self.selector_cache["login_button"] = (selector, by)
                        return True
                except Exception:
                    continue

            return False

        except Exception:
            return False

    def _handle_stay_logged_out(self) -> bool:
        """Handle 'Stay logged out' button with short timeout."""
        try:
            if not self.driver:
                return False

            stay_logged_out_xpath = "//a[contains(text(), 'Stay logged out')]"

            stay_logged_out = WebDriverWait(self.driver, 2).until(
                EC.element_to_be_clickable((By.XPATH, stay_logged_out_xpath))
            )

            if stay_logged_out:
                stay_logged_out.click()
                logger.info("Clicked 'Stay logged out'")
                time.sleep(1)
                return True

        except TimeoutException:
            logger.debug("No 'Stay logged out' button found")
        except Exception as e:
            logger.warning(f"Error handling 'Stay logged out': {e}")

        return False

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
                "#upload-file-btn",  # Specific ID for upload button
                "button[aria-label*='attach']",
                "button[aria-label*='file']",
                "button[aria-label*='upload']",
                "button[aria-label*='Add files']",  # Match "Add files is unavailable"
                "[data-testid='attachment-button']",
                "[data-testid='file-upload-button']",
                "button:has(svg[data-icon='paperclip'])",
                "button:has(svg[data-icon='attachment'])",
                "button[class*='attach']",
                "button[class*='composer-btn']",  # Match the composer-btn class
                ".attachment-button",
                "button[title*='attach']",
                "button[title*='file']",
            ]

            # Try cached selector first
            cached_selector = self.selector_cache.get("attachment_button")
            if cached_selector:
                try:
                    element = self.driver.find_element(By.CSS_SELECTOR, cached_selector)
                    if element and element.is_displayed():
                        return element
                except Exception:
                    logger.debug("Cached attachment button selector failed, trying all")
                    del self.selector_cache["attachment_button"]

            for selector in attachment_selectors:
                try:
                    element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    if element and element.is_displayed():
                        self.selector_cache["attachment_button"] = selector
                        return element
                except Exception:
                    continue

            return None

        except Exception as e:
            logger.error(f"Error finding attachment button: {e}")
            return None

    def _check_upload_limit_in_menu(self) -> int:
        """Check if the attachment dropdown menu shows 'Upload limit reached'.

        After clicking the attachment button, ChatGPT may show a radix dropdown
        menu where the upload option is disabled with 'Upload limit reached' text
        and a subtext like 'Wait 41 minutes to upload again'.

        Returns:
            The number of minutes to wait (parsed from menu text + 1 buffer),
            or 0 if no upload limit is detected.
        """
        try:
            if not self.driver:
                return 0

            # Check for disabled menu items indicating upload limit
            limit_selectors = [
                "div[role='menuitem'][aria-disabled='true'][data-disabled]",
                "div[role='menuitem'][aria-disabled='true']",
                "div[role='menu'] [data-disabled]",
            ]

            for selector in limit_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for element in elements:
                        if element.is_displayed():
                            text = element.text
                            if "upload limit" in text.lower():
                                logger.warning(f"Upload limit detected in menu: {text}")
                                return self._parse_wait_minutes(text)
                except Exception:
                    continue

            return 0

        except Exception as e:
            logger.warning(f"Error checking upload limit in menu: {e}")
            return 0

    @staticmethod
    def _parse_wait_minutes(text: str) -> int:
        """Parse wait minutes from upload limit menu text.

        Handles formats like:
          'Wait 41 minutes to upload again'
          'Wait 1 hour to upload again'

        Returns parsed minutes + 1 buffer, or 61 as fallback.
        """
        import re

        text_lower = text.lower()

        # Match "N minutes"
        minutes_match = re.search(r"(\d+)\s*minutes?", text_lower)
        if minutes_match:
            return int(minutes_match.group(1)) + 1

        # Match "N hour(s)"
        hours_match = re.search(r"(\d+)\s*hours?", text_lower)
        if hours_match:
            return int(hours_match.group(1)) * 60 + 1

        # Fallback
        logger.warning(
            f"Could not parse wait time from: {text!r}, defaulting to 61 minutes"
        )
        return 61

    def _dismiss_dropdown_menu(self):
        """Dismiss any open radix dropdown menu by pressing Escape."""
        try:
            if not self.driver:
                return
            from selenium.webdriver.common.keys import Keys

            body = self.driver.find_element(By.TAG_NAME, "body")
            body.send_keys(Keys.ESCAPE)
            time.sleep(0.5)
        except Exception as e:
            logger.debug(f"Error dismissing dropdown menu: {e}")

    def _check_upload_rate_limit_via_menu(self) -> int:
        """Click the attachment button to open the dropdown and check for upload rate limit.

        Returns the number of minutes to wait if rate-limited, or 0 if not.
        Always dismisses the dropdown menu before returning.
        """
        try:
            attachment_btn = self._find_attachment_button()
            if not attachment_btn:
                return 0

            attachment_btn.click()
            time.sleep(1)

            wait_minutes = self._check_upload_limit_in_menu()
            if wait_minutes > 0:
                logger.warning("Upload rate limit detected via dropdown menu")
                self._dismiss_dropdown_menu()
                return wait_minutes

            self._dismiss_dropdown_menu()
            return 0
        except Exception as e:
            logger.debug(f"Error checking upload rate limit via menu: {e}")
            self._dismiss_dropdown_menu()
            return 0

    def _wait_for_attachment_rate_limit(self, wait_minutes: int = 61) -> bool:
        """Wait for attachment rate limit to expire."""
        try:
            logger.warning(f"Upload rate-limited. Waiting {wait_minutes} minutes...")

            # Wait in chunks to allow for graceful interruption
            wait_time = wait_minutes * 60
            chunk_size = 60  # 1 minute chunks

            for i in range(0, wait_time, chunk_size):
                remaining = wait_time - i
                minutes_remaining = remaining // 60
                logger.info(
                    f"Rate limit wait: {minutes_remaining} minutes remaining..."
                )
                time.sleep(min(chunk_size, remaining))

            logger.info("Rate limit wait completed. Attempting upload again...")
            # Refresh the page to reset any stale elements and clear potential UI issues
            self.refresh_page()
            return True

        except Exception as e:
            logger.error(f"Error during rate limit wait: {e}")
            return False

    def _find_file_upload_input(self):
        """Find the file upload input element."""
        try:
            if not self.driver:
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

            # Try cached selector first
            cached_selector = self.selector_cache.get("file_upload_input")
            if cached_selector:
                try:
                    element = self.driver.find_element(By.CSS_SELECTOR, cached_selector)
                    if element:  # Hidden inputs are fine
                        return element
                except Exception:
                    logger.debug("Cached file upload selector failed, trying all")
                    del self.selector_cache["file_upload_input"]

            # First try to find visible file inputs
            for selector in file_selectors:
                try:
                    element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    if element:
                        self.selector_cache["file_upload_input"] = selector
                        return element
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
                                element = self.driver.find_element(
                                    By.CSS_SELECTOR, selector
                                )
                                if element:
                                    return element
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

            # Pre-check: click attachment button to detect upload rate limit
            # before wasting time on a silent upload failure
            wait_minutes = self._check_upload_rate_limit_via_menu()
            if wait_minutes > 0:
                if self._wait_for_attachment_rate_limit(wait_minutes):
                    # Retry after waiting
                    return self._upload_pdf_file(pdf_path)
                return False

            # Find and use file upload input
            file_input = self._find_file_upload_input()
            if not file_input:
                logger.error("Could not find file upload input")
                return False

            # Upload the file
            file_input.send_keys(os.path.abspath(pdf_path))

            # Wait for upload to complete by monitoring send button state
            logger.info("Waiting for file upload to complete...")
            if self._wait_for_upload_completion_by_button():
                logger.info("PDF uploaded successfully")

                # Check specifically for unknown error after upload
                time.sleep(1)  # Brief wait for any error to appear
                if self._check_for_unknown_error_after_upload():
                    self._handle_usage_limit_error(
                        "Unknown error occurred after upload"
                    )
                    # This won't return since _handle_usage_limit_error calls exit()

                return True
            else:
                logger.error("Upload failed - no upload activity detected")

                # Post-check: maybe rate limit appeared after the attempt
                post_wait_minutes = self._check_upload_rate_limit_via_menu()
                if post_wait_minutes > 0:
                    if self._wait_for_attachment_rate_limit(post_wait_minutes):
                        return self._upload_pdf_file(pdf_path)

                return False

        except Exception as e:
            logger.error(f"Failed to upload PDF: {e}")
            return False

    def _wait_for_upload_confirmation(self, pdf_path: str) -> bool:
        """Wait for confirmation that the file was uploaded."""
        try:
            if not self.driver:
                logger.warning("Driver not available for upload confirmation")
                return False

            pdf_name = Path(pdf_path).name

            # Look for the filename in the UI
            confirmation_selectors = [
                "//*[contains(text(), '{pdf_name}')]",
                "//*[contains(@class, 'file') and contains(@class, 'uploaded')]",
                "//*[contains(@class, 'attachment')]",
            ]

            # Try cached selector first
            cached_selector = self.selector_cache.get("upload_confirmation")
            if cached_selector:
                try:
                    if "{pdf_name}" in cached_selector:
                        selector_to_try = cached_selector.format(pdf_name=pdf_name)
                    else:
                        selector_to_try = cached_selector
                    WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.XPATH, selector_to_try))
                    )
                    return True
                except TimeoutException:
                    logger.debug("Cached upload confirmation selector failed")
                    del self.selector_cache["upload_confirmation"]

            for selector in confirmation_selectors:
                try:
                    if "{pdf_name}" in selector:
                        selector_to_try = selector.format(pdf_name=pdf_name)
                    else:
                        selector_to_try = selector
                    WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.XPATH, selector_to_try))
                    )
                    self.selector_cache["upload_confirmation"] = selector
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

            # Find text input area with retry logic
            text_input = self._find_text_input_with_retry()
            if not text_input:
                logger.error("Could not find text input area")
                return False

            # Clear and type message with stale element handling and validation
            message_sent_successfully = False
            max_attempts = 3

            for attempt in range(max_attempts):
                try:
                    # Clear the input first
                    text_input.clear()
                    time.sleep(0.5)  # Brief pause after clearing

                    # Send the message
                    text_input.send_keys(message)
                    time.sleep(1)  # Wait for text to be typed

                    # Validate that the message was actually typed
                    current_value = self._get_text_input_value(text_input)
                    if current_value and message.strip() in current_value.strip():
                        logger.debug(
                            f"Message successfully typed (attempt {attempt + 1})"
                        )
                        message_sent_successfully = True
                        break
                    elif current_value and len(current_value.strip()) > 0:
                        # Text input has content but doesn't match our message
                        logger.warning(
                            f"Text input contains unexpected content (attempt {attempt + 1}): '{current_value[:100]}...'"
                        )
                        # Try to clear and re-type
                        text_input.clear()
                        time.sleep(0.5)
                        text_input.send_keys(message)
                        time.sleep(1)
                        # Re-check
                        current_value = self._get_text_input_value(text_input)
                        if current_value and message.strip() in current_value.strip():
                            logger.debug(
                                f"Message successfully typed after clearing (attempt {attempt + 1})"
                            )
                            message_sent_successfully = True
                            break
                    else:
                        logger.warning(
                            f"Message not properly typed (attempt {attempt + 1}): expected '{message[:50]}...', got '{current_value[:50] if current_value else 'empty'}...'"
                        )
                        if attempt < max_attempts - 1:
                            time.sleep(1)  # Wait before retry
                            # Try to re-find the text input in case it changed
                            text_input = self._find_text_input_with_retry()
                            if not text_input:
                                logger.error("Could not re-find text input for retry")
                                return False

                except ElementNotInteractableException as e:
                    logger.warning(
                        f"Element not interactable (attempt {attempt + 1}): {e.msg if hasattr(e, 'msg') else e}. Trying JS fallback."
                    )
                    text_input = self._find_text_input_with_retry()
                    if text_input and self._type_message_js(text_input, message):
                        logger.info("Message typed via JS fallback")
                        message_sent_successfully = True
                        break
                except Exception as e:
                    if "stale element reference" in str(e).lower():
                        logger.debug(
                            f"Stale element during text input (attempt {attempt + 1}), retrying..."
                        )
                        text_input = self._find_text_input_with_retry()
                        if not text_input:
                            logger.error(
                                "Could not re-find text input after stale element"
                            )
                            return False
                    else:
                        logger.warning(
                            f"Error typing message (attempt {attempt + 1}): {e}"
                        )
                        if attempt == max_attempts - 1:
                            raise

            if not message_sent_successfully:
                logger.warning(
                    "send_keys retries exhausted; attempting JS typing fallback"
                )
                text_input = self._find_text_input_with_retry()
                if text_input and self._type_message_js(text_input, message):
                    logger.info("Message typed via JS fallback after send_keys failed")
                    message_sent_successfully = True

            if not message_sent_successfully:
                logger.error(
                    "Failed to type message into text input after all attempts"
                )
                return False

            # Find and click send button with retry logic
            send_button = self._find_send_button_with_retry()

            if send_button:
                try:
                    send_button.click()

                    # Verify the message was sent by checking if text input was cleared
                    time.sleep(1)  # Brief wait for UI to update
                    current_text_input = self._find_text_input_with_retry()
                    if current_text_input:
                        remaining_value = self._get_text_input_value(current_text_input)
                        if not remaining_value or remaining_value.strip() == "":
                            logger.info(
                                "Message sent successfully - text input cleared"
                            )
                            return True
                        else:
                            logger.warning(
                                f"Message may not have been sent - text input still contains: '{remaining_value[:50]}...'. Trying JS click fallback."
                            )
                            if self._click_send_button_js():
                                time.sleep(1)
                                final_text_input = self._find_text_input_with_retry()
                                if final_text_input:
                                    final_value = self._get_text_input_value(
                                        final_text_input
                                    )
                                    if not final_value or final_value.strip() == "":
                                        logger.info(
                                            "Message sent successfully via JS click fallback"
                                        )
                                        return True
                            logger.error(
                                "Send button click did not submit message (input not cleared)"
                            )
                            return False

                    logger.info("Message sent successfully")
                    return True
                except Exception as e:
                    if "stale element reference" in str(e).lower():
                        logger.debug(
                            "Stale element during send button click, retrying..."
                        )
                        send_button = self._find_send_button_with_retry()
                        if send_button:
                            send_button.click()
                            logger.info("Message sent successfully")
                            return True
                    raise
            else:
                # Fallback: Try pressing Enter on a fresh text input
                logger.debug("No send button found, trying Enter key")
                fresh_text_input = self._find_text_input_with_retry()
                if fresh_text_input:
                    fresh_text_input.send_keys(Keys.RETURN)

                    # Verify the message was sent by checking if text input was cleared
                    time.sleep(1)
                    post_enter_value = self._get_text_input_value(fresh_text_input)
                    if not post_enter_value or post_enter_value.strip() == "":
                        logger.info("Message sent successfully via Enter key")
                        return True
                    else:
                        logger.warning(
                            "Enter key may not have sent message - text input not cleared"
                        )
                        return False
                else:
                    logger.error("Could not find text input for Enter key fallback")
                    return False

        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False

    def _type_message_js(self, element, message: str) -> bool:
        """Write text into the editor via JS, bypassing native focus.

        Why: when Chrome is occluded (e.g. a fullscreen film is in front),
        send_keys silently no-ops on ProseMirror. Dispatching a synthetic
        paste event with a DataTransfer payload is the path ProseMirror's
        own clipboard handler takes, so its internal state stays in sync
        with the DOM — unlike textContent assignment, which leaves the
        send button thinking the editor is empty.
        """
        if not self.driver:
            return False
        script = """
            const el = arguments[0];
            const text = arguments[1];
            try { el.scrollIntoView({block: 'center'}); } catch (e) {}
            el.focus();
            try {
                const sel = window.getSelection();
                const range = document.createRange();
                range.selectNodeContents(el);
                sel.removeAllRanges();
                sel.addRange(range);
                document.execCommand('delete', false);
            } catch (e) {}
            const dt = new DataTransfer();
            dt.setData('text/plain', text);
            const pasteEvent = new ClipboardEvent('paste', {
                clipboardData: dt,
                bubbles: true,
                cancelable: true,
            });
            el.dispatchEvent(pasteEvent);
            if (!(el.textContent || '').includes(text) && !(el.value || '').includes(text)) {
                const ok = document.execCommand('insertText', false, text);
                if (!ok) {
                    if (el.tagName === 'TEXTAREA' || el.tagName === 'INPUT') {
                        el.value = text;
                    } else {
                        el.textContent = text;
                    }
                    el.dispatchEvent(new InputEvent('input', {
                        inputType: 'insertText', data: text, bubbles: true
                    }));
                }
            }
            return el.textContent || el.value || '';
        """
        try:
            self.driver.execute_script(script, element, message)
        except Exception as e:
            logger.warning(f"JS typing fallback raised: {e}")
            return False

        time.sleep(0.5)
        current = self._get_text_input_value(element)
        if current and message.strip() in current.strip():
            return True
        logger.warning(
            f"JS typing fallback did not populate input (got '{current[:50] if current else 'empty'}...')"
        )
        return False

    def _click_send_button_js(self) -> bool:
        """Click the send button via JS, bypassing native event dispatch.

        Why: when Chrome is occluded the Selenium-driven native click can
        be dropped. A JS .click() runs synchronously in page context and
        triggers the React onClick handler regardless of window focus.
        """
        if not self.driver:
            return False
        send_button_selectors = [
            "#composer-submit-button",
            "[data-testid='send-button']",
            "button[aria-label*='Send']",
            "button[aria-label*='send']",
            "button[type='submit']",
        ]
        script = """
            const selectors = arguments[0];
            for (const sel of selectors) {
                const btn = document.querySelector(sel);
                if (btn && !btn.disabled) {
                    btn.click();
                    return true;
                }
            }
            return false;
        """
        try:
            return bool(self.driver.execute_script(script, send_button_selectors))
        except Exception as e:
            logger.warning(f"JS send click raised: {e}")
            return False

    def _find_text_input_with_retry(self):
        """Find text input with retry logic to handle dynamic content."""
        text_input_selectors = [
            "#prompt-textarea",
            "[data-testid='prompt-textarea']",
            "textarea[placeholder*='message']",
            "textarea[placeholder*='type']",
            ".prompt-textarea",
            "div[contenteditable='true']",
        ]

        # Try cached selector first
        cached_selector = self.selector_cache.get("text_input")
        if cached_selector:
            for attempt in range(2):  # Try cached twice
                try:
                    element = self.driver.find_element(By.CSS_SELECTOR, cached_selector)
                    if element and element.is_displayed():
                        return element
                except Exception:
                    time.sleep(0.5)
            logger.debug("Cached text input selector failed, trying all")
            del self.selector_cache["text_input"]

        for attempt in range(3):  # Try 3 times
            for selector in text_input_selectors:
                try:
                    element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    if element and element.is_displayed():
                        self.selector_cache["text_input"] = selector
                        return element
                except Exception:
                    continue

            if attempt < 2:  # Don't sleep on the last attempt
                time.sleep(0.5)

        return None

    def _get_text_input_value(self, text_input_element) -> str:
        """Get the current value from a text input element."""
        try:
            # For regular textarea elements
            value = text_input_element.get_attribute("value")
            if value:
                return value

            # For contenteditable divs (common in modern chat interfaces)
            if text_input_element.get_attribute("contenteditable") == "true":
                return text_input_element.text

            # Fallback: try to get text content
            text_content = text_input_element.text
            if text_content:
                return text_content

            return ""
        except Exception as e:
            logger.debug(f"Error getting text input value: {e}")
            return ""

    def _find_send_button_with_retry(self):
        """Find send button with retry logic to handle dynamic content."""
        send_button_selectors = [
            "#composer-submit-button",
            "[data-testid='send-button']",
            "button[aria-label*='Send']",
            "button[aria-label*='send']",
            "button[type='submit']",
            ".send-button",
        ]

        # Try cached selector first
        cached_selector = self.selector_cache.get("send_button")
        if cached_selector:
            for attempt in range(2):  # Try cached twice
                try:
                    element = self.driver.find_element(By.CSS_SELECTOR, cached_selector)
                    if element and element.is_displayed() and element.is_enabled():
                        return element
                except Exception:
                    time.sleep(0.5)
            logger.debug("Cached send button selector failed, trying all")
            del self.selector_cache["send_button"]

        for attempt in range(3):  # Try 3 times
            for selector in send_button_selectors:
                try:
                    element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    if element and element.is_displayed() and element.is_enabled():
                        self.selector_cache["send_button"] = selector
                        return element
                except Exception:
                    continue

            if attempt < 2:  # Don't sleep on the last attempt
                time.sleep(0.5)

        return None

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
            # Try cached selector first
            cached_selector = self.selector_cache.get("response_container")
            if cached_selector:
                try:
                    response_element = WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located(
                            (By.CSS_SELECTOR, cached_selector)
                        )
                    )
                except TimeoutException:
                    logger.debug("Cached response container selector failed")
                    del self.selector_cache["response_container"]

            if not response_element:
                for selector in response_container_selectors:
                    try:
                        response_element = WebDriverWait(self.driver, 30).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                        )
                        self.selector_cache["response_container"] = selector
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
                    response_selector = self.selector_cache.get(
                        "response_container", response_container_selectors[0]
                    )
                    all_responses = self.driver.find_elements(
                        By.CSS_SELECTOR, response_selector
                    )
                    if all_responses:
                        current_text = all_responses[-1].text
                        current_length = len(current_text)

                        if current_length > 0 and current_length == last_length:
                            stable_count += 1
                            if stable_count >= 3:  # Response stable for 3 checks
                                logger.info("Response appears complete")
                                return current_text
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
                response_selector = self.selector_cache.get(
                    "response_container", response_container_selectors[0]
                )
                all_responses = self.driver.find_elements(
                    By.CSS_SELECTOR, response_selector
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

            # Check for immediate usage limit errors after sending
            time.sleep(1)  # Brief wait for any immediate error to appear
            usage_error = self._check_for_usage_limits()
            if usage_error:
                self._handle_usage_limit_error(usage_error)
                # This won't return since _handle_usage_limit_error calls exit()

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
            self._save_debug_screenshot(Path(pdf_path).stem, "pdf_review_error")
            # Try to start new conversation to clean up
            try:
                self.start_new_conversation()
            except Exception:
                pass
            return None

    def upload_pdf_and_request_review(
        self, pdf_path: str, review_request: str
    ) -> Optional[str]:
        """Alias for send_pdf_review_request to maintain compatibility with generic interface."""
        return self.send_pdf_review_request(pdf_path, review_request)

    def _save_debug_screenshot(self, pdf_stem: str, tag: str) -> None:
        """Save a browser screenshot to aid unattended-run debugging.

        No-op if disabled in config or if the driver is not available.
        """
        if not getattr(self.config, "screenshot_on_failure", False):
            return
        try:
            if not self.driver:
                return
            out_dir = getattr(
                self.config, "screenshot_dir", "results/debug_screenshots"
            )
            os.makedirs(out_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(out_dir, f"{pdf_stem}_{tag}_{ts}.png")
            self.driver.save_screenshot(path)
            logger.info(f"Saved debug screenshot: {path}")
        except Exception as e:
            logger.debug(f"Failed to save debug screenshot: {e}")

    def refresh_page(self):
        """Refresh the current page."""
        try:
            if self.driver:
                logger.info("Refreshing page...")
                self.driver.refresh()
                time.sleep(3)
                self._handle_initial_dialogs()
        except Exception as e:
            logger.error(f"Error refreshing page: {e}")
            raise

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

            # Check for usage limits after reload (common place for limit messages)
            usage_error = self._check_for_usage_limits()
            if usage_error:
                self._handle_usage_limit_error(usage_error)
                # This won't return since _handle_usage_limit_error calls exit()

            # Handle any initial dialogs that might appear
            self._handle_initial_dialogs()

            logger.info("Successfully started new conversation")
            return True

        except Exception as e:
            logger.warning(f"Error starting new conversation: {e}")
            return False

    def _check_for_usage_limits(self) -> Optional[str]:
        """Check for usage cap or limit error messages."""
        try:
            if not self.driver:
                return None

            logger.debug("Checking for usage limit errors...")

            # Usage limit patterns to detect
            limit_patterns = [
                "usage cap for gpt",
                "hit the edu plan limit",
                "plan limit for gpt",
                "unknown error occurred",
                "internal error",
            ]

            # Use cached selector for performance if available
            cached_selector = self.selector_cache.get("usage_limit_error")
            if cached_selector:
                try:
                    elements = self.driver.find_elements(
                        By.CSS_SELECTOR, cached_selector
                    )
                    for element in elements:
                        if element.is_displayed():
                            text = element.text.lower()
                            for pattern in limit_patterns:
                                if pattern in text:
                                    logger.error(
                                        f"Usage limit detected (cached): {element.text}"
                                    )
                                    return element.text
                except Exception:
                    logger.debug("Cached usage limit selector failed")
                    del self.selector_cache["usage_limit_error"]

            # Combined selector for faster single DOM query
            combined_selector = (
                ".text-token-text-error, "
                "div.text-token-text-error, "
                "aside.flex.w-full, "
                "div[class*='text-token-text-error'], "
                "div[class*='border-token-surface-error'], "
                "aside[class*='rounded-3xl'][class*='border'], "
                "div[class*='border-red-500'][role='alert'], "
                "div.border-red-500[role='alert'], "
                "[role='alert'][class*='border-red-500']"
            )

            # Single DOM query for all error elements
            try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, combined_selector)
                for element in elements:
                    if element.is_displayed():
                        text = element.text.lower()
                        for pattern in limit_patterns:
                            if pattern in text:
                                logger.error(f"Usage limit detected: {element.text}")
                                # Simple caching - just cache the combined selector since it worked
                                self.selector_cache["usage_limit_error"] = (
                                    combined_selector
                                )
                                return element.text
            except Exception:
                pass

            # Quick check for regenerate button (single query)
            try:
                regenerate_button = self.driver.find_element(
                    By.CSS_SELECTOR, "[data-testid='regenerate-thread-error-button']"
                )
                if regenerate_button and regenerate_button.is_displayed():
                    parent = regenerate_button.find_element(By.XPATH, "./..")
                    if parent:
                        error_text = parent.text.lower()
                        if any(pattern in error_text for pattern in limit_patterns):
                            logger.error(
                                f"Usage limit detected via regenerate button: {parent.text}"
                            )
                            return parent.text
            except Exception:
                pass

            return None

        except Exception as e:
            logger.warning(f"Error checking for usage limits: {e}")
            return None

    def _handle_usage_limit_error(self, error_message: str):
        """Handle usage limit errors by stopping execution with a clear notice."""
        logger.error("=" * 80)
        logger.error("🚨 CRITICAL: USAGE LIMIT REACHED 🚨")
        logger.error("=" * 80)
        logger.error(f"Error message: {error_message}")
        logger.error("")
        logger.error(
            "The script has been stopped because ChatGPT usage limits have been reached."
        )
        logger.error(
            "Please wait for the limit to reset or upgrade your plan before continuing."
        )
        logger.error("=" * 80)

        # Print to console as well for visibility
        print("\n" + "=" * 80)
        print("🚨 CRITICAL: CHATGPT USAGE LIMIT REACHED 🚨")
        print("=" * 80)
        print(f"Error message: {error_message}")
        print("")
        print(
            "The script has been stopped because ChatGPT usage limits have been reached."
        )
        print(
            "Please wait for the limit to reset or upgrade your plan before continuing."
        )
        print("=" * 80 + "\n")

        # Cleanup and exit
        self.cleanup()
        exit(1)

    def _wait_for_login_completion(self):
        """Wait for the user to complete the login process."""
        try:
            if not self.driver:
                return

            logger.info("Waiting for login completion...")
            logger.info(
                "Please log in using the browser window. Press Enter when done or Ctrl+C to cancel."
            )

            # Wait for user confirmation that login is complete
            try:
                input("Press Enter after you have completed the login process...")
            except KeyboardInterrupt:
                logger.info("Login process cancelled by user")
                raise

            # Additional check - look for signs that we're logged in
            # Common indicators of successful login
            logged_in_indicators = [
                "textarea[placeholder*='message']",
                "[data-testid='prompt-textarea']",
                "#prompt-textarea",
                ".chat-input",
                "div[contenteditable='true']",
            ]

            login_successful = False
            # Try cached selector first
            cached_selector = self.selector_cache.get("logged_in_indicator")
            if cached_selector:
                try:
                    element = self.driver.find_element(By.CSS_SELECTOR, cached_selector)
                    if element and element.is_displayed():
                        login_successful = True
                except Exception:
                    logger.debug("Cached logged in indicator failed")
                    del self.selector_cache["logged_in_indicator"]

            if not login_successful:
                for indicator in logged_in_indicators:
                    try:
                        element = self.driver.find_element(By.CSS_SELECTOR, indicator)
                        if element and element.is_displayed():
                            self.selector_cache["logged_in_indicator"] = indicator
                            login_successful = True
                            break
                    except Exception:
                        continue

            if login_successful:
                logger.info("Login appears successful - chat interface detected")
            else:
                logger.warning("Could not confirm successful login - continuing anyway")

        except Exception as e:
            logger.warning(f"Error during login wait: {e}")

    def _wait_for_upload_completion_by_button(self) -> bool:
        """Wait for upload completion by monitoring the send button's disabled state."""
        try:
            if not self.driver:
                logger.error("Driver not available")
                return False

            send_button_selectors = [
                "#composer-submit-button",
                "[data-testid='send-button']",
                "button[aria-label*='Send']",
                "button[aria-label*='send']",
                "button[type='submit']",
            ]

            # Wait for button to become disabled (upload starting)
            start_time = time.time()
            upload_detected = False

            logger.info("Monitoring send button for upload progress...")

            while time.time() - start_time < 30:  # 30 second timeout
                send_button = None

                # Re-find the button on each iteration to avoid stale element references
                # Try cached selector first
                cached_selector = self.selector_cache.get("send_button")
                if cached_selector:
                    try:
                        send_button = self.driver.find_element(
                            By.CSS_SELECTOR, cached_selector
                        )
                    except Exception:
                        logger.debug("Cached send button selector failed in monitor")
                        del self.selector_cache["send_button"]

                if not send_button:
                    for selector in send_button_selectors:
                        try:
                            send_button = self.driver.find_element(
                                By.CSS_SELECTOR, selector
                            )
                            if send_button:
                                self.selector_cache["send_button"] = selector
                                break
                        except Exception:
                            continue

                if not send_button:
                    logger.warning("Could not find send button, retrying...")
                    time.sleep(0.5)
                    continue

                try:
                    if not send_button.is_enabled():
                        if not upload_detected:
                            logger.info("Upload detected - send button disabled")
                            upload_detected = True
                    else:
                        if upload_detected:
                            logger.info("Upload completed - send button re-enabled")
                            return True

                    time.sleep(0.5)  # Check every 500ms

                except Exception as e:
                    # Log stale element errors at debug level since they're expected
                    if "stale element reference" in str(e).lower():
                        logger.debug(f"Stale element reference (expected): {e}")
                    else:
                        logger.warning(f"Error checking button state: {e}")
                    time.sleep(0.5)

            if upload_detected:
                logger.warning("Upload may have completed but button state unclear")
                return True
            else:
                logger.warning("No upload activity detected via button monitoring")
                return False

        except Exception as e:
            logger.warning(f"Error monitoring upload via button state: {e}")
            return False

    def _check_for_unknown_error_after_upload(self) -> bool:
        """Check specifically for 'unknown error' messages in red alert boxes after upload."""
        try:
            if not self.driver:
                return False

            logger.debug("Checking for unknown error after upload...")

            # Specific selectors for red alert banners that appear after upload
            error_selectors = [
                "div[class*='border-red-500'][role='alert']",  # Red alert banners
                "div.border-red-500[role='alert']",  # Specific red error alerts
                "[role='alert'][class*='border-red-500']",  # Alert role with red border
            ]

            # Only check for "unknown error occurred" pattern
            unknown_error_pattern = "unknown error occurred"

            for selector in error_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for element in elements:
                        if element.is_displayed():
                            text = element.text.lower()
                            if unknown_error_pattern in text:
                                logger.error(
                                    f"Unknown error detected after upload: {element.text}"
                                )
                                return True
                except Exception:
                    continue

            return False

        except Exception as e:
            logger.warning(f"Error checking for unknown error after upload: {e}")
            return False
