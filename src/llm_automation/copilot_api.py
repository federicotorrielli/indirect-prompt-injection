"""
Microsoft Copilot automation API using Selenium WebDriver.
Handles PDF uploads and review requests.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import undetected_chromedriver as uc
from config import Config
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

logger = logging.getLogger(__name__)


class CopilotAutomator:
    """Handles automated interactions with Microsoft Copilot web interface."""

    def __init__(self, config: Config):
        self.config = config
        self.driver: Optional[WebDriver] = None
        self.selector_cache: Dict[str, Any] = {}
        # Update URL for Copilot
        self.copilot_url = "https://copilot.microsoft.com/"

    def initialize(self) -> bool:
        """Initialize the WebDriver and navigate to Copilot."""
        try:
            logger.info("Setting up Chrome driver for Copilot...")

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

                logger.info("Navigating to Copilot...")
                self.driver.get(self.copilot_url)

            # Wait for page to load
            time.sleep(3)

            # Check for usage limits immediately after loading
            usage_error = self._check_for_usage_limits()
            if usage_error:
                self._handle_usage_limit_error(usage_error)

            # Check if we need to handle any initial dialogs
            self._handle_initial_dialogs()

            logger.info("Copilot automation initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Copilot automator: {e}")
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
            logger.info("Copilot automator cleaned up")

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

            # Copilot-specific selectors based on the HTML structure provided
            chat_indicators = [
                "#userInput",  # Main textarea with id="userInput"
                "[data-testid='composer-input']",  # data-testid for the input
                "textarea[placeholder*='Message Copilot']",  # Placeholder text
                "div[data-testid='composer']",  # Composer container
            ]

            for selector in chat_indicators:
                try:
                    element = WebDriverWait(self.driver, 1).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                    )
                    if element and element.is_displayed():
                        logger.debug(f"Found chat interface using selector: {selector}")
                        self.selector_cache["input_area"] = (selector, By.CSS_SELECTOR)
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
                ("//button[contains(text(), 'Sign in')]", By.XPATH),
                ("//a[contains(text(), 'Sign in')]", By.XPATH),
                ("//a[contains(@href, 'login')]", By.XPATH),
                ("//button[contains(text(), 'Log in')]", By.XPATH),
            ]

            # Try cached selector first
            cached = self.selector_cache.get("login_button")
            if cached:
                cached_selector, by = cached
                try:
                    element = WebDriverWait(self.driver, 1).until(
                        EC.presence_of_element_located((by, cached_selector))
                    )
                    if element and element.is_displayed():
                        return True
                except Exception:
                    del self.selector_cache["login_button"]

            for selector, by in login_selectors:
                try:
                    element = WebDriverWait(self.driver, 1).until(
                        EC.presence_of_element_located((by, selector))
                    )
                    if element and element.is_displayed():
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
                time.sleep(2)
                return True

        except TimeoutException:
            logger.debug("No 'Stay logged out' button found")
        except Exception as e:
            logger.warning(f"Error handling 'Stay logged out': {e}")

        return False

    def _wait_for_login_completion(self, max_wait_time: int = 300):
        """Wait for user to complete login process."""
        logger.info("Waiting for login completion...")
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            try:
                if self._is_chat_interface_ready():
                    logger.info("Login completed successfully")
                    return True
                time.sleep(2)
            except Exception as e:
                logger.debug(f"Error checking login status: {e}")

        logger.error("Login did not complete within the expected time")
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
                raise Exception("Driver not initialized")

            # Copilot-specific attachment button selectors
            attachment_selectors = [
                "[data-testid='plus-button']",  # Plus button that opens upload
                "button[title='Attach files']",
                "button[aria-label='Attach files']",
                "button[title='Upload files']",
                "button[aria-label='Upload files']",
                ".attachment-button",
                "button[class*='attach']",
                "input[type='file']",  # Sometimes the file input is directly accessible
            ]

            for selector in attachment_selectors:
                try:
                    element = self.wait.until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                    )
                    if element and element.is_displayed() and element.is_enabled():
                        logger.debug(f"Found attachment button using: {selector}")
                        return element
                except TimeoutException:
                    continue

            raise Exception("Could not find attachment button")

        except Exception as e:
            logger.error(f"Error finding attachment button: {e}")
            raise

    def _is_attachment_button_disabled(self) -> bool:
        """Check if attachment button is disabled."""
        try:
            attachment_button = self._find_attachment_button()
            return not attachment_button.is_enabled()
        except Exception:
            return True

    def _find_input_area(self):
        """Find the text input area."""
        try:
            if not self.driver:
                raise Exception("Driver not initialized")

            # Try cached selector first
            cached = self.selector_cache.get("input_area")
            if cached:
                cached_selector, by = cached
                try:
                    element = self.wait.until(
                        EC.presence_of_element_located((by, cached_selector))
                    )
                    if element and element.is_displayed():
                        return element
                except Exception:
                    del self.selector_cache["input_area"]

            # Copilot-specific input selectors
            input_selectors = [
                "#userInput",
                "[data-testid='composer-input']",
                "textarea[placeholder*='Message Copilot']",
            ]

            for selector in input_selectors:
                try:
                    element = self.wait.until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                    )
                    if element and element.is_displayed():
                        logger.debug(f"Found input area using: {selector}")
                        self.selector_cache["input_area"] = (selector, By.CSS_SELECTOR)
                        return element
                except TimeoutException:
                    continue

            raise Exception("Could not find input area")

        except Exception as e:
            logger.error(f"Error finding input area: {e}")
            raise

    def _find_send_button(self):
        """Find the send button."""
        try:
            if not self.driver:
                raise Exception("Driver not initialized")

            # Copilot-specific send button selectors
            send_selectors = [
                "[data-testid='submit-button']",
                "button[title='Submit message']",
                "button[aria-label='Submit message']",
                "button.rounded-submitButton",  # Based on the class in HTML
            ]

            for selector in send_selectors:
                try:
                    element = self.wait.until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                    )
                    if element and element.is_displayed() and element.is_enabled():
                        logger.debug(f"Found send button using: {selector}")
                        return element
                except TimeoutException:
                    continue

            raise Exception("Could not find send button")

        except Exception as e:
            logger.error(f"Error finding send button: {e}")
            raise

    def upload_pdf_and_request_review(self, pdf_path: str, review_request: str) -> str:
        """Upload a PDF and request a review with retry logic."""
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                if not self._ensure_driver_alive():
                    raise Exception("Driver not available")

                # Ensure interface is ready before starting
                if not self._validate_ready_for_input():
                    logger.warning(
                        f"Interface not ready for attempt {attempt + 1}, preparing..."
                    )
                    if not self._prepare_for_new_conversation():
                        if attempt < max_retries:
                            logger.warning(f"Attempt {attempt + 1} failed, retrying...")
                            continue
                        raise Exception("Could not prepare interface for input")

                logger.info(
                    f"Uploading PDF: {Path(pdf_path).name} (attempt {attempt + 1})"
                )

                # Convert to absolute path
                abs_pdf_path = os.path.abspath(pdf_path)

                # Check if PDF file exists
                if not os.path.exists(abs_pdf_path):
                    raise FileNotFoundError(f"PDF file not found: {abs_pdf_path}")

                # Step 1: Find and click the attachment button
                attachment_button = self._find_attachment_button()
                attachment_button.click()
                time.sleep(1)

                # Step 2: Handle file input
                file_input_selectors = [
                    "input[type='file']",
                    "input[accept*='pdf']",
                    "input[accept*='.pdf']",
                ]

                file_input = None
                for selector in file_input_selectors:
                    try:
                        file_input = self.wait.until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                        )
                        if file_input:
                            break
                    except TimeoutException:
                        continue

                if not file_input:
                    raise Exception(
                        "Could not find file input after clicking attachment button"
                    )

                # Upload the file using absolute path
                file_input.send_keys(abs_pdf_path)

                # Wait for file upload confirmation element to appear
                logger.info("Waiting for file upload confirmation...")
                self._wait_for_file_upload_confirmation(Path(pdf_path).name)

                # Enter the review request text
                logger.info("Entering review request...")
                input_area = self._find_input_area()
                input_area.clear()
                input_area.send_keys(review_request)

                # Click send button
                logger.info("Sending request...")
                send_button = self._find_send_button()
                send_button.click()

                # Wait for response
                logger.info("Waiting for response...")
                response = self._wait_for_response()

                logger.info("Successfully received response")
                return response

            except Exception as e:
                logger.error(
                    f"Error in upload_pdf_and_request_review attempt {attempt + 1}: {e}"
                )
                if attempt < max_retries:
                    logger.info(
                        f"Retrying in 3 seconds... ({attempt + 1}/{max_retries})"
                    )
                    time.sleep(3)
                    continue
                raise

        # This should never be reached due to the raise above, but satisfy type checker
        raise Exception("All retry attempts failed")

    def _wait_for_response(self) -> str:
        """Wait for and extract the response from Copilot."""
        try:
            start_time = time.time()
            max_wait = self.config.response_timeout

            # Wait for response to appear and complete
            while time.time() - start_time < max_wait:
                try:
                    # Look for Copilot's specific response structure
                    response_selectors = [
                        "div.group\\/ai-message-item",  # Main response container
                        "div[class*='group/ai-message-item']",  # Alternative class matching
                        "[id*='-content-']",  # Content containers with specific ID pattern
                        "div[class*='space-y-3'][class*='break-words']",  # Response content area
                    ]

                    if not self.driver:
                        continue

                    for selector in response_selectors:
                        try:
                            response_elements = self.driver.find_elements(
                                By.CSS_SELECTOR, selector
                            )
                            if response_elements:
                                # Get the latest response element
                                latest_response = response_elements[-1]

                                # Check if this response is still being generated
                                if self._is_response_still_generating(latest_response):
                                    continue

                                response_text = latest_response.text.strip()

                                if (
                                    response_text and len(response_text) > 50
                                ):  # Increased threshold for meaningful response
                                    logger.info("Response received and complete")
                                    # Prepare for next conversation
                                    self._prepare_for_new_conversation()
                                    return response_text

                        except Exception:
                            continue

                    time.sleep(2)

                except Exception as e:
                    logger.debug(f"Error while waiting for response: {e}")
                    time.sleep(2)

            raise TimeoutException(f"No response received within {max_wait} seconds")

        except Exception as e:
            logger.error(f"Error waiting for response: {e}")
            raise

    def _is_response_still_generating(self, response_element) -> bool:
        """Check if the response is still being generated."""
        try:
            if not self.driver:
                return False

            # Check for the "stop button" (interrupt message button) which indicates response is still generating
            stop_button_selectors = [
                "[data-testid='stop-button']",  # The interrupt message button
                "button[title='Interrupt message']",  # Alternative selector
                "button[aria-label='Interrupt message']",  # Another alternative
            ]

            # Check in the whole page, not just the response element
            for selector in stop_button_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        for element in elements:
                            if element.is_displayed():
                                logger.debug(
                                    "Response still generating - found stop button"
                                )
                                return True
                except Exception:
                    continue

            return False

        except Exception:
            return False

    def _check_for_usage_limits(self) -> Optional[str]:
        """Check for usage limit errors or restrictions."""
        try:
            if not self.driver:
                return None

            error_indicators = [
                "rate limit",
                "usage limit",
                "too many requests",
                "please wait",
                "try again later",
                "quota exceeded",
            ]

            page_text = self.driver.find_element(By.TAG_NAME, "body").text.lower()

            for indicator in error_indicators:
                if indicator in page_text:
                    return f"Usage limit detected: {indicator}"

            return None

        except Exception:
            return None

    def _handle_usage_limit_error(self, error_message: str):
        """Handle usage limit errors."""
        logger.error(f"Usage limit encountered: {error_message}")
        logger.error("Copilot automation cannot continue due to usage limits")

        # Clean up and exit
        self.cleanup()
        import sys

        sys.exit(1)

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
        """Start a new conversation."""
        try:
            if not self.driver:
                raise Exception("Driver not initialized")

            # Step 1: Click the actions menu button (three dots menu)
            actions_menu_selectors = [
                "button[title='Open actions menu']",
                "button[aria-haspopup='dialog']",
                "button[aria-expanded='false'][aria-haspopup='dialog']",
            ]

            actions_button = None
            for selector in actions_menu_selectors:
                try:
                    actions_button = self.wait.until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                    )
                    if actions_button and actions_button.is_displayed():
                        logger.debug(f"Found actions menu using: {selector}")
                        break
                except TimeoutException:
                    continue

            if not actions_button:
                logger.warning(
                    "No actions menu button found, unable to start new conversation"
                )
                return False

            # Click the actions menu
            actions_button.click()
            time.sleep(0.4)

            # Step 2: Click "Create new conversation" button
            new_conversation_selectors = [
                "button[title='Create new conversation']",
                "button:contains('Create new conversation')",
                "//button[contains(text(), 'Create new conversation')]",
            ]

            for selector in new_conversation_selectors:
                try:
                    if selector.startswith("//"):
                        # XPath selector
                        button = self.wait.until(
                            EC.element_to_be_clickable((By.XPATH, selector))
                        )
                    else:
                        # CSS selector
                        button = self.wait.until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                        )

                    if button and button.is_displayed():
                        button.click()
                        time.sleep(2)
                        logger.info("Started new conversation")
                        return True
                except TimeoutException:
                    continue

            logger.warning(
                "Create new conversation button not found after opening menu"
            )
            return False

        except Exception as e:
            logger.error(f"Error starting new conversation: {e}")
            return False

    def _wait_for_file_upload_confirmation(self, filename: str):
        """Wait for the file upload confirmation element to appear."""
        try:
            if not self.driver:
                logger.warning("Driver not available for upload confirmation check")
                return

            start_time = time.time()
            max_wait = self.config.upload_timeout

            while time.time() - start_time < max_wait:
                try:
                    # Look for the uploaded file confirmation element
                    # Based on the HTML structure provided - look for the file display container
                    file_confirmation_selectors = [
                        f"p:contains('{filename}')",  # Text containing the filename
                        "div[class*='relative'][class*='flex'][class*='justify-center'][class*='gap-2']",  # Main container
                        "p.font-ligatures-none",  # Filename paragraph with specific class
                        "button[title='Remove file']",  # Remove button indicates file is uploaded
                    ]

                    for selector in file_confirmation_selectors:
                        try:
                            if selector.startswith("p:contains"):
                                # Use XPath for text content search
                                xpath = f"//p[contains(text(), '{filename}')]"
                                element = WebDriverWait(self.driver, 2).until(
                                    EC.presence_of_element_located((By.XPATH, xpath))
                                )
                            else:
                                element = WebDriverWait(self.driver, 2).until(
                                    EC.presence_of_element_located(
                                        (By.CSS_SELECTOR, selector)
                                    )
                                )

                            if element and element.is_displayed():
                                logger.info(
                                    f"File upload confirmed - found element using: {selector}"
                                )
                                return
                        except TimeoutException:
                            continue

                    time.sleep(1)

                except Exception as e:
                    logger.debug(f"Error checking for upload confirmation: {e}")
                    time.sleep(1)

            logger.warning(
                f"Upload confirmation not found within {max_wait} seconds, proceeding anyway"
            )

        except Exception as e:
            logger.warning(
                f"Error waiting for upload confirmation: {e}, proceeding anyway"
            )

    def _validate_ready_for_input(self) -> bool:
        """Validate that the interface is ready to accept input."""
        try:
            if not self.driver:
                return False

            # Check if we can find the input area
            input_area = self._find_input_area()
            if not input_area:
                return False

            # Check if input area is enabled and visible
            return input_area.is_displayed() and input_area.is_enabled()

        except Exception:
            return False

    def _prepare_for_new_conversation(self) -> bool:
        """Prepare the interface for a new conversation, with fallback to refresh."""
        try:
            # First attempt: try to start a new conversation properly
            if self.start_new_conversation():
                # Validate that we're ready for input after starting new conversation
                if self._validate_ready_for_input():
                    return True
                else:
                    logger.warning(
                        "Interface not ready after starting new conversation"
                    )

            # Fallback: refresh page if new conversation failed or interface not ready
            logger.info(
                "New conversation failed or interface not ready, falling back to page refresh"
            )
            self.refresh_page()

            # Validate again after refresh
            return self._validate_ready_for_input()

        except Exception as e:
            logger.error(f"Error preparing for new conversation: {e}")
            # Last resort: refresh page
            try:
                self.refresh_page()
                return self._validate_ready_for_input()
            except Exception as refresh_error:
                logger.error(f"Page refresh also failed: {refresh_error}")
                return False
