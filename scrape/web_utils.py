import time
import logging
import logging
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import requests
from fake_useragent import UserAgent
from itertools import cycle
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from tenacity import retry, stop_after_attempt, wait_fixed, before_sleep_log
from bs4 import BeautifulSoup



from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from tenacity import retry, stop_after_attempt, wait_fixed, before_sleep_log

logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(2),
    before_sleep=before_sleep_log(logger, logging.ERROR),
)
def setup_driver_old(proxy_pool, user_agent_pool):
    """
    Configures the Selenium WebDriver with advanced anti-detection measures.

    Args:
        proxy_pool (iterator): An iterator that provides proxy strings.
        user_agent_pool (iterator): An iterator that provides user-agent strings.

    Returns:
        WebDriver: An instance of Selenium WebDriver configured for web scraping.
    """
    chrome_options = Options()

    # Anti-detection settings
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    #chrome_options.add_argument(f"user-agent={next(user_agent_pool)}")
    chrome_options.add_argument("--incognito")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--headless=new")  # Newer headless mode for better support
    chrome_options.add_argument("--disable-dev-shm-usage")  # Helps with resource constraints in some environments
    chrome_options.add_argument("--disable-infobars")  # Disable the "Chrome is being controlled by automated software" banner
    chrome_options.add_argument("--disable-extensions")  # Disable extensions that might be enabled by default

    # Setup proxies if needed
    #proxy = next(proxy_pool)
    #chrome_options.add_argument(f"--proxy-server={proxy}")

    # Experimental options to avoid detection
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)
    chrome_options.add_experimental_option("prefs", {
        "profile.default_content_setting_values.notifications": 2,  # Disable notifications
        "credentials_enable_service": False,  # Disable Chrome's password manager
        "profile.password_manager_enabled": False,
    })

    # Instantiate WebDriver with configured options
    driver = webdriver.Chrome(options=chrome_options)

    # Further anti-detection adjustments using DevTools Protocol
    #driver.execute_cdp_cmd(
    #    "Network.setUserAgentOverride", {"userAgent": next(user_agent_pool)}
    #)
    
    driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )

    # Optimize timeouts for better reliability in scraping
    driver.set_page_load_timeout(60)  # Adjust this as per the speed of the target websites
    driver.set_script_timeout(60)

    # Optional: Define window size to make headless mode less detectable
    driver.set_window_size(1920, 1080)

    # Log driver setup completion
    logging.info("WebDriver setup complete with the following configuration:")
    #logging.info(f"Proxy: {proxy}")
    #logging.info(f"User-Agent: {chrome_options.arguments[1]}")

    return driver
import logging
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from tenacity import retry, stop_after_attempt, wait_fixed, before_sleep_log

@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(2),
    before_sleep=before_sleep_log(logging, logging.ERROR),
)
def setup_driver(proxy_pool, user_agent_pool):
    """
    Configures the Selenium WebDriver with basic anti-detection measures.

    Returns:
        WebDriver: An instance of Selenium WebDriver configured for web scraping.
    """
    chrome_options = Options()

    # Basic anti-detection settings
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--incognito")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--headless=new")  # Newer headless mode for better support
    chrome_options.add_argument("--disable-dev-shm-usage")  # Helps with resource constraints in some environments
    chrome_options.add_argument("--disable-infobars")  # Disable the "Chrome is being controlled by automated software" banner
    chrome_options.add_argument("--disable-extensions")  # Disable extensions that might be enabled by default

    # Experimental options to avoid detection
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)
    chrome_options.add_experimental_option("prefs", {
        "profile.default_content_setting_values.notifications": 2,  # Disable notifications
        "credentials_enable_service": False,  # Disable Chrome's password manager
        "profile.password_manager_enabled": False,
    })

    # Instantiate WebDriver with configured options
    driver = webdriver.Chrome(options=chrome_options)

    # Further anti-detection adjustments using DevTools Protocol
    driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )

    # Optimize timeouts for better reliability in scraping
    driver.set_page_load_timeout(30)  # Adjust this as per the speed of the target websites
    driver.set_script_timeout(30)

    # Optional: Define window size to make headless mode less detectable
    driver.set_window_size(1920, 1080)

    # Log driver setup completion
    logger.info("WebDriver setup complete with default configuration.")

    return driver

@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(2),
    before_sleep=before_sleep_log(logger, logging.ERROR),
)
def get_html(url, driver):
    try:
        # Try to get the page using Selenium
        driver.get(url)
        
        # Wait until a specific element is present (e.g., body)
        time.sleep(3)  # Optional: if you want to mimic some delay

        # Get HTML content of the page
        html_content = driver.page_source
        
        return html_content
    except (TimeoutException, WebDriverException) as e:
        logger.exception(f"Exception while loading the page with Selenium: {url} - {e}")
        logger.exception("Falling back to requests.")

        try:
            # Fallback to using requests
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
            time.sleep(1)  # Optional: if you want to mimic some delay
            return response.text  # Return the content of the response
        except requests.exceptions.RequestException as e:
            logger.exception(f"Exception while loading the page with requests: {url} - {e}")
            return None  # Return None or handle the error as needed


@retry(
    stop=stop_after_attempt(2),
    wait=wait_fixed(2),
    before_sleep=before_sleep_log(logger, logging.ERROR),
)
def click_button(driver, identifier_type, identifier_value):
    """
    Clicks a button based on the identifier type and value.

    Args:
    driver (webdriver): The Selenium WebDriver.
    identifier_type (str): The type of identifier ('id', 'name', 'class_name', 'xpath', 'css_selector', 'link_text').
    identifier_value (str): The value of the identifier.

    Raises:
    Exception: If the identifier type is not recognized or button is not found.
    """
    try:
        human_like_interaction(driver)
        if "id" in str(identifier_type).strip().lower():
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, identifier_value))
            )
            element = driver.find_element(By.ID, identifier_value)

        elif "name" in str(identifier_type).strip().lower():
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.NAME, identifier_value))
            )
            element = driver.find_element(By.NAME, identifier_value)

        elif "class_name" in str(identifier_type).strip().lower():
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, identifier_value))
            )
            element = driver.find_element(By.CLASS_NAME, identifier_value)

        elif "xpath" in str(identifier_type).strip().lower():
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, identifier_value))
            )
            element = driver.find_element(By.XPATH, identifier_value)

        elif "css_selector" in str(identifier_type).strip().lower():
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, identifier_value))
            )
            element = driver.find_element(By.CSS_SELECTOR, identifier_value)

        elif "link_text" in str(identifier_type).strip().lower():
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.LINK_TEXT, identifier_value))
            )
            element = driver.find_element(By.LINK_TEXT, identifier_value)
        else:
            raise ValueError("Unsupported identifier type provided.")

    except NoSuchElementException:
        raise Exception(
            f"Button with {identifier_type}='{identifier_value}' not found."
        )
    try:
        element.click()
        time.sleep(2)
        current_url = driver.current_url
        return current_url
    except Exception:
        raise Exception(f"`login in URL cound not be retrieved`")
        return False


@retry(
    stop=stop_after_attempt(2),
    wait=wait_fixed(2),
    before_sleep=before_sleep_log(logger, logging.ERROR),
)
def authentificate(
    driver,
    login_url,
    username_field_id,
    username,
    password_field_id,
    password,
    submit_button_id,
):
    driver.get(login_url)
    # Human-like interaction before filling in the form
    human_like_interaction(driver)

    # Step 2: Locate and fill in the username field
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, response.username_field_id))
    )
    username_input = driver.find_element(By.ID, response.username_field_id)
    username_input.send_keys(username)

    # Human-like delay
    time.sleep(random.uniform(1, 3))

    # Step 3: Locate and fill in the password field
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, response.password_field_id))
    )
    password_input = driver.find_element(By.ID, response.password_field_id)
    password_input.send_keys(password)

    # Human-like delay
    time.sleep(random.uniform(1, 3))

    # Step 4: Locate and click the submit button
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, response.submit_button_id))
    )
    submit_button = driver.find_element(By.ID, response.submit_button_id)
    submit_button.click()

    # Allow the page to process the login
    time.sleep(random.uniform(5, 7))

    # Capture session cookies and headers after successful login
    logger.info("Submit button clicked. Capturing session cookies.")
    for cookie in driver.get_cookies():
        session.cookies.set(cookie["name"], cookie["value"])

    # Capture other important headers if needed
    session.headers.update(
        {
            "User-Agent": driver.execute_script("return navigator.userAgent;"),
        }
    )

    logger.info("Session cookies and headers captured. Proceeding to scrape data.")
    return session


import random
import time
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.common.exceptions import MoveTargetOutOfBoundsException

import random
import time
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.common.exceptions import (
    MoveTargetOutOfBoundsException,
    JavascriptException,
    NoSuchElementException,
    ElementNotInteractableException,
    WebDriverException,
)


def human_like_interaction(driver):
    """
    Simulates human-like interactions to avoid detection.

    Args:
        driver (webdriver): Selenium WebDriver instance.
    """
    try:
        # Randomly scroll the page
        scroll_pause_time = random.uniform(1, 3)
        last_height = driver.execute_script("return document.body.scrollHeight")

        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(scroll_pause_time)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        # Randomly move the mouse with bounds checking
        action = ActionChains(driver)
        try:
            action.move_by_offset(
                random.randint(-100, 100), random.randint(-100, 100)
            ).perform()
        except MoveTargetOutOfBoundsException:
            # Handle the exception by resetting the mouse position to a visible part of the page
            action.move_by_offset(0, 0).perform()

        time.sleep(random.uniform(0.5, 1.5))

        # Randomly interact with an element (click or hover) with bounds and visibility checking
        elements = driver.find_elements(By.XPATH, "//*")
        visible_elements = [
            el
            for el in elements
            if el.is_displayed() and el.size["width"] > 0 and el.size["height"] > 0
        ]

        if visible_elements:
            random_element = random.choice(visible_elements)

            try:
                # Ensure the element is within the viewport
                driver.execute_script(
                    "arguments[0].scrollIntoView({block: 'center'});", random_element
                )
                action.move_to_element(random_element).perform()
                if random.random() > 0.9:
                    random_element.click()
            except (
                MoveTargetOutOfBoundsException,
                JavascriptException,
                ElementNotInteractableException,
            ) as e:
                # If there's an exception, log it and continue without action
                print(f"Error interacting with element: {e}")

    except (NoSuchElementException, WebDriverException) as e:
        # Catch broader exceptions and log them to avoid breaking the script
        print(f"General interaction error: {e}")
    except Exception as e:
        # Log any unexpected exceptions
        print(f"Unexpected error during human-like interaction: {e}")


def rotate_headers():
    """
    Generates a new set of headers with a random User-Agent.

    Returns:
        dict: Headers dictionary with rotated User-Agent.
    """
    return {
        "User-Agent": next(user_agent_pool),
        "Accept-Language": random.choice(
            ["en-US,en;q=0.9", "en-GB,en;q=0.8", "en-CA,en;q=0.7"]
        ),
        "Referer": "https://www.google.com",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }


from selenium.common.exceptions import WebDriverException


def retry_request(url, session, driver, proxy_pool, retries=3):
    """
    Attempts to send a GET request or navigate to a URL with retries in case of failure using either
    requests.Session or Selenium WebDriver.

    Args:
        url (str): URL to send the request to or navigate to.
        retries (int): Number of retry attempts.
        session (requests.Session, optional): Session object for HTTP requests.
        driver (selenium.webdriver, optional): WebDriver instance for web navigation.
        proxy_pool (iterator, optional): Iterator to rotate proxies.

    Returns:
        str: The response object HTML as text or None if all retries fail.
    """
    for attempt in range(retries):
        try:
            if session:
                if proxy_pool:
                    proxy = next(proxy_pool)
                    session.proxies = {"http": proxy, "https": proxy}
                response = session.get(url)
                if response.status_code == 200:
                    return response.text
                else:
                    logger.warning(
                        f"HTTP request failed with status code {response.status_code}. Retrying ({attempt + 1}/{retries})..."
                    )
            elif driver:
                driver.get(url)
                # Additional check to ensure the page has loaded properly could be added here
                return driver.page_source
            else:
                raise ValueError(
                    "Neither session nor driver is provided for making requests."
                )

        except (requests.RequestException, WebDriverException) as e:
            logger.error(
                f"An error occurred: {e}. Retrying ({attempt + 1}/{retries})..."
            )

        time.sleep(random.uniform(1, 3))  # Random delay before retrying

    return None
