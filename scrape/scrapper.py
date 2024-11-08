import time
import logging
import random
import os
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
from .python_utils import filter_and_normalize_links
from bs4 import BeautifulSoup
from .web_utils import setup_driver, human_like_interaction, retry_request, get_html
from collections import deque
from .llm_tools import classify_page, extract_article, select_likely_URLS
from typing import List
BUCKET_NAME = os.getenv("BUCKET_NAME")
logger = logging.getLogger(__name__)


def scrape(row, driver, alr_visited, session, proxy_pool, max_pages):
    start_url = row.get("website")
    # Initialize deque for efficient popping from the left
    to_visit = deque([start_url])
    visited_urls = set(alr_visited)

    articles = []

    while to_visit and len(articles) < max_pages:
        current_url = to_visit.popleft()

        if (current_url in visited_urls):
            continue

        logger.info(f"Crawling: {current_url}")
        try:
            html_content = get_html(current_url, driver)
        except Exception as e:
            logger.exception(f"An error occurred during html retrieving of: {current_url}")
            try :
                html_content = retry_request(
                    current_url, session, driver, proxy_pool, retries=1
                )
            except : 
                pass
        if not html_content:
            continue

        # Find more links to crawl after
        soup = BeautifulSoup(html_content, "html.parser")
        links = soup.find_all("a", href=True)

        # Normalize and filter links before adding to queue
        new_urls = list(filter_and_normalize_links(start_url, links, visited_urls))
        logger.debug(f"list of links is {str(new_urls)[:500]} ")

        filtered_urls = select_likely_URLS(new_urls)

        if filtered_urls :
            model_output_list = filtered_urls.likely_urls
            unwanted = [item for item in new_urls if item not in model_output_list]
            hallucinated = [item for item in model_output_list  if item not in new_urls ]
            new_urls = [item for item in model_output_list if item in new_urls]
            new_urls = set(new_urls)

            logger.debug(f"{len(new_urls)} new_urls is : {new_urls}")
            logger.debug(f"{len(unwanted)} unwanted is : {unwanted}")
            logger.debug(f"{len(hallucinated)} Hallucinated are : {hallucinated}")

            visited_urls.update(unwanted)
        else :
            logger.debug(f"LLM couldn't assist in selecting links")
        for url in list(new_urls):
            to_visit.append(url)
         
        if current_url == start_url:
            continue

        try : 
            classification = classify_page(current_url, html_content)
            visited_urls.add(current_url)
            time.sleep(1)
            if classification:
                logger.debug(f"Classification result is : {classification}")

                if  classification.success:
                    articles.append(current_url)
                    logger.debug(f"Classification result is : {classification}")
                    extraction = extract_article(current_url, html_content, BUCKET_NAME)
                else :
                    logger.debug(f"Classification result is : {classification}")
        except Exception as e : 
            logger.exception(f"Classification failed because :{e}")
        #human_like_interaction(driver)
        time.sleep(random.uniform(1, 3))  # Be polite to the server

    driver.quit()
    return row, visited_urls
