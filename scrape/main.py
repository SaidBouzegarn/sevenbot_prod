import csv
import os
from src.logger_config import setup_logging_prod, setup_logging
import logging
import boto3
from dotenv import load_dotenv
from typing import Optional, Tuple
from src.login import (
    log_in,
    extract_credentials_ids,
)
from src.web_utils import setup_driver, setup_driver_old, get_html
from src.scrapper import scrape
from src.s3_utils import read_csv_from_s3, read_json_from_s3, write_csv_to_s3
import time
import time
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



# Load environment variables from .env file
load_dotenv()

# Replace with your OpenAI API key
openai_key = os.getenv("OPENAI_API_KEY")
BUCKET_NAME = os.getenv("BUCKET_NAME")
DATAFRAME_KEY = os.getenv("DATAFRAME_KEY")
VISITED_URLS_KEY = os.getenv("VISITED_URLS_KEY")
DO_NOT_VISIT_KEY = os.getenv("DO_NOT_VISIT_KEY")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")
PROXIES = os.getenv("PROXIES")
MAX_PAGES_CRAWL = int(os.getenv("MAX_PAGES_CRAWL"))

# Set up User-Agent rotation
ua = UserAgent()
user_agents = [ua.chrome, ua.firefox, ua.safari, ua.edge]
user_agent_pool = cycle(user_agents)

# Set up proxy rotation (proxy list)

proxy_pool = cycle(PROXIES)

def laod_files(logger):

    ##### reading websites dataframe
    try:
        data = read_csv_from_s3(BUCKET_NAME, DATAFRAME_KEY)
        logger.info(f"website dataframe finished loading")
        logger.info(f"{data.head()}")
    except Exception as e:
        error_message = f"Erreur lors du chargement du dataframe des sites web: {str(e)}"
        logger.error(error_message)
        logger.error(f"Traceback complet:\n{traceback.format_exc()}")
    finally:
        logger.info("Fin de la tentative de lecture du dataframe des sites web")

    ###### reading already visited URLS

    try:
        already_visited = read_json_from_s3(BUCKET_NAME, VISITED_URLS_KEY)
        logger.info(f"visited websites list finished loading")
        logger.info(f"{already_visited}")

    except Exception as e:
        error_message = (
            f"Erreur lors du chargement du dataframe des sites web visités: {str(e)}"
        )
        logger.error(error_message)
        logger.error(f"Traceback complet:\n{traceback.format_exc()}")
    finally:
        logger.info("Fin de la tentative de lecture du dataframe des sites web visités")
    return data, already_visited

def main():
    #setup_logging_prod()
    setup_logging_prod()
    logger = logging.getLogger(__name__)

    logger.info("Main function started")

    logger.info(f"BUCKET_NAME : {BUCKET_NAME}")
    logger.info(f"DATAFRAME_KEY : {DATAFRAME_KEY}")
    logger.info(f"VISITED_URLS_KEY : {VISITED_URLS_KEY}")
    logger.info(f"AWS_DEFAULT_REGION : {AWS_DEFAULT_REGION}")
    logger.info(f"PROXIES : {PROXIES}")


    df, visited_urls = laod_files(logger)
    ###### Processing rows
    for index, row in df.iterrows():
        website = row.get("website")

        logger.info(f"""
         ====================================================
         ====================================================
                        Scrapping : {website}
         ====================================================
         ====================================================
        """)

        driver = setup_driver_old(proxy_pool, user_agent_pool)
        website_html = get_html(website, driver)
        logger.info(f"getting website html : {website}")

        result = log_in(row, driver, website_html)
        if isinstance(result, tuple) and len(result) == 2:
            new_row, session = result
        else:
            # Handle the login failure, perhaps by setting new_row and session to None or logging an error
            new_row, session = row, None
            logger.warning(f"Login failed for the given {row}")
        
        new_row, alr_visited = scrape(
            new_row, driver, visited_urls, session, proxy_pool, MAX_PAGES_CRAWL
        )
        if isinstance(result, tuple) and len(result) == 2:
            alr_visited , donot_visit = result
        else:
            # Handle the login failure, perhaps by setting new_row and session to None or logging an error
            logger.warning(f"Screapping failed for the given {row}")
        df.loc[index] = new_row

    return df, visited_urls


if __name__ == "__main__":

    data, already_visited = main()
    write_csv_to_s3(data, BUCKET_NAME, DATAFRAME_KEY)
    write_json_to_s3(already_visited, BUCKET_NAME, VISITED_URLS_KEY)
