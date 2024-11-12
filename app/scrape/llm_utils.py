from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List
from tenacity import retry, stop_after_attempt
import os
from utils.logging_config import setup_cloudwatch_logging

# Initialize logger
logger = setup_cloudwatch_logging('llm_utils')

openai_key = os.getenv('OPENAI_API_KEY')

############ Select likely URLs ############    

class URLListResponse(BaseModel):
    likely_urls : List[str] 

@retry(stop=stop_after_attempt(3))
def select_likely_URLS(prompt):
    """Detect good to go links from bad links."""
    try:
        logger.info("Starting URL selection process")
        
        # Patch the OpenAI client
        client = OpenAI(api_key=openai_key)
        
        logger.debug(f"Processing prompt: {prompt[:100]}...")  # Log first 100 chars
        
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "you are an assistant that is tasked with selecting a list of URLs that meet the criteria for likely news articles the most and are not suspected bot traps nor user related nor categories webpages."},
                {"role": "user", "content": prompt}
            ],
            response_format=URLListResponse,
            timeout=60,
            temperature=0.1618,
        )

        response = completion.choices[0].message.parsed
        logger.info(f"Successfully selected {len(response.likely_urls)} URLs")
        return response
        
    except Exception as e:
        logger.error(f"URL selection failed: {str(e)}", exc_info=True)
        raise

############ Detect login url  ############   

class FormFieldLoginUrl(BaseModel):
    login_url: str

@retry(stop=stop_after_attempt(3))
def detect_login_url(prompt):
    try:
        logger.info("Starting login URL detection")
        client = OpenAI(api_key=openai_key)

        completion = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Detect login url in the a list of urls"},
                {"role": "user", "content": prompt}
            ],
            response_format=FormFieldLoginUrl,
            timeout=20,
            temperature=0.1618,
        )
        response = completion.choices[0].message.parsed
        logger.info(f"looking for login url in {prompt}")
        logger.info(f"Login URL detected: {response}")
        return response
        
    except Exception as e:
        logger.error(f"Login URL detection failed: {str(e)}", exc_info=True)
        raise

############ Detect css selectors  ############   

class FormFieldInfoCredentials(BaseModel):
    username_selector: str
    password_selector: str
    submit_button_selector: str
    comment: str

@retry(stop=stop_after_attempt(3))
def detect_selectors(prompt):
    try:
        logger.info("Starting CSS selector detection")
        client = OpenAI(api_key=openai_key)

        completion = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Detect username field, password fields, and submit button css selectors in the cleaned HTML"},
                {"role": "user", "content": prompt}
            ],
            response_format=FormFieldInfoCredentials,
            timeout=20,
            temperature=0.1618,
        )
        response = completion.choices[0].message.parsed
        logger.info(f"looking for css selectors in {prompt}")
        logger.info(f"CSS selectors detected successfully : {response}")
        logger.debug(f"Detected selectors: {response}")
        return response
        
    except Exception as e:
        logger.error(f"Selector detection failed: {str(e)}", exc_info=True)
        raise

############ Classify page and Extract news article ############   

class FormFieldNewsArticleExtractor(BaseModel):
    classification: bool
    title: str
    author: str
    body: str
    date_published: str
    comment: str

@retry(stop=stop_after_attempt(3))
def classify_and_extract_news_article(prompt):
    try:
        logger.info("Starting article classification and extraction")
        client = OpenAI(api_key=openai_key)

        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "you are an assistant that is tasked with classifying a webpage as either a 'full Article webpage' or 'Not an Article webpage' based on its cleaned HTML content and extracting the full article content when it is an article webpage."},
                {"role": "user", "content": prompt}
            ],
            response_format=FormFieldNewsArticleExtractor,
            timeout=60,
            temperature=0.1618,
        )

        response = completion.choices[0].message.parsed
        
        if response.classification:
            logger.info(f"Article extracted successfully: {response.title}")
            logger.debug(f"Article details: {response}")
        else:
            logger.info("Page classified as non-article")
            
        return response
        
    except Exception as e:
        logger.error(f"Article extraction failed: {str(e)}", exc_info=True)
        raise

