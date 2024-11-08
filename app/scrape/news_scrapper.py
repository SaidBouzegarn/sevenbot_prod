from playwright.sync_api import sync_playwright
from playwright_stealth import stealth_sync
from jinja2 import Environment, FileSystemLoader
import os
from utils import extract_content, clean_html_for_login_detection
from llm_utils import select_likely_URLS, detect_login_url, detect_selectors, classify_and_extract_news_article
import sqlite3
from datetime import datetime
from collections import deque
from urllib.parse import urlparse
import traceback
import dotenv
import time
import random

dotenv.load_dotenv()

class NewsScrapper:
    def __init__(self, website_url, login_url=None, username=None, password=None,
                 username_selector=None, password_selector=None, submit_button_selector=None, crawl=True, max_pages=25):
        self.website_url = website_url
        self.domain = self._extract_domain(website_url)
        self.login_url = login_url
        self.username = username
        self.password = password
        self.username_selector = username_selector
        self.password_selector = password_selector
        self.submit_button_selector = submit_button_selector
        self.crawl = crawl
        self.session_cookies = None
        self.max_pages = max_pages
        self.jinja_env = Environment(loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), '..', 'Data', 'scrapping_prompts')))
        self.headless = True  # Set to False if you want to see the browser actions
        self.db_path = os.path.join(os.path.dirname(__file__), '..', 'Data', 'dbs' 'news_scrapper.db')

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._initialize_database()

        #delete the website url from the visited urls set, its only url we allow to be visited multiple times
        self.visited_urls = set(self.get_visited_urls()) - {self.website_url}

        # Initialize Playwright browser with anti-bot measures
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(
            headless=self.headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-infobars",
                "--disable-extensions",
                "--disable-gpu",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-setuid-sandbox",
            ],
        )

        # Create a new context with stealth settings
        self.context = self.browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) " +
                       "AppleWebKit/537.36 (KHTML, like Gecko) " +
                       "Chrome/96.0.4664.45 Safari/537.36",
            viewport={"width": 1920, "height": 1080},
            locale="en-US",
        )

        # Open a new page and navigate to the website
        self.page = self.context.new_page()
        self.page.goto(self.website_url)
        self.page.wait_for_load_state('networkidle')  # Wait for the page to be fully loaded
        
        # Apply stealth techniques
        stealth_sync(self.page)

        # Check authentication requirements and login if needed
        if self._can_authenticate():
            if self._initialize_login():
                self.login()
        else:
            print("No authentication credentials provided. Running in anonymous mode.")

    def _can_authenticate(self):
        """Check if authentication is possible with provided credentials."""
        return self.username is not None and self.password is not None

    def _initialize_login(self):
        """Initialize login requirements before attempting authentication."""
        if not self.login_url:
            try: 
                self.login_url = self.get_login_url()
            except Exception as e:
                print(f"Warning: Could not determine login URL. Continuing without authentication...Full error:\n{traceback.format_exc()}")
                return False
        if not isinstance(self.login_url, str):
            print(f"Warning: Login URL is not a string and its type is {type(self.login_url)}. Continuing without authentication...")
            return False
        # Verify all required selectors are present
        if not self.username_selector or not self.password_selector or not self.submit_button_selector:
            try:
                # Detect login selectors
                selectors = self.get_login_selectors()
                if not all(selectors):  # Check if any selector is None or empty
                    print("Warning: One or more login selectors are empty or invalid")
                    return False
                    
                self.username_selector, self.password_selector, self.submit_button_selector = selectors
                return True
                
            except Exception as e:
                print(f"Error detecting login selectors: {e}")
                return False

        return True  # Add explicit return True when all checks pass

    def get_login_url(self):
        """Find and return the login URL for the website."""
        links = extract_content(self.page, output_type="links")
        html_content = extract_content(self.page, output_type="full_html")
        prompt = self.jinja_env.get_template('get_login_url_prompt.j2').render(
            #links=links,
            links = html_content
        )

        response = detect_login_url(prompt)
        print(response)
        return response.login_url
    
    def get_login_selectors(self):

        self.page.goto(self.login_url)
        self.page.wait_for_load_state('networkidle')
        # Extract page content
        page_content = extract_content(self.page, output_type="full_html")
        # Clean HTML for login detection
        cleaned_html = clean_html_for_login_detection(page_content)
        # Generate prompt for login selectors
        print(f"the given html for form detection is {cleaned_html}")

        prompt = self.jinja_env.get_template('login_selectors_prompt.j2').render(
            html=cleaned_html
        )
        # Detect login selectors
        response = detect_selectors(prompt)
        print(response)
        print(f"found selectors: username: {response.username_selector}, password: {response.password_selector}, submit_button: {response.submit_button_selector}")
        return response.username_selector, response.password_selector, response.submit_button_selector

    def login(self):
        # Navigate to the login page
        self.page.goto(self.login_url)

        # Wait for the login page to load
        self.page.wait_for_load_state('networkidle')

        # Fill in the username and password fields
        self.page.fill(self.username_selector, self.username)
        self.page.fill(self.password_selector, self.password)

        # Click the submit button
        self.page.click(self.submit_button_selector)

        # Wait for the navigation to complete
        self.page.wait_for_load_state('networkidle')

        # Add session verification
        self._verify_session()
        
    def _verify_session(self):
        """Verify that the session is still active."""
        # Store cookies after successful login
        self.session_cookies = self.context.cookies()
        
        # Basic session check
        if not self.session_cookies:
            raise Exception("No session cookies found after login")
            
    def scrape(self):
        # Only restore cookies if they exist and are valid
        if hasattr(self, 'session_cookies') and self.session_cookies:
            self.context.add_cookies(self.session_cookies)
        
        responses = []
        time.sleep(random.uniform(0, 2))
        self.page.goto(self.website_url)
        self.page.wait_for_load_state('networkidle')  # Wait for the page to be fully loaded
        urls = extract_content(self.page, output_type="links")
        print(f"found links {(urls)} ")
        prompt = self.jinja_env.get_template('articles_links_prompt.j2').render(
            urls=urls
        )
        response = select_likely_URLS(prompt)
        print(response)

        lucky_urls = response.likely_urls
        url_list = [link['href'] for link in urls]

        # Filter lucky_urls to only include URLs that exist in 'urls'
        if isinstance(lucky_urls, list):
            n_lucky_urls = [url for url in lucky_urls if url in url_list]
        else:
            n_lucky_urls = [lucky_urls] if lucky_urls in url_list else []

        print(f"found {len(n_lucky_urls)} new urls")
        print(f"llm hallucinated {len(lucky_urls) - len(n_lucky_urls)} urls")
        # Initialize deque for efficient popping from the left
        to_visit = deque()
        to_visit.extend(n_lucky_urls)

        # Navigate to the target URL while staying logged in
        self.page.wait_for_load_state('networkidle')  # Wait for the page to be fully loaded
        structured_content = extract_content(self.page, output_type="formatted_text")
        prompt = self.jinja_env.get_template('classification_extraction_prompt.j2').render(
            cleaned_html=structured_content
        )
        try: 
            response = classify_and_extract_news_article(prompt)
            print(response)
            responses.append((self.website_url, response))
        except Exception as e: 
            print(f"Warning: Could not classify and extract news article. Full error:\n{traceback.format_exc()}")
        
        if not self.crawl: 
            self.add_visited_urls([(self.website_url, response.classification if response else None)])
            return responses
        
        newly_visited_urls = [(self.website_url, response.classification if response else None)]

        while to_visit and len(responses) < self.max_pages:
            current_url = to_visit.popleft()
            print(f"crawling url {current_url}")
            if current_url in self.visited_urls:
                continue

            self.page.goto(current_url)

            try:
                self.page.wait_for_load_state('networkidle')  # Wait for the page to be fully loaded

                html_content = extract_content(self.page, output_type="formatted_text")
                prompt = self.jinja_env.get_template('classification_extraction_prompt.j2').render(
                    cleaned_html=html_content
                )
                response = classify_and_extract_news_article(prompt)
                print(response)
                if response: 
                    responses.append((current_url, response))
                newly_visited_urls.append((current_url, response.classification if response else None))
            except Exception as e: 
                print(f"Warning: Could not classify and extract news article from {current_url}. Skipping... Full error:\n{traceback.format_exc()}")
            
            self.page.wait_for_load_state('networkidle')  # Wait for the page to be fully loaded

            new_links = extract_content(self.page, output_type="links")
            print(f"found {len(new_links)} new links")
            prompt = self.jinja_env.get_template('articles_links_prompt.j2').render(
                urls=new_links
            )
            try: 
                response = select_likely_URLS(prompt)
                print(response)
                new_lucky_urls = response.likely_urls

                # Filter new_lucky_urls to only include URLs that exist in 'new_links'
                if isinstance(new_lucky_urls, list):
                    n_lucky_urls = [url for url in new_lucky_urls if url in new_links]
                else:
                    n_lucky_urls = [new_lucky_urls] if new_lucky_urls in new_links else []

                print(f"found {len(n_lucky_urls)} new urls")
                print(f"llm hallucinated {len(new_lucky_urls) - len(n_lucky_urls)} urls")

                to_visit.extend(n_lucky_urls)
            except Exception as e: 
                print(f"Warning: Could not select likely URLs from {current_url}. Full error:\n{traceback.format_exc()}")

        self.add_visited_urls(newly_visited_urls)
        return responses
    
    def close(self):
        # Close the browser and playwright
        self.browser.close()
        self.playwright.stop()

    def get_visited_urls(self):
        """
        Retrieve all visited URLs for the current domain
        Returns:
            set: Set of tuples containing (url, is_article)
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT url, is_article FROM visited_urls WHERE domain = ?', (self.domain,))
        urls = {(row[0], row[1]) for row in c.fetchall()}
        conn.close()
        return urls

    def add_visited_urls(self, url_tuples):
        """
        Add multiple URLs to the visited database
        Args:
            url_tuples: List of tuples, each containing (url, is_article)
        """
        if not url_tuples:
            return
            
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Prepare data for batch insert
        now = datetime.now()
        data = [(url, self.domain, now, is_article) for url, is_article in url_tuples]
        
        try:
            c.executemany('''INSERT OR IGNORE INTO visited_urls 
                            (url, domain, visit_date, is_article) 
                            VALUES (?, ?, ?, ?)''', data)
            conn.commit()
            
            # Update the in-memory set
            self.visited_urls.update(set(url_tuples))
            print("added visited urls to databse successfully")
        except sqlite3.Error as e:
            print(f"Database error: {e}")
        finally:
            conn.close()

    def _initialize_database(self):
        """Initialize the SQLite database and create necessary tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        try:
            # Check if tables exist
            c.execute("""SELECT name FROM sqlite_master 
                        WHERE type='table' 
                        AND name IN ('visited_urls', 'websites')""")
            existing_tables = {row[0] for row in c.fetchall()}
            
            # Only create tables that don't exist
            if 'visited_urls' not in existing_tables:
                c.execute('''CREATE TABLE visited_urls
                            (url TEXT PRIMARY KEY,
                             domain TEXT,
                             visit_date TIMESTAMP,
                             is_article BOOLEAN)''')
            
            if 'websites' not in existing_tables:
                c.execute('''CREATE TABLE websites
                            (url TEXT PRIMARY KEY,
                             login_url TEXT,
                             username TEXT,
                             password TEXT,
                             username_selector TEXT,
                             password_selector TEXT,
                             submit_button_selector TEXT)''')
            
            conn.commit()
        except sqlite3.Error as e:
            print(f"Database initialization error: {e}")
        finally:
            conn.close()

    def add_website(self):
        """Add or update a website's credentials and selectors in the database."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''INSERT OR REPLACE INTO websites 
                    (url, login_url, username, password, 
                        username_selector, password_selector, submit_button_selector)
                    VALUES (?, ?, ?, ?, ?, ?, ?)''',
                    (self.website_url, self.login_url, self.username, self.password,
                    self.username_selector, self.password_selector, self.submit_button_selector))
        conn.commit()
        conn.close()

    def _extract_domain(self, url):
        """Extract domain from URL."""
        from urllib.parse import urlparse
        return urlparse(url).netloc

def main():
    # Test website configuration
    test_config = {
        'website_url': 'https://www.jeuneafrique.com',
        'username': 'Anas.abdoun@gmail.com',
        'password': 'Kolxw007',
        'crawl': True,
        'max_pages': 3  # Limit for testing
    }
    
    # Initialize scraper
    scraper = NewsScrapper(**test_config)
    
    print(f"Starting scrape of {test_config['website_url']}...")
    
    # Perform scraping
    articles = scraper.scrape()
    
    # Print results
    print(f"\nFound {len(articles)} articles:")
    for i, (url, article) in enumerate(articles, 1):
        print(f"\nArticle {i}:")
        print(f"URL: {url}")
        print(f"Classification: {article.classification}")
        print(f"Title: {article.title[:100]}..." if hasattr(article, 'title') else "No title")
        print(f"Content length: {len(article.body)} characters" if hasattr(article, 'body') else "No article body")
        


if __name__ == "__main__":
    main()

