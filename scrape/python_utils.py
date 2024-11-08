import re
from bs4 import BeautifulSoup, Comment
import requests
import httpx
import urllib3
import logging
from urllib.parse import urljoin, urlparse
import validators

logger = logging.getLogger(__name__)


def trim_text_to_four_words(text):
    """Trim the text to the first four words if it is longer."""
    words = text.split()
    text = " ".join(words[:3]) if len(words) > 4 else text
    words = text.split("-")
    text = "-".join(words[:3]) if len(words) > 4 else text
    return text


def clean_html_for_login(html):
    """Clean the HTML content."""

    soup = BeautifulSoup(html, "html.parser")
    black_listed_elements = {
        "script",
        "style",
        "meta",
        "iframe",
        "path",
        "svg",
        "noscript",
        "link",
    }
    for tag in soup(black_listed_elements):
        tag.extract()
    for element in soup.descendants:
        if isinstance(element, str):
            trimmed_text = trim_text_to_four_words(element)
            element.replace_with(re.sub(r"\s+", " ", trimmed_text))
    logger.info("HTML content cleaned.")
    return soup


def clean_html_for_credentials_id(html):
    """Clean the HTML content."""
    soup = BeautifulSoup(html, "html.parser")
    black_listed_elements = {
        "script",
        "style",
        "meta",
        "iframe",
        "path",
        "svg",
        "noscript",
        "link",
    }
    for tag in soup(black_listed_elements):
        tag.extract()
    for element in soup.descendants:
        if isinstance(element, str):
            trimmed_text = trim_text_to_four_words(element)
            element.replace_with(re.sub(r"\s+", " ", trimmed_text))
    logger.info("HTML content cleaned.")
    return str(soup)


def truncate_text_based_on_word_count(text):
    """
    Truncate the text based on specified rules:
    - If text < 3 words, do nothing.
    - If text < 15 words, truncate to 5 words.
    - If text < 70 words, truncate to 20 words.
    - If text < 300 words, truncate to 90 words.
    - If text < 800 words, truncate to 250 words.
    - If text > 800 words, truncate to 1000 words.
    """
    words = text.split()
    word_count = len(words)

    if word_count < 3:
        return text
    elif word_count < 15:
        return " ".join(words[:5]) + "..."
    elif word_count < 70:
        return " ".join(words[:20]) + "..."
    elif word_count < 300:
        return " ".join(words[:90]) + "..."
    elif word_count < 800:
        return " ".join(words[:250]) + "..."
    else:
        return " ".join(words[:1000]) + "..."


def clean_html_for_classification_V0(html):
    """
    Clean HTML content to help an LLM classify the webpage.
    Retains essential structure and content while removing noise and unnecessary elements.
    Applies text truncation based on length rules.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove scripts, styles, and comments as they are not relevant for classification
    for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
        tag.decompose()

    for comment in soup.findAll(text=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Retain essential tags for structure and content
    essential_tags = {
        "h1",
        "h2",
        "h3",
        "h4",
        "p",
        "div",
        "section",
        "article",
        "span",
        "ul",
        "ol",
        "li",
        "header",
        "footer",
        "main",
        "time",
    }
    for tag in soup.find_all(True):  # True matches all tags
        if tag.name not in essential_tags:
            tag.unwrap()  # Keep content but remove the tag itself

    # Apply text truncation based on the rules provided
    for tag in soup.find_all(string=True):
        if tag.parent.name in essential_tags:
            truncated_text = truncate_text_based_on_word_count(tag)
            tag.replace_with(truncated_text)

    # Minimize attributes that aren't necessary for classification
    for tag in soup.find_all(True):
        for attribute in list(tag.attrs):
            if attribute not in ["id", "class", "href", "src", "alt"]:
                del tag.attrs[attribute]

    # Return the cleaned and truncated HTML as a string
    cleaned_html = str(soup)[1000:6000]
    logger.info("HTML content cleaned.")

    return cleaned_html

def clean_html_for_classification_V1(html):
    """
    Clean HTML content to help an LLM classify the webpage.
    Retains essential structure and content while removing noise, repeated sections, and unnecessary elements.
    Applies text truncation based on length rules.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove scripts, styles, and comments as they are not relevant for classification
    for tag in soup(["script", "style", "noscript", "iframe", "svg", "form", "input", "button"]):
        tag.decompose()

    for comment in soup.findAll(text=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Retain essential tags for structure and content
    essential_tags = {
        "h1", "h2", "h3", "h4", "p", "div", "section", "article", "span",
        "ul", "ol", "li", "header", "footer", "main", "time", "a", "img"
    }

    # Further clean up redundant tags and attributes
    for tag in soup.find_all(True):  # True matches all tags
        if tag.name not in essential_tags:
            tag.unwrap()  # Keep content but remove the tag itself
        else:
            # Further filter 'a' tags to reduce redundancy
            if tag.name == "a" and len(tag.get_text(strip=True)) < 5:
                tag.unwrap()

    # Remove duplicate sections based on simple text or structure patterns
    seen_texts = set()
    for tag in soup.find_all(essential_tags):
        text = tag.get_text(strip=True)
        if text in seen_texts:
            tag.decompose()
        else:
            seen_texts.add(text)

    # Apply text truncation based on the rules provided
    for tag in soup.find_all(string=True):
        if tag.parent.name in essential_tags:
            truncated_text = truncate_text_based_on_word_count(tag)
            tag.replace_with(truncated_text)

    # Minimize attributes that aren't necessary for classification
    for tag in soup.find_all(True):
        for attribute in list(tag.attrs):
            if attribute not in ["id", "class", "href", "src", "alt"]:
                del tag.attrs[attribute]

    # Return the cleaned and truncated HTML as a string
    cleaned_html = str(soup)[1000:6000]
    logger.info("HTML content cleaned.")

    return cleaned_html

def clean_html_for_classification(html_content):
    # Parse HTML
    soup = BeautifulSoup(html_content, 'html.parser')

    # 1. Remove non-essential tags (script, style, meta, link, header, footer)
    for tag in soup(['script', 'style', 'meta', 'link', 'header', 'footer', 'nav', 'aside']):
        tag.decompose()

    # 2. Simplify or remove inline attributes, keeping only 'href' and 'alt'
    for tag in soup.find_all(True):
        attrs = dict(tag.attrs)
        for attr in attrs:
            if attr not in ['href', 'alt']:
                del tag.attrs[attr]

    # 3. Flatten nested structures (custom logic can be applied here based on specific needs)
    def flatten_nested_tags(soup):
        # Example: Flatten multiple nested <div> tags
        for div in soup.find_all('div'):
            if div.find('div'):
                div.unwrap()  # This removes the current <div> but keeps its content

    flatten_nested_tags(soup)

    # 4. Map non-semantic tags to semantic tags
    tag_mapping = {
        'b': 'strong',
        'i': 'em',
        'u': 'ins',
        'div': 'section',  # example mapping; adjust as needed
        'span': 'p'  # example mapping; adjust as needed
    }

    for old_tag, new_tag in tag_mapping.items():
        for tag in soup.find_all(old_tag):
            tag.name = new_tag

    # 5. Shorten text content judiciously (simplified example)
    def shorten_long_text(soup, max_length=100):
        for tag in soup.find_all(text=True):
            if len(tag) > max_length:
                tag.replace_with(tag[:max_length] + '...')

    shorten_long_text(soup)

    # 6. Remove non-primary content (e.g., sidebars, ads)
    for tag in soup(['sidebar', 'ad', 'footer', 'header', 'nav']):
        tag.decompose()

    # 7. Condense whitespace and newlines
    cleaned_html = ' '.join(soup.prettify().split())

    # 8. Extract and condense lists (optional logic for long lists)
    for ul in soup.find_all('ul'):
        items = ul.find_all('li')
        if len(items) > 5:  # Example condition for long lists
            for extra_item in items[5:]:
                extra_item.decompose()

    # 9. Further processing for token count if needed (tokenization can be done here)
    # Example: Splitting content based on token count can be added here

    return cleaned_html




def clean_html_for_extraction(html):
    """
    Clean HTML content to help an LLM efficiently extract article details like date, author, and body.
    Retains all potentially useful textual content while minimizing noise and token usage.
    """
    soup = BeautifulSoup(html, "html.parser")

    # 1. Remove clearly non-essential tags (scripts, styles, iframes, ads, etc.)
    for tag in soup(["script", "style", "iframe", "noscript", "svg", "link", "meta", "form", "input", "button", "aside", "advertisement", "footer", "header", "nav", "figure"]):
        tag.decompose()

    # 2. Remove comments
    for comment in soup.findAll(text=lambda text: isinstance(text, Comment)):
        comment.extract()

    # 3. Simplify the HTML by reducing attributes
    allowed_attributes = ["id", "class", "datetime", "alt", "href", "src"]
    for tag in soup.find_all(True):
        # Only keep essential attributes that might be relevant
        tag.attrs = {key: tag.attrs[key] for key in tag.attrs if key in allowed_attributes}

    # 4. Keep tags that may contain content, unwrap everything else
    # This retains the text and structure from likely useful elements
    essential_tags = {
        "h1", "h2", "h3", "h4", "p", "div", "section", "article", "span", "time", "ul", "ol", "li", "blockquote", "strong", "em", "a", "table", "tr", "td", "th"
    }
    for tag in soup.find_all(True):  # True matches all tags
        if tag.name not in essential_tags:
            tag.unwrap()  # Keep the content but remove the tag itself

    # 5. Remove any remaining empty tags to reduce noise
    for tag in soup.find_all():
        if not tag.get_text(strip=True):
            tag.decompose()

    # 6. Normalize whitespace and reduce excessive newlines
    cleaned_html = ' '.join(soup.get_text().split())

    return cleaned_html



def clean_html_for_extraction_V0(html):
    """
    Clean HTML content to help an LLM efficiently extract article details like date, author, and body.
    Retains full textual content while minimizing noise and token usage.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove scripts, styles, iframes, and comments
    for tag in soup(["script", "style", "iframe", "noscript", "svg", "link", "meta"]):
        tag.decompose()

    for comment in soup.findAll(text=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Retain only the most essential tags that are likely to contain article content
    essential_tags = {
        "h1",
        "h2",
        "h3",
        "h4",
        "p",
        "div",
        "section",
        "article",
        "span",
        "time",
    }
    for tag in soup.find_all(True):  # True matches all tags
        if tag.name not in essential_tags:
            tag.unwrap()  # Keep the content but remove the tag itself

    # Simplify the HTML by reducing attributes
    for tag in soup.find_all(True):
        # Only keep essential attributes for extraction
        for attribute in list(tag.attrs):
            if attribute not in ["id", "class", "datetime", "alt"]:
                del tag.attrs[attribute]

    # Optionally truncate long sections of text to save tokens
    for tag in soup.find_all(["p", "div", "section", "article"]):
        if len(tag.get_text(strip=True)) > 500:
            truncated_text = tag.get_text(strip=True)[:500] + "..."
            tag.string = (
                truncated_text  # Replace the tag's content with the truncated text
            )

    # Return the cleaned HTML as a string
    cleaned_html = str(soup)
    logger.info("HTML content cleaned.")

    return cleaned_html


def split_html_into_chunks(html: str, min_tokens: int = 1000, margin_tokens: int = 100):
    """
    Splits an HTML string into chunks with a minimum token count.
    Chunks are split around <div> tags with a margin to preserve context.

    Args:
        html (str): The HTML content to be split.
        min_tokens (int): The minimum number of tokens per chunk.
        margin_tokens (int): The number of tokens to include before and after a <div> tag for intercalation.

    Returns:
        List[str]: A list of HTML chunks.
    """

    def add_element_to_chunk(
        element_html: str, current_chunk: str, current_chunk_size: int
    ) -> (str, int):
        """
        Adds an HTML element to the current chunk and updates the chunk size.

        Args:
            element_html (str): The HTML string of the element to add.
            current_chunk (str): The current chunk being built.
            current_chunk_size (int): The size of the current chunk in tokens.

        Returns:
            Tuple[str, int]: Updated current chunk and its size.
        """
        current_chunk += element_html
        current_chunk_size += len(element_html.split())
        return current_chunk, current_chunk_size

    def split_at_nearest_div(current_chunk: str) -> (str, str):
        """
        Splits the current chunk at the nearest <div> tag with a margin.

        Args:
            current_chunk (str): The chunk to be split.

        Returns:
            Tuple[str, str]: The first part of the chunk up to the split point, and the remaining part.
        """
        nearest_div_index = current_chunk.find("<div")
        if nearest_div_index != -1:
            start_cut = max(0, nearest_div_index - margin_tokens)
            end_cut = min(len(current_chunk), nearest_div_index + margin_tokens)
            return current_chunk[:end_cut], current_chunk[end_cut:]
        else:
            return current_chunk, ""  # If no <div> is found, return the whole chunk

    soup = BeautifulSoup(html, "html.parser")
    chunks = []
    current_chunk = ""
    current_chunk_size = 0

    # Check if the body exists
    if soup.body is None:
        # Handle the case where there's no body tag
        logger.warning("Warning: No <body> tag found in the HTML.")
        return []

    for element in soup.body.children:
        element_html = str(element)
        current_chunk, current_chunk_size = add_element_to_chunk(
            element_html, current_chunk, current_chunk_size
        )

        if current_chunk_size >= min_tokens:
            first_part, remaining_part = split_at_nearest_div(current_chunk)
            chunks.append(first_part)
            current_chunk = remaining_part
            current_chunk_size = len(current_chunk.split())

    # Add any remaining content as the last chunk
    if current_chunk:
        chunks.append(current_chunk)

    return chunks

from urllib.parse import urlparse, urljoin, urldefrag

from urllib.parse import urlparse, urljoin, urldefrag

def normalize_url(base_url, href):
    """
    Normalize URLs by resolving relative URLs and removing fragments.
    """
    # Resolve relative URLs
    joined_url = urljoin(base_url, href)
    # Remove URL fragment (anything after the #)
    defragged_url = urldefrag(joined_url)[0]
    # Return the cleaned URL
    return defragged_url


def filter_and_normalize_links(base_url, links, visited_urls, visited_check=True):
    """
    Filter out unnecessary URLs, normalize them, and log the process.
    """
    valid_links = set()
    allowed_domain = urlparse(
        base_url
    ).netloc  # Extract domain from base URL for consistency

    if visited_check == False:
        visited_urls = set()

    for link in links:
        href = link.get("href")
        if href:
            new_url = normalize_url(base_url, href)
            if new_url:
                url_parts = urlparse(new_url)

                # Ensure the URL is part of the target domain and not visited
                if url_parts.netloc == allowed_domain and new_url not in visited_urls:
                    if new_url.startswith(base_url):
                        valid_links.add(new_url)
                        #logger.info(f"Added valid URL: {new_url}")
                    else:
                        #logger.debug(
                        #    f"Rejected URL not starting with base URL: {new_url}"
                        #)
                        pass
                else:
                    #logger.debug(
                    #    f"Rejected URL outside domain or already visited: {new_url}"
                    #)
                    pass
            else:
                #logger.debug(f"Failed to normalize or rejected URL: {href}")
                pass
        else:
            logger.debug("Missing href in link object")
            pass
    return valid_links


def extract_links(website, html):
    """Extract all links from the HTML and filter them."""
    links = re.findall(r'href="(https?://[^"]+)"', html)
    filtered_links = [link for link in links if link.count("/") <= 4]
    filtered_links = list(filter_and_normalize_links(website, filtered_links, set(), visited_check=False ))
    logger.debug(
        f"Extracted {len(links)} links, {len(filtered_links)} after filtering."
    )
    return filtered_links


def validate(website: str, url: str) -> str:
    """
    Validates an existing login URL or attempts to retrieve it if invalid/missing.

    Args:
        website (str): The website to process.
        url (str): The current login URL.

    Returns:
        str: A valid login URL or an empty string if not retrievable.
    """
    if url and validators.url(url):
        logger.info(f"Valid URL found for {website}: {url}")
        return url
    else:
        logger.debug(f"Invalid or missing login URL for {website}: {url}. ")
        return False
