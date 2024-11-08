from bs4 import BeautifulSoup, Comment
import re
from urllib.parse import urlparse, urljoin

def extract_content(page, output_type="full_html"):
    """
    Extract different types of content from the page based on specified types.
    
    """

    page.wait_for_selector('a[href]')
    # Extract links if requested
    if output_type == 'links':
        # Wait for the page to fully load (ensure all resources are loaded)
        page.wait_for_load_state('networkidle')
        base_url = page.url

        parsed_base_url = urlparse(base_url)
        base_netloc = parsed_base_url.netloc.lower()
        # Remove 'www.' if present
        if base_netloc.startswith('www.'):
            base_netloc = base_netloc[4:]

        # Evaluate JavaScript to get all links
        links = page.evaluate('''() => {
            const anchors = Array.from(document.querySelectorAll('a[href]'));
            return anchors.map(anchor => ({
                text: anchor.textContent.trim(),
                href: anchor.getAttribute('href')
            }));
        }''')
        print(f'extracted {len(links)} links ')

        # Filter internal links
        internal_links = []
        for link in links:
            # Ensure href is not None or empty
            if not link['href']:
                continue

            absolute_href = urljoin(base_url, link['href'])
            parsed_url = urlparse(absolute_href)
            link_scheme = parsed_url.scheme.lower()
            link_netloc = parsed_url.netloc.lower()

            # Remove 'www.' if present
            if link_netloc.startswith('www.'):
                link_netloc = link_netloc[4:]
            
            # Filter only http and https links
            if link_scheme in ['http', 'https']:
                if link_netloc == base_netloc:
                    link['href'] = absolute_href  # Update href to the absolute URL
                    internal_links.append(link)

        print(f'Found {len(internal_links)} internal links.')
        return internal_links
    # Extract full HTML if requested
    if output_type == 'full_html':
        return page.content()

    # Extract formatted text if requested
    if output_type == 'formatted_text':
        return page.evaluate('''() => {
            function processNode(node, result = '') {
                // Handle different node types
                switch(node.nodeName) {
                    case 'H1': case 'H2': case 'H3': case 'H4': case 'H5': case 'H6':
                        const level = node.nodeName.charAt(1);
                        return '\\n\\n' + '#'.repeat(level) + ' ' + node.textContent.trim() + '\\n';
                    case 'P':
                    case 'DIV':
                    case 'SECTION':
                    case 'ARTICLE':
                        return '\\n' + node.textContent.trim() + '\\n';
                    case 'LI':
                        return '\\n• ' + node.textContent.trim();
                    case 'TABLE':
                        return '\\n[Table content]\\n';
                    case 'BR':
                        return '\\n';
                    default:
                        if (node.nodeType === Node.TEXT_NODE) {
                            const text = node.textContent.trim();
                            return text ? text + ' ' : '';
                        }
                        return '';
                }
            }
            
            function traverseNode(node) {
                let result = '';
                if (node.style && node.style.display === 'none') return '';
                
                result += processNode(node);
                for (const child of node.childNodes) {
                    result += traverseNode(child);
                }
                return result;
            }
            
            return traverseNode(document.body)
                .replace(/\\n\\s*\\n/g, '\\n\\n')  // Remove extra newlines
                .trim();
        }''')

    # Extract structured content if requested
    if output_type == 'structured':
        return page.evaluate('''() => {
            function extractStructured(element) {
                const result = {
                    tag: element.tagName.toLowerCase(),
                    type: element.nodeType,
                    text: element.textContent.trim()
                };
                
                // Add specific attributes based on tag
                if (element.tagName === 'A') {
                    result.href = element.href;
                }
                if (element.tagName === 'IMG') {
                    result.src = element.src;
                    result.alt = element.alt;
                }
                
                // Extract classes and IDs
                if (element.className) {
                    result.classes = element.className.split(' ');
                }
                if (element.id) {
                    result.id = element.id;
                }
                
                // Extract children
                const children = Array.from(element.children);
                if (children.length > 0) {
                    result.children = children.map(child => extractStructured(child));
                }
                
                return result;
            }
            
            // Start from main content area or body
            const mainContent = document.querySelector('main') || document.body;
            return extractStructured(mainContent);
        }''')
    
def clean_html_for_login_detection(html):
    """
    Clean HTML content specifically for login form detection by LLMs.
    Focuses on preserving form elements, inputs, and their surrounding context
    while removing unnecessary content to reduce token usage.
    
    Args:
        html (str): Raw HTML content from Playwright
        
    Returns:
        str: Cleaned HTML string optimized for login form detection
    """
    soup = BeautifulSoup(html, 'html.parser')
    
    # 1. Remove definitely unnecessary elements
    for tag in soup(['script', 'style', 'noscript', 'svg', 'img', 'picture', 
                    'video', 'audio', 'canvas', 'map', 'track', 'head']):
        tag.decompose()
    
    # 2. Remove all comments
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()
    
    # 3. Keep only relevant form-related elements and their containers
    relevant_tags = {
        'form', 'input', 'button', 'div', 'label', 'span', 'a', 
        'p', 'h1', 'h2', 'h3', 'h4', 'section', 'main'
    }
    
    for tag in soup.find_all(True):  # True matches all tags
        if tag.name not in relevant_tags:
            tag.unwrap()  # Keep content but remove the tag
            
    # 4. Preserve only useful attributes for form detection
    important_attrs = {
        'type', 'name', 'id', 'class', 'placeholder', 'value', 
        'aria-label', 'role', 'for'
    }
    
    for tag in soup.find_all(True):
        # Keep only important attributes
        attrs = dict(tag.attrs)
        for attr in attrs:
            if attr not in important_attrs:
                del tag.attrs[attr]
                
        # Special handling for input fields
        if tag.name == 'input':
            # Keep type, especially for password fields
            if 'type' not in tag.attrs:
                tag['type'] = 'text'
                
    # 5. Remove empty containers that don't contribute to form structure
    for tag in soup.find_all(['div', 'section', 'main']):
        if not tag.find(['form', 'input', 'button', 'label']) and not tag.get_text(strip=True):
            tag.decompose()
            
    # 6. Clean up text content
    for tag in soup.find_all(string=True):
        if tag.parent.name not in ['button', 'label', 'input']:
            # Truncate long text that's not in important elements
            text = tag.strip()
            if len(text) > 50:  # Arbitrary length limit
                tag.replace_with(' '.join(text.split()[:7]) + '...')
                
    # 7. Remove duplicate forms if they exist (keep only unique ones)
    seen_forms = set()
    for form in soup.find_all('form'):
        form_signature = str(form.find_all(['input', 'button']))
        if form_signature in seen_forms:
            form.decompose()
        else:
            seen_forms.add(form_signature)
            
    # 8. Ensure proper nesting and structure
    cleaned_html = str(soup)
    
    # 9. Final cleanup: remove excessive whitespace and newlines
    cleaned_html = re.sub(r'\s+', ' ', cleaned_html)
    cleaned_html = re.sub(r'>\s+<', '><', cleaned_html)
    
    return cleaned_html

