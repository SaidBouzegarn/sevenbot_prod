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
        base_netloc = parsed_base_url.netloc.lower().replace('www.', '')

        # Enhanced link extraction to catch more internal links
        links = page.evaluate('''() => {
            function getAllLinks() {
                // Get all elements that could potentially be or contain links
                const elements = document.querySelectorAll('*');
                const links = new Set();
                
                elements.forEach(element => {
                    // Check regular href attributes
                    if (element.href) {
                        links.add({
                            text: element.textContent.trim(),
                            href: element.href
                        });
                    }
                    
                    // Check data attributes that might contain URLs
                    for (const attr of element.attributes) {
                        if (attr.name.includes('href') || attr.name.includes('link') || 
                            attr.name.includes('url') || attr.value.startsWith('http')) {
                            links.add({
                                text: element.textContent.trim(),
                                href: attr.value
                            });
                        }
                    }
                });
                
                return Array.from(links);
            }
            return getAllLinks();
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
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove all scripts, styles, and comments
    for element in soup(['script', 'style', 'noscript', 'iframe', 'meta', 'link', 'head']):
        element.decompose()
    for element in soup(string=lambda text: isinstance(text, Comment)):
        element.extract()
    
    # Remove elements that typically contain non-essential content
    for element in soup.find_all(['header', 'footer', 'nav', 'aside', 'menu']):
        element.decompose()
    
    # Process remaining elements
    for element in soup.find_all(True):
        # Get text content
        text = element.get_text().strip()
        if text:
            # Truncate text to maximum 7 words
            words = text.split()
            if len(words) > 7:
                truncated_text = ' '.join(words[:7]) + '...'
                # Replace the text content while preserving the element
                for text_node in element.find_all(string=True, recursive=False):
                    if text_node.strip():
                        text_node.replace_with(truncated_text)
                        break
    
    # Final cleanup
    cleaned_html = str(soup)
    #cleaned_html = re.sub(r'\s+', ' ', cleaned_html)  # Normalize whitespace
    #cleaned_html = re.sub(r'>\s+<', '><', cleaned_html)  # Remove whitespace between tags
    
    return cleaned_html

