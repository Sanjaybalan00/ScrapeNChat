import requests
from bs4 import BeautifulSoup

def scrape_wikipedia(url: str):
    """
    Extracts the main content from a Wikipedia page, including paragraphs and headings.
    """
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to retrieve the page. Status code: {response.status_code}")
    
    soup = BeautifulSoup(response.text, 'html.parser')
    content = soup.find(id="mw-content-text")
    
    # Initialize a list to store text from paragraphs and headings
    elements = []
    
    # Extract text from paragraphs and headings (h1 to h6 tags)
    for tag in content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        text = tag.get_text().strip()
        if text:
            elements.append(text)
    
    return elements

# Example usage
