import os

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.tools import tool
from tavily import TavilyClient

load_dotenv()

api_key = os.getenv("TAVILY_API_KEY")
tavily = TavilyClient(api_key=api_key) if api_key else None


@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information on a topic."""
    if not tavily:
        return "Tavily API key is missing. Add TAVILY_API_KEY to your .env file."

    try:
        results = tavily.search(query=query, max_results=5)
        items = results.get("results", [])
        if not items:
            return "No search results were returned for this query."

        output = []
        for item in items:
            title = item.get("title", "Untitled")
            url = item.get("url", "")
            content = item.get("content", "")
            output.append(f"Title: {title}\nURL: {url}\nSnippet: {content[:300]}\n")
        return "\n----\n".join(output)
    except Exception as exc:
        return f"Web search failed: {exc}"


@tool
def scrape_url(url: str) -> str:
    """Scrape and return clean text content from a given URL for deeper reading."""
    if not url:
        return "No URL provided to scrape."

    try:
        response = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return text[:3000] if text else "No readable content was found on the page."
    except Exception as exc:
        return f"Could not scrape URL: {exc}"

