"""
Serper web search + scraping pipeline.
Search → retrieve URLs → scrape content → local summarization.
"""

import asyncio
import logging
import os
import re
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger("compressed.search")

SERPER_API_URL = "https://google.serper.dev/search"
MAX_RESULTS = 5
MAX_SCRAPE_CHARS = 4000
SCRAPE_TIMEOUT = 8.0


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


@dataclass
class ScrapedPage:
    url: str
    title: str
    text: str  # Cleaned/truncated body text


@dataclass
class SearchBundle:
    """Complete search output ready for model consumption."""
    query: str
    results: List[SearchResult] = field(default_factory=list)
    scraped: List[ScrapedPage] = field(default_factory=list)
    summaries: List[str] = field(default_factory=list)
    error: Optional[str] = None


class SerperSearch:
    """Serper API search client."""

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.getenv("SERPER_API_KEY", "")

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    async def search(self, query: str, num_results: int = MAX_RESULTS) -> List[SearchResult]:
        if not self._api_key:
            logger.warning("SERPER_API_KEY not set — skipping web search")
            return []

        headers = {"X-API-KEY": self._api_key, "Content-Type": "application/json"}
        payload = {"q": query, "num": num_results}

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(SERPER_API_URL, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        results = []
        for item in data.get("organic", [])[:num_results]:
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
            ))
        return results


class WebScraper:
    """Lightweight async web scraper — extracts body text from HTML."""

    # Domains we should not scrape (paywalls, auth-gated, etc.)
    _SKIP_DOMAINS = {"twitter.com", "x.com", "facebook.com", "instagram.com", "linkedin.com"}

    async def scrape(self, urls: List[str], max_chars: int = MAX_SCRAPE_CHARS) -> List[ScrapedPage]:
        tasks = [self._fetch_one(url, max_chars) for url in urls if self._should_scrape(url)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if isinstance(r, ScrapedPage)]

    def _should_scrape(self, url: str) -> bool:
        try:
            parsed = urlparse(url)
            return parsed.hostname not in self._SKIP_DOMAINS if parsed.hostname else False
        except Exception:
            return False

    async def _fetch_one(self, url: str, max_chars: int) -> ScrapedPage:
        async with httpx.AsyncClient(
            timeout=SCRAPE_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": "SentinelE-Bot/1.0 (research)"},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove scripts, styles, nav elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        # Extract title
        title = soup.title.string.strip() if soup.title and soup.title.string else ""

        # Extract main text
        text = soup.get_text(separator=" ", strip=True)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text)
        # Truncate
        text = text[:max_chars]

        return ScrapedPage(url=url, title=title, text=text)


def build_search_context(bundle: SearchBundle) -> str:
    """Format search results into a compressed context block for LLM consumption."""
    if not bundle.summaries and not bundle.results:
        return ""

    parts = [f"Web search for: '{bundle.query}'"]

    # Use summaries if available, else fall back to snippets
    if bundle.summaries:
        for i, summary in enumerate(bundle.summaries, 1):
            parts.append(f"[Source {i}]: {summary}")
    else:
        for i, r in enumerate(bundle.results, 1):
            parts.append(f"[{i}] {r.title}: {r.snippet}")

    return "\n".join(parts)
