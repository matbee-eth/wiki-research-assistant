# data_sources.py

import os
import json
import string
import aiohttp
import asyncio
from typing import Dict, List, Any, Optional, Union
from txtai.embeddings import Embeddings
import logging
from typing import AsyncGenerator

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class CacheManager:
    def __init__(self):
        self.cache = {}

    def cache_article(self, article_id: str, article_data: Dict[str, Any]) -> None:
        self.cache[article_id] = article_data

    def get_cached_article(self, article_id: str) -> Optional[Dict[str, Any]]:
        return self.cache.get(article_id)

class DataSources:
    def __init__(self):
        """Initialize data sources."""
        self.embeddings = None
        self.cache_manager = CacheManager()
        self.session = None
        self.logger = logger  # Use the module-level logger

    def _setup_logging(self):
        """Set up logging configuration."""
        pass  # No need to set up logging again, using module-level logger

    async def __aenter__(self):
        """Async context manager entry."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        if not self.embeddings:
            self.embeddings = await self._init_embeddings()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        try:
            if self.session and not self.session.closed:
                logger.debug("Closing data sources session...")
                await self.session.close()
                logger.debug("Session closed")
            
            # Clean up embeddings if needed
            if self.embeddings:
                logger.debug("Cleaning up embeddings...")
                self.embeddings = None
                
        except Exception as e:
            logger.error(f"Error closing data sources session: {str(e)}", exc_info=True)

    async def _init_embeddings(self):
        """Initialize embeddings asynchronously."""
        try:
            logger.debug("Loading pre-trained txtai embeddings...")
            embeddings = Embeddings()
            embeddings.load(provider="huggingface-hub", container="neuml/txtai-wikipedia")
            logger.info("Embeddings loaded successfully")
            return embeddings
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}", exc_info=True)
            raise
        
    async def get_wikipedia_page(self, page_identifier: Union[str, int]) -> Dict[str, Any]:
        """
        Get Wikipedia page metadata by page ID or title.
        
        Args:
            page_identifier: Either a page ID (int) or page title (str)
            
        Returns:
            Dict containing page metadata
        """
        if not page_identifier:
            return {}
            
        try:
            # Base API endpoint
            base_url = "https://en.wikipedia.org/w/api.php"
            
            # Determine if we're using a page ID or title
            if isinstance(page_identifier, int):
                params = {
                    "action": "query",
                    "format": "json",
                    "pageids": str(page_identifier),
                    "prop": "info|extracts|categories",
                    "inprop": "url|displaytitle",
                    "exintro": "1",
                    "explaintext": "1"
                }
            else:
                # Clean the title for API use
                title = page_identifier.replace(" ", "_")
                params = {
                    "action": "query",
                    "format": "json",
                    "titles": title,
                    "prop": "info|extracts|categories",
                    "inprop": "url|displaytitle",
                    "exintro": "1",
                    "explaintext": "1"
                }
            
            async with self.session.get(base_url, params=params) as response:
                if response.status != 200:
                    logger.warning(f"Failed to get Wikipedia page. Status: {response.status}")
                    return {}
                
                data = await response.json()
                
                # Extract page data
                pages = data.get("query", {}).get("pages", {})
                if not pages:
                    logger.warning("No pages found in API response")
                    return {}
                
                # Get the first (and should be only) page
                page = next(iter(pages.values()))
                
                if "missing" in page:
                    logger.warning(f"Page not found: {page_identifier}")
                    return {}
                
                # Extract relevant metadata
                metadata = {
                    "pageid": page.get("pageid"),
                    "title": page.get("title"),
                    "url": page.get("fullurl", f"https://en.wikipedia.org/wiki/{page.get('title', '').replace(' ', '_')}"),
                    "content": page.get("extract"),
                    "last_modified": page.get("touched"),
                    "length": page.get("length")
                }
                
                logger.info(f"Successfully fetched Wikipedia page: {metadata['title']}")
                return metadata
                
        except Exception as e:
            logger.error(f"Error fetching Wikipedia page: {str(e)}", exc_info=True)
            return {}
            
    async def search_wikidata(self, query: str, limit: int = 5):
        # Implement Wikidata search using their SPARQL endpoint or API
        # Placeholder for actual implementation
        logger.info(f"Searching Wikidata for query: '{query}' with limit {limit}")
        return []

    async def get_real_time_news(self, query: str, limit: int = 5):
        logger.info(f"Getting real-time news for query: '{query}' with limit {limit}")
        if not NEWS_API_KEY:
            logger.warning("NEWS_API_KEY not set. Skipping real-time news integration.")
            return []
        url = 'https://newsapi.org/v2/everything'
        params = {
            'q': query,
            'pageSize': limit,
            'apiKey': NEWS_API_KEY,
            'language': 'en'
        }
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
                
            logger.debug(f"Making API request with params: {params}")
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.debug(f"API response for query {query}: {data}")
                    return data.get('articles', [])
                else:
                    logger.error(f"News API error: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching real-time news: {e}", exc_info=True)
            return []

    def initialize(self):
        """Initialize embeddings and other resources."""
        try:
            if self.embeddings is None:
                logger.info("Initializing txtai embeddings")
                from txtai.embeddings import Embeddings
                self.embeddings = Embeddings()
                self.embeddings.load(provider="huggingface-hub", container="neuml/txtai-wikipedia")
                logger.info("Embeddings initialized successfully")
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}", exc_info=True)
            raise

    def _cache_article(self, article_id: str, article_data: Dict[str, Any]) -> None:
        """Cache article data."""
        self.cache_manager.cache_article(article_id, article_data)

    def _get_cached_article(self, article_id: str) -> Optional[Dict[str, Any]]:
        """Get article from cache."""
        return self.cache_manager.get_cached_article(article_id)
    
    async def _fetch_wikipedia_article(self, article_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch article content from Wikipedia.
        
        Args:
            article_id: Wikipedia article ID or title
            
        Returns:
            Dict with article data or None if not found
        """
        try:
            # Initialize session if needed
            if self.session is None:
                import aiohttp
                self.session = aiohttp.ClientSession()
                
            # Convert spaces to underscores and handle URL encoding
            from urllib.parse import quote
            article_title = quote(article_id.replace(' ', '_'))
            
            # Use Wikipedia's REST API
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{article_title}"
            logger.debug(f"Fetching article from: {url}")
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'title': data.get('title'),
                        'url': data.get('content_urls', {}).get('desktop', {}).get('page'),
                        'text': data.get('extract'),
                    }
                else:
                    logger.warning(f"Failed to fetch article {article_id}, status: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching article {article_id}: {str(e)}")
            return None
            
    async def stream_search_wikipedia(self, data: List[string], config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        results = []
        seen_articles = set()
        
        try:
            for query in data:
                if not query:
                    logger.error("No query provided")
                    continue
                    
                search_config = config or {}
                min_score = search_config.get('min_score', 0.7)
                min_percentile = search_config.get('min_percentile', 0.0)
                limit = search_config.get('limit', 2000)
                
                if self.embeddings is None:
                    self.initialize()
                    
                if not self.embeddings:
                    logger.error("Failed to initialize embeddings")
                    return results
                    
                logger.info(f"Searching Wikipedia for query: {query}")
                raw_results = self.embeddings.search(query, limit=limit)
                
                for result in raw_results:
                    score = float(result.get('score', 0.0))
                    if score < min_score:
                        continue
                        
                    article_id = result.get('id', '')
                    
                    # Skip if we've seen this article before
                    if article_id in seen_articles:
                        continue
                    
                    # Check cache first
                    article_data = self._get_cached_article(article_id)
                    if not article_data:
                        article_data = await self._fetch_wikipedia_article(article_id)
                        if article_data:
                            self._cache_article(article_id, article_data)
                    
                    if not article_data:
                        continue
                        
                    search_result = {
                        'article_id': article_id,
                        'title': article_data.get('title', ''),
                        'url': f"https://en.wikipedia.org/wiki/{article_id}",
                        'document': article_data.get('text', ''),
                        'score': score,
                        'query': query
                    }
                    
                    seen_articles.add(article_id)
                    results.append(search_result)

        except Exception as e:
            logger.error(f"Error in stream_search_wikipedia: {str(e)}", exc_info=True)
            raise

        return results