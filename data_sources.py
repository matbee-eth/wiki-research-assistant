# data_sources.py

import os
import json
import aiohttp
import asyncio
from typing import Dict, List, Any, Optional, Union
from txtai.embeddings import Embeddings
import logging
from typing import AsyncGenerator

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class DataSources:
    def __init__(self):
        """Initialize data sources."""
        self.session = None
        self.embeddings = None
        self._cache = {}
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
        
    async def search_wikipedia(self, query: str, min_score: float = 0.8) -> List[Dict[str, Any]]:
        """
        Search Wikipedia articles using semantic search.
        
        Args:
            query: Search query
            min_score: Minimum similarity score threshold (0-1)
            
        Returns:
            List of search results with metadata above the score threshold
        """
        logger.info(f"Searching Wikipedia for query: '{query}' with min_score {min_score}")
        try:
            # Perform semantic search with a high initial limit to get enough results for filtering
            logger.debug("Performing txtai semantic search...")
            raw_results = self.embeddings.search(str(query), limit=1000)  # Get more results initially for better filtering
            
            # Process and filter results
            results = []
            for r in raw_results:
                if isinstance(r, (list, tuple)):
                    score = r[0]
                else:
                    score = r.get('score', 0.0)
                
                if score >= min_score:
                    results.append(r)
            
            if not results:
                logger.warning(f"No results found for query: {query}")
                return []
                
            logger.info(f"Found {len(results)} results above score threshold {min_score}")
            
            # Process results
            processed_results = []
            for result in results:  # Process all results that meet score threshold
                try:
                    # Handle both tuple and dict result formats
                    if isinstance(result, (list, tuple)):
                        score, text, article_id = result
                    else:
                        score = result.get('score', 0.0)
                        text = result.get('text', '')
                        article_id = result.get('id', '')
                    
                    # Skip invalid results
                    if not text or not article_id:
                        continue
                        
                    # Get article metadata
                    metadata = await self.get_wikipedia_page(article_id)
                    if not metadata:
                        continue
                        
                    # Create result data
                    result_data = {
                        **metadata,
                        "score": float(score),
                        "text": text
                    }
                    processed_results.append(result_data)
                    
                except Exception as e:
                    logger.error(f"Error processing result: {str(e)}", exc_info=True)
                    continue
            
            logger.info(f"Successfully processed {len(processed_results)} results")
            return processed_results
            
        except Exception as e:
            logger.error(f"Error during Wikipedia search: {str(e)}", exc_info=True)
            return []
            
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
            
        if page_identifier in self._cache:
            return self._cache[page_identifier]
            
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
                
                # Cache the result
                self._cache[page_identifier] = metadata
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

    async def stream_search_wikipedia(self, query: str, min_score: float = 0.8) -> AsyncGenerator[List[Dict[str, Any]], None]:
        """
        Stream Wikipedia search results as they are processed.
        
        Args:
            query: Search query
            min_score: Minimum similarity score threshold (0-1)
            
        Yields:
            Batches of search results with metadata above the score threshold
        """
        logger.info(f"Streaming Wikipedia search for query: '{query}' with min_score {min_score}")
        try:
            # Perform semantic search with a high initial limit to get enough results for filtering
            logger.debug("Performing txtai semantic search...")
            raw_results = self.embeddings.search(str(query), limit=1000)  # Get more results initially for better filtering
            
            # Process and filter results
            filtered_results = []
            for r in raw_results:
                if isinstance(r, (list, tuple)):
                    score = r[0]
                else:
                    score = r.get('score', 0.0)
                
                if score >= min_score:
                    filtered_results.append(r)
            
            if not filtered_results:
                logger.warning(f"No results found for query: {query}")
                yield []
                return
                
            logger.info(f"Found {len(filtered_results)} results above score threshold {min_score}")
            
            # Process results in batches
            batch_size = 10
            current_batch = []
            
            for result in filtered_results:
                try:
                    # Handle both tuple and dict result formats
                    if isinstance(result, (list, tuple)):
                        score, text, article_id = result
                    else:
                        score = result.get('score', 0.0)
                        text = result.get('text', '')
                        article_id = result.get('id', '')
                    
                    # Skip invalid results
                    if not text or not article_id:
                        continue
                        
                    # Get article metadata
                    metadata = await self.get_wikipedia_page(article_id)
                    if not metadata:
                        continue
                        
                    # Create result data
                    result_data = {
                        **metadata,
                        "score": float(score),
                        "text": text
                    }
                    current_batch.append(result_data)
                    
                    # Yield batch when it reaches the desired size
                    if len(current_batch) >= batch_size:
                        yield current_batch
                        current_batch = []
                    
                except Exception as e:
                    logger.error(f"Error processing result: {str(e)}", exc_info=True)
                    continue
            
            # Yield any remaining results
            if current_batch:
                yield current_batch
            
            logger.info("Completed streaming search results")
            
        except Exception as e:
            logger.error(f"Error during streaming Wikipedia search: {str(e)}", exc_info=True)
            yield []

    async def _fetch_wikipedia_article(self, article_id: str) -> Dict[str, Any]:
        """Fetch article content from Wikipedia API."""
        try:
            # Construct API URL
            params = {
                'action': 'query',
                'format': 'json',
                'titles': article_id,
                'prop': 'extracts|info',
                'exintro': True,
                'explaintext': True,
                'inprop': 'url'
            }
            
            url = 'https://en.wikipedia.org/w/api.php'
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    pages = data['query']['pages']
                    
                    # Get the first (and only) page
                    page = next(iter(pages.values()))
                    
                    return {
                        'title': page.get('title', ''),
                        'extract': page.get('extract', ''),
                        'url': page.get('fullurl', '')
                    }
                else:
                    logger.error(f"Error fetching article {article_id}: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching article {article_id}: {str(e)}")
            return None
