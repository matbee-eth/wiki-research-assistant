# data_sources.py

import aiohttp
import re
import mwparserfromhell
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from txtai.embeddings import Embeddings
import logging

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
            
    async def get_full_wikipedia_page(self, page_identifier: Union[str, int]) -> Dict[str, Any]:
        """
        Get the full content of a Wikipedia page by title or ID.
        
        Args:
            page_identifier: Either the title of the page or its page ID
            
        Returns:
            Dictionary containing:
                - title: Page title
                - pageid: Wikipedia page ID
                - url: URL to the page
                - content: Raw MediaWiki content
                - categories: List of categories
                - timestamp: Last modification time
                - sections: List of sections, each containing:
                    - level: Heading level (1-6)
                    - title: Section title
                    - content: Section content
            
        Raises:
            ValueError: If the page identifier is empty or invalid, or if the page is not found
        """
        if not page_identifier and not isinstance(page_identifier, int):
            raise ValueError("Page identifier cannot be empty")
            
        base_url = "https://en.wikipedia.org/w/api.php"
        
        # Build the query parameters
        params = {
            "action": "query",
            "format": "json",
            "prop": "revisions|categories|info",
            "rvprop": "content|timestamp",
            "rvslots": "main",
            "inprop": "url|displaytitle",
            "formatversion": "2",
            "redirects": "1"
        }
        
        # Add either titles or pageids parameter
        if isinstance(page_identifier, int):
            params["pageids"] = str(page_identifier)
        else:
            params["titles"] = page_identifier
            
        async with self.session.get(base_url, params=params) as response:
            data = await response.json()
            
            if "error" in data:
                raise ValueError(f"API Error: {data['error']['info']}")
                
            if "query" not in data or "pages" not in data["query"]:
                raise ValueError("Invalid API response")
                
            # Get the page data
            pages = data["query"]["pages"]
            if not pages:
                raise ValueError(f"Page '{page_identifier}' not found")
                
            page = pages[0]  # formatversion=2 returns an array
            
            if "missing" in page:
                raise ValueError(f"Page '{page_identifier}' not found")
                
            # Extract the content from revisions
            if "revisions" not in page or not page["revisions"]:
                raise ValueError("No content found")
                
            content = page["revisions"][0]["slots"]["main"]["content"]
            
            # Parse sections from the content
            sections = []
            current_section = {"level": 0, "title": "Introduction", "content": []}
            current_content = []
            
            for line in content.split('\n'):
                # Check for section headers (== Title ==)
                if line.strip().startswith('=='):
                    # Count the number of = signs to determine level
                    left = len(line.strip()) - len(line.strip().lstrip('='))
                    right = len(line.strip()) - len(line.strip().rstrip('='))
                    if left == right:  # Balanced = signs
                        # This is a section header
                        if current_content:
                            current_section["content"] = '\n'.join(current_content).strip()
                            sections.append(current_section)
                            current_content = []
                        
                        level = left//2
                        title = line.strip('= \t\n\r')
                        current_section = {"level": level, "title": title, "content": []}
                        continue
                
                current_content.append(line)
            
            # Add the last section
            if current_content:
                current_section["content"] = '\n'.join(current_content).strip()
                sections.append(current_section)
            
            # Extract templates (like infoboxes) from the first section
            if sections:
                intro = sections[0]["content"]
                templates = []
                template_start = -1
                brace_count = 0
                
                for i, char in enumerate(intro):
                    if char == '{' and i+1 < len(intro) and intro[i+1] == '{':
                        if brace_count == 0:
                            template_start = i
                        brace_count += 1
                    elif char == '}' and i+1 < len(intro) and intro[i+1] == '}':
                        brace_count -= 1
                        if brace_count == 0 and template_start != -1:
                            template = intro[template_start:i+2]
                            templates.append(template)
                
                # Add templates to result
                result = {
                    "title": page.get("title", ""),
                    "pageid": page.get("pageid"),
                    "url": page.get("fullurl", ""),
                    "content": content,
                    "categories": [cat.get("title", "") for cat in page.get("categories", [])],
                    "timestamp": page["revisions"][0].get("timestamp"),
                    "sections": sections,
                    "templates": templates
                }
            else:
                result = {
                    "title": page.get("title", ""),
                    "pageid": page.get("pageid"),
                    "url": page.get("fullurl", ""),
                    "content": content,
                    "categories": [cat.get("title", "") for cat in page.get("categories", [])],
                    "timestamp": page["revisions"][0].get("timestamp"),
                    "sections": sections
                }
            
            logging.info(f"Successfully fetched full Wikipedia page: {result['title']}")
            return result

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
            
    async def stream_search_wikipedia(self, queries: List[Dict[str, Any]], config: Dict[str, Any] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Search Wikipedia for relevant articles based on queries.
        
        Args:
            queries: List of query items
            config: Optional configuration parameters including min_score
            
        Yields:
            Search results with metadata including full article content and sections
        """
        config = config or {}
        min_score = config.get('min_score', 0.7)
        seen_articles = set()
        
        try:
            # Ensure session is initialized
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            for query_item in queries:
                if not query_item:
                    logger.error("No query provided")
                    continue
                
                query = query_item.get('query', '')
                if not query:
                    logger.error("No query provided")
                    continue
                    
                search_config = config or {}
                min_percentile = search_config.get('min_percentile', 0.0)
                limit = search_config.get('limit', 100)
                
                # Initialize embeddings if needed
                if self.embeddings is None:
                    self.initialize()
                
                if not self.embeddings:
                    logger.error("Failed to initialize embeddings")
                    return
                
                # Search using embeddings
                raw_results = self.embeddings.search(query, limit=limit)
                
                # Process each article
                for result in raw_results:
                    score = float(result.get('score', 0.0))
                    if score < min_score:
                        continue
                        
                    article_id = result.get('id', '')
                    if article_id in seen_articles:
                        continue
                    seen_articles.add(article_id)
                    
                    try:
                        # Check cache first
                        article_data = self._get_cached_article(article_id)
                        if not article_data:
                            # Fetch and cache if not found
                            article_data = await self.get_full_wikipedia_page(article_id)
                            if article_data:
                                self._cache_article(article_id, article_data)
                        
                        if not article_data:
                            continue
                        
                        # Clean and render the wiki content
                        article_data = self.render_wiki_content(article_data)
                        
                        # Extract a summary from the introduction section
                        intro_section = next((s for s in article_data["sections"] if s["level"] == 0 and s["title"] == "Introduction"), None)
                        summary = intro_section["content"] if intro_section else ""
                        
                        # Clean up the summary by removing templates
                        summary = re.sub(r'\{\{[^}]*\}\}', '', summary)  # Remove templates
                        summary = re.sub(r'\[\[[^\]]*\|([^\]]*)\]\]', r'\1', summary)  # Clean up links
                        summary = re.sub(r'\[\[([^\]]*)\]\]', r'\1', summary)  # Clean up remaining links
                        summary = re.sub(r'<[^>]+>', '', summary)  # Remove HTML tags
                        summary = re.sub(r'\s+', ' ', summary).strip()  # Clean up whitespace
                        
                        yield {
                            **query_item,
                            'article_id': article_id,
                            'title': article_data['title'],
                            'url': article_data['url'],
                            'score': score,
                            'document': article_data['content'],
                            'summary': summary,
                            'sections': article_data['sections'],
                            'categories': article_data['categories'],
                            'timestamp': article_data['timestamp']
                        }
                    except ValueError as e:
                        logger.warning(f"Could not fetch article {article_id}: {str(e)}")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing article {article_id}: {str(e)}", exc_info=True)
                        continue
                    
        except Exception as e:
            logger.error(f"Error in stream_search_wikipedia: {str(e)}", exc_info=True)
            raise

    def _wiki_to_html(self, text: str) -> str:
        """
        Convert MediaWiki markup to HTML using mwparserfromhell.
        
        Args:
            text: Raw MediaWiki markup text
            
        Returns:
            HTML-formatted text
        """
        if not text:
            return ""
            
        # Parse the wikitext
        wikicode = mwparserfromhell.parse(text)
        
        # Convert to HTML
        html_parts = []
        for node in wikicode.nodes:
            if isinstance(node, mwparserfromhell.nodes.heading.Heading):
                level = node.level
                title = str(node.title).strip()
                html_parts.append(f"<h{level}>{title}</h{level}>")
            elif isinstance(node, mwparserfromhell.nodes.text.Text):
                html_parts.append(str(node))
            elif isinstance(node, mwparserfromhell.nodes.wikilink.Wikilink):
                title = str(node.title)
                text = str(node.text) if node.text else title
                html_parts.append(f'<a href="{title}">{text}</a>')
            elif isinstance(node, mwparserfromhell.nodes.external_link.ExternalLink):
                url = str(node.url)
                text = str(node.title) if node.title else url
                html_parts.append(f'<a href="{url}">{text}</a>')
            elif isinstance(node, mwparserfromhell.nodes.tag.Tag):
                if node.tag == 'ref':
                    continue  # Skip references
                html_parts.append(str(node))
            elif isinstance(node, mwparserfromhell.nodes.template.Template):
                continue  # Skip templates
            else:
                html_parts.append(str(node))
        
        html_text = ''.join(html_parts)
        
        # Convert remaining wiki markup
        html_text = re.sub(r"'''(.*?)'''", r'<strong>\1</strong>', html_text)  # Bold
        html_text = re.sub(r"''(.*?)''", r'<em>\1</em>', html_text)  # Italic
        
        # Convert lists
        lines = html_text.split('\n')
        in_list = False
        list_type = None
        processed_lines = []
        
        for line in lines:
            if line.startswith('* '):
                if not in_list or list_type != 'ul':
                    if in_list:
                        processed_lines.append(f'</{list_type}>')
                    processed_lines.append('<ul>')
                    in_list = True
                    list_type = 'ul'
                processed_lines.append(f'<li>{line[2:]}</li>')
            elif line.startswith('# '):
                if not in_list or list_type != 'ol':
                    if in_list:
                        processed_lines.append(f'</{list_type}>')
                    processed_lines.append('<ol>')
                    in_list = True
                    list_type = 'ol'
                processed_lines.append(f'<li>{line[2:]}</li>')
            else:
                if in_list:
                    processed_lines.append(f'</{list_type}>')
                    in_list = False
                processed_lines.append(line)
        
        if in_list:
            processed_lines.append(f'</{list_type}>')
        
        html_text = '\n'.join(processed_lines)
        
        # Clean up whitespace
        html_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', html_text)
        html_text = re.sub(r'  +', ' ', html_text)
        
        return html_text.strip()
        
    def render_wiki_content(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render MediaWiki content as HTML using mwparserfromhell.
        
        Args:
            content: Dictionary containing Wikipedia article data
            
        Returns:
            Dictionary with HTML-rendered content ready for display
        """
        if not content:
            return content
            
        # Convert main content to HTML
        content['content'] = self._wiki_to_html(content['content'])
        
        # Convert each section's content
        if 'sections' in content:
            for section in content['sections']:
                section['content'] = self._wiki_to_html(section['content'])
                
        return content