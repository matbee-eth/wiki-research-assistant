# search_engine.py

from typing import List, Dict, Any, Optional, AsyncGenerator
from data_sources import DataSources
import logging
from llm_manager import LLMManager
from cache_manager import CacheManager

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class SearchEngine:
    """Main search engine class that coordinates query processing and search operations."""
    
    def __init__(self):
        """
        Initialize the search engine with necessary components.
        
        Args:
            pipeline: Configured pipeline instance for processing queries
        """
        self.llm_manager = LLMManager()
        self.data_sources = DataSources()
        self._results = []
        self.session = None
        self.api_key = None
        self.cache_manager = CacheManager()
        self._article_cache = {}  # Initialize empty cache
        self.exporter = None
        self._all_results = []
        self._seen_urls = set()  # Track seen URLs to prevent duplicates

    async def initialize(self):
        """Initialize the search engine components."""
        if not self.data_sources:
            self.data_sources = DataSources()
            await self.data_sources.initialize()

    async def initialize_data_source(self):
        """Initialize the data source and other components."""
        if self.data_sources is None:
            self.data_sources = DataSources()
            await self.data_sources.__aenter__()

    async def __aenter__(self):
        """Async context manager entry."""
        logger.info("Setting up SearchEngine resources")
        if self.session is None:  # Only create session if not exists
            self.session = aiohttp.ClientSession()
        await self.initialize_data_source()
        logger.info("SearchEngine setup complete")
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        logger.info("Cleaning up SearchEngine resources")
        try:
            if self.session and not self.session.closed:
                logger.debug("Closing aiohttp session...")
                await self.session.close()
                logger.debug("Session closed")
            if self.data_sources:
                logger.debug("Cleaning up data sources...")
                await self.data_sources.__aexit__(exc_type, exc_val, exc_tb)
                logger.debug("Data sources cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}", exc_info=True)
            # Don't re-raise the exception to ensure cleanup continues

    def get_all_results(self):
        """Get all collected results sorted by score."""
        # First deduplicate by URL
        unique_results = []
        seen_urls = set()
        
        for result in self._all_results:
            url = result.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)
                
        # Then sort by score
        return sorted(unique_results, key=lambda x: float(x['score']), reverse=True)

    def clear_results(self):
        """Clear all collected results."""
        self._all_results = []
        self._seen_urls = set()

    async def export_results(self, results: List[Dict[str, Any]], query: str, format: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Export search results in the specified format.
        
        Args:
            results: List of search results to export
            query: Original search query
            format: Export format (html, pdf, parquet, markdown)
            
        Yields:
            Dict containing progress updates, status messages, and final export path
        """
        try:
            # Initialize export process
            yield {
                'stream': "🚀 Initializing export process...",
                'progress': 0.1
            }
            
            # Validate and process results
            yield {
                'stream': "📊 Processing results for export...",
                'progress': 0.3
            }
            
            # Format-specific processing
            yield {
                'stream': f"🔄 Converting to {format.upper()} format...",
                'progress': 0.5
            }
            
            # Create export directory if needed
            export_dir = Path("exports")
            export_dir.mkdir(exist_ok=True)
            
            # Generate export path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = export_dir / f"research_results_{timestamp}.{format}"
            
            # Export data using appropriate format
            if format.lower() == 'html':
                yield {
                    'stream': "🎨 Generating HTML document...",
                    'progress': 0.7
                }
                await self.exporter.export_html(results, query, export_path)
            elif format.lower() == 'pdf':
                yield {
                    'stream': "📄 Generating PDF document...",
                    'progress': 0.7
                }
                await self.exporter.export_pdf(results, query, export_path)
            elif format.lower() == 'parquet':
                yield {
                    'stream': "💾 Generating Parquet file...",
                    'progress': 0.7
                }
                await self.exporter.export_parquet(results, query, export_path)
            elif format.lower() == 'markdown':
                yield {
                    'stream': "📝 Generating Markdown document...",
                    'progress': 0.7
                }
                await self.exporter.export_markdown(results, query, export_path)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            # Complete
            yield {
                'stream': f"✅ Export complete! Saved to {export_path}",
                'progress': 1.0,
                'data': {'export_path': str(export_path)}
            }
            
        except Exception as e:
            error_msg = f"Export error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            yield {
                'stream': f"❌ {error_msg}",
                'progress': 1.0,
                'error': error_msg
            }

    def _get_cached_article(self, article_id: str) -> Optional[Dict[str, Any]]:
        """Get article from cache."""
        return self.cache_manager.get_cached_article(article_id)

    def _cache_article(self, article_id: str, article_data: Dict[str, Any]) -> None:
        """Cache article data."""
        self.cache_manager.cache_article(article_id, article_data)

    def _load_cache(self):
        """Load the search results cache."""
        try:
            # Cache is loaded automatically by CacheManager during initialization
            logger.info("Search results cache loaded successfully")
        except Exception as e:
            logger.error(f"Error loading cache: {str(e)}")
            self._article_cache = {}

    def _save_cache(self):
        """Save the search results cache."""
        try:
            # Cache is saved automatically by CacheManager when caching articles
            logger.info("Search results cache saved successfully")
        except Exception as e:
            logger.error(f"Error saving cache: {str(e)}")
