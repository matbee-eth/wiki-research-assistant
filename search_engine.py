# search_engine.py

import asyncio
import json
from typing import List, Dict, Any, Optional, AsyncGenerator, TYPE_CHECKING
from data_sources import DataSources
from nlp_utils import preprocess_text, extract_entities, perform_topic_modeling
from utils import cache_results, retry_on_error
import logging
import aiohttp
from config import OPENAI_API_KEY
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from pathlib import Path
from datetime import datetime, timedelta
from exporters import SearchResultExporter
from stream_interface import StreamInterface

# Import new modules
from cache_manager import CacheManager
from query_processor import QueryProcessor
from result_processor import ResultProcessor
from llm_manager import LLMManager

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class SearchEngine:
    def __init__(self, min_score: float = 0.8):
        """Initialize the search engine."""
        logger.info("Initializing SearchEngine")
        self.min_score = min_score
        self.data_source = None
        self.session = None
        
        # Initialize managers
        self.cache_manager = CacheManager()
        self.llm_manager = LLMManager()
        self.query_processor = QueryProcessor(self.llm_manager)
        self.result_processor = ResultProcessor()
        self.exporter = SearchResultExporter()

    async def initialize(self):
        """Initialize the search engine components."""
        if not self.data_source:
            self.data_source = WikipediaDataSource()
            await self.data_source.initialize()

    async def initialize_data_source(self):
        """Initialize the data source and other components."""
        if self.data_source is None:
            self.data_source = DataSources()
            await self.data_source.__aenter__()
            logger.info("Data source initialized")

    async def __aenter__(self):
        """Async context manager entry."""
        logger.info("Setting up SearchEngine resources")
        await self.llm_manager.__aenter__()
        await self.initialize_data_source()  # Ensure data source is initialized
        logger.info("SearchEngine setup complete")
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        logger.info("Cleaning up SearchEngine resources")
        try:
            await self.llm_manager.__aexit__(exc_type, exc_val, exc_tb)
            if self.data_source:
                await self.data_source.__aexit__(exc_type, exc_val, exc_tb)
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}", exc_info=True)

    async def search(
        self,
        query: str,
        min_score: float = 0.7,
        max_variations: int = 2,
        chunk_size: int = 300
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Search for relevant content based on the query."""
        try:
            # Initialize data source if not already initialized
            if not self.data_source:
                await self.initialize_data_source()

            variations = [query]  # Start with original query
            progress = 0.1
            progress_step = 0.1
            total_results = 0

            # Get query variations if requested
            if max_variations > 1:
                # Generate query variations
                enrichment_params = [
                    {'focus': 'broad', 'depth': 'shallow'},
                    {'focus': 'specific', 'depth': 'deep'},
                    {'focus': 'technical', 'depth': 'medium'},
                    {'focus': 'historical', 'depth': 'deep'},
                    {'focus': 'contemporary', 'depth': 'medium'}
                ][:max_variations-1]  # Limit to max_variations-1 since we already have the original query
                
                # Get query variations using query processor
                for params in enrichment_params:
                    yield {
                        'stream': f"✨ Generating variation with {params['focus']} focus...",
                        'progress': min(progress, 0.99)
                    }
                    
                    async for update in self.query_processor.enrich_query(query, params):
                        if update.get('type') == 'result':
                            data = update.get('data', {})
                            enriched = data.get('enriched_query')
                            if enriched and enriched not in variations:
                                variations.append(enriched)
                                break
                    
                    progress = min(progress + progress_step, 0.99)

            # Process each variation
            for i, variation in enumerate(variations):
                logger.info(f"Processing variation {i+1}/{len(variations)}: {variation}")
                
                # Update progress for starting new variation
                yield {
                    'stream': f"🔍 Searching for: {variation}",
                    'progress': progress + (i/len(variations)) * 0.4
                }

                # Process results in batches
                current_batch = []
                
                async for batch in self.data_source.stream_search_wikipedia(
                    variation,
                    min_score=min_score
                ):
                    # Add new results to current batch
                    if batch:
                        current_batch.extend(batch)
                        yield {
                            'stream': f"📚 Found {len(batch)} new results...",
                            'progress': min(progress + (i/len(variations)) * 0.5, 0.99)
                        }
                    
                    # Process batch when it reaches size limit
                    if len(current_batch) >= 10:
                        logger.info(f"Processing batch of {len(current_batch)} results")
                        async for update in self.result_processor.process_result_batch(
                            current_batch, 
                            variation,
                            self.llm_manager,
                            self.cache_manager
                        ):
                            # Pass through all updates
                            yield update
                            
                            # Track total results for reporting
                            if update['type'] == 'detailed_result':
                                total_results += 1
                                # Store the result for later retrieval
                                self.result_processor.add_result(update['data'])
                                
                        current_batch = []  # Clear the batch after processing

                # Process any remaining results in the final batch
                if current_batch:
                    logger.info(f"Processing final batch of {len(current_batch)} results")
                    async for update in self.result_processor.process_result_batch(
                        current_batch,
                        variation,
                        self.llm_manager,
                        self.cache_manager
                    ):
                        # Pass through all updates
                        yield update
                        
                        # Track total results for reporting
                        if update['type'] == 'detailed_result':
                            total_results += 1
                            # Store the result for later retrieval
                            self.result_processor.add_result(update['data'])
            # Final progress update
            yield {
                'stream': f"✅ Search complete! Found {total_results} relevant articles",
                'progress': 1.0
            }

        except Exception as e:
            logger.error(f"Search error: {str(e)}", exc_info=True)
            yield {
                'stream': f"❌ Error during search: {str(e)}",
                'progress': 1.0
            }
            
        finally:
            # Ensure we clean up resources
            try:
                if hasattr(self.data_source, '__aexit__'):
                    await self.data_source.__aexit__(None, None, None)
                if hasattr(self.llm_manager, '__aexit__'):
                    await self.llm_manager.__aexit__(None, None, None)
            except Exception as e:
                logger.error(f"Error cleaning up resources: {str(e)}", exc_info=True)

    def get_all_results(self):
        """Get all collected results sorted by score."""
        return self.result_processor.get_all_results()

    def clear_results(self):
        """Clear all collected results."""
        self.result_processor.clear_results()

    async def batch_search(self, queries: List[str], min_score: float = None) -> List[Dict[str, Any]]:
        """Process multiple search queries concurrently."""
        logger.info(f"Starting batch search for {len(queries)} queries")
        tasks = []
        
        async def process_single_query(query: str) -> Dict[str, Any]:
            results = []
            async for update in self.search(query, min_score=min_score if min_score is not None else self.min_score):
                if update.get("type") == "results":
                    results = update.get("data", [])
            return {"query": query, "results": results}
        
        for query in queries:
            tasks.append(process_single_query(query))
        
        try:
            results = await asyncio.gather(*tasks)
            logger.info(f"Completed batch search for {len(queries)} queries")
            return results
        except Exception as e:
            logger.error(f"Batch search failed: {str(e)}", exc_info=True)
            raise

    async def stream_batch_search(self, queries: List[str], min_score: float = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Process multiple search queries concurrently with streaming updates."""
        logger.info(f"Starting streaming batch search for {len(queries)} queries")
        tasks = {query: self.search(query, min_score=min_score if min_score is not None else self.min_score) 
                for query in queries}
        pending = set(tasks.values())
        
        try:
            while pending:
                done, pending = await asyncio.wait(
                    pending,
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                for task in done:
                    try:
                        result = await task
                        yield result
                    except StopAsyncIteration:
                        continue
                    except Exception as e:
                        logger.error(f"Error processing batch result: {str(e)}", exc_info=True)
                        yield {"type": "error", "data": str(e)}
                        
        except Exception as e:
            logger.error(f"Streaming batch search failed: {str(e)}", exc_info=True)
            yield {
                "type": "error",
                "data": f"Processing failed: {str(e)}"
            }

    async def decompose_query(self, query: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Break down a complex query into simpler sub-queries."""
        async for result in self.query_processor.decompose_query(query):
            yield result

    async def enrich_query(self, query: str, params: Dict[str, str] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Enrich implicit components of the query with additional context and keywords."""
        async for result in self.query_processor.enrich_query(query, params):
            yield result

    async def refine_query(self, query: str, context: List[Dict]) -> AsyncGenerator[Dict[str, Any], None]:
        """Refine the search query based on initial search results."""
        async for result in self.query_processor.refine_query(query, context):
            yield result

    async def analyze_query(self, query: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Analyze the query to understand its intent and key components."""
        async for result in self.query_processor.analyze_query(query):
            yield result

    async def generate_research_plan(self, query: str, analysis: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate a structured research plan based on query analysis."""
        async for result in self.query_processor.generate_research_plan(query, analysis):
            yield result

    async def export_results(self, results: List[Dict[str, Any]], query: str, format: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Export search results in the specified format."""
        async for result in self.exporter.export_results(results, query, format):
            yield result
