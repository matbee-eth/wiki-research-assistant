from typing import List, Dict, Any
from datetime import datetime
import logging
from nlp_utils import generate_literature_review
from fact_checker import FactChecker

logger = logging.getLogger(__name__)

class ResultProcessor:
    def __init__(self):
        self._all_results = []
        self._seen_urls = set()
        self.fact_checker = None  # Will be initialized when llm_manager is available

    async def process_result_batch(self, batch: List[Dict], query: str, llm_manager, cache_manager):
        """Process a batch of search results with analysis."""
        try:
            if not batch:
                return
                
            logger.info(f"Starting to process batch of {len(batch)} results for query: {query}")
            
            # Initialize fact checker if not already done
            if self.fact_checker is None:
                self.fact_checker = FactChecker(llm_manager)
            
            # Process each result in the batch
            for result in batch:
                try:
                    article_id = str(result.get('pageid', result.get('title', '')))
                    cached = cache_manager.get_cached_article(article_id)
                    
                    if cached:
                        # Handle cached result
                        yield {
                            'type': 'detailed_result',
                            'data': cached,
                            'status': f'Retrieved cached result: {cached.get("title", "Unknown")}',
                            'thought': 'Retrieved from cache',
                            'stream': f'📖 Found in cache: {cached.get("title", "Unknown")}'
                        }
                    else:
                        # Validate relevance using fact checker first
                        yield {
                            'type': 'status_update',
                            'data': {
                                'title': result.get('title', 'Unknown'),
                                'stage': 'fact_checking'
                            },
                            'status': f'Fact-checking: {result.get("title", "Unknown")}',
                            'thought': 'Validating document relevance',
                            'stream': f'🔍 Checking relevance: {result.get("title", "Unknown")}'
                        }
                        
                        validation_result = await self.fact_checker.validate_claim(
                            document=result.get('text', ''),
                            claim=f"This document contains information relevant to: {query}"
                        )

                        # Skip further processing if document is not relevant
                        if not validation_result['is_valid']:
                            logger.info(f"Skipping irrelevant article {article_id}: {validation_result['explanation']}")
                            yield {
                                'type': 'fact_check_result',
                                'data': {
                                    'title': result.get('title', 'Unknown'),
                                    'reason': validation_result['explanation'],
                                    'verdict': 'irrelevant'
                                },
                                'status': f'Skipped irrelevant result: {result.get("title", "Unknown")}',
                                'thought': 'Document deemed not relevant to query',
                                'stream': f'❌ Not relevant: {result.get("title", "Unknown")} - {validation_result["explanation"]}'
                            }
                            continue

                        # Document is relevant, emit fact check success
                        yield {
                            'type': 'fact_check_result',
                            'data': {
                                'title': result.get('title', 'Unknown'),
                                'reason': validation_result['explanation'],
                                'verdict': 'relevant'
                            },
                            'status': f'Validated relevance: {result.get("title", "Unknown")}',
                            'thought': 'Document confirmed relevant to query',
                            'stream': f'✅ Relevant: {result.get("title", "Unknown")} - {validation_result["explanation"]}'
                        }

                        # Generate analysis for relevant result
                        analysis_prompt = f"""Analyze this Wikipedia article excerpt in relation to the query: "{query}"
                        
                        Focus on:
                        1. Key facts and findings relevant to the query
                        2. Historical context and developments
                        3. Impact and implications
                        4. Connections to other relevant topics
                        
                        Article: {result.get('text', '')}
                        
                        Provide a concise, factual analysis focusing on aspects most relevant to the query."""
                        
                        analysis = await llm_manager.get_string_response(
                            prompt=analysis_prompt,
                            model="phi4"  # Use phi4 for analysis
                        )
                        if not analysis:
                            logger.warning(f"Failed to generate analysis for article {article_id}")
                            continue

                        # Generate literature review
                        try:
                            article_text = result.get('text', '')
                            article_title = result.get('title', 'Unknown')
                            
                            if article_text:
                                literature_review = generate_literature_review([article_text])
                                result['literature_review'] = literature_review
                        except Exception as e:
                            logger.error(f"Error generating literature review: {str(e)}")
                            result['literature_review'] = "Error generating literature review"
                        
                        # Process and cache result
                        processed_result = {
                            **result,
                            'analysis': analysis,
                            'content': f"""### Summary\n{result.get('text', 'No summary available.')}\n\n### Analysis\n{analysis}""",
                            'processed_at': datetime.now().isoformat(),
                            'relevance_validation': validation_result
                        }
                        
                        # Cache the processed result
                        cache_manager.cache_article(article_id, processed_result)
                        
                        # Emit the detailed result first
                        yield {
                            'type': 'detailed_result',
                            'data': {
                                **processed_result,
                                'fact_checking_validation': validation_result
                            },
                            'status': f'Processed result: {result.get("title", "Unknown")}',
                            'thought': 'Generated analysis and cached result',
                            'stream': f'📚 Analyzed: {result.get("title", "Unknown")}'
                        }
                        
                        # Then emit the additional events for UI updates
                        yield {
                            'type': 'wiki_summary',
                            'data': result.get('text', 'No summary available.'),
                            'status': f'Retrieved wiki summary for: {result.get("title", "Unknown")}',
                            'thought': 'Retrieved article summary',
                            'stream': f'📖 Summary: {result.get("title", "Unknown")}'
                        }
                        
                        yield {
                            'type': 'analysis',
                            'data': analysis,
                            'status': f'Generated analysis for: {result.get("title", "Unknown")}',
                            'thought': 'Generated article analysis',
                            'stream': f'🔍 Analysis: {result.get("title", "Unknown")}'
                        }
                        
                except Exception as e:
                    logger.error(f"Error processing result: {str(e)}", exc_info=True)
                    continue
                    
        except Exception as e:
            logger.error(f"Batch processing error: {str(e)}", exc_info=True)
            yield {
                'type': 'error',
                'status': 'Batch processing error',
                'thought': 'Error occurred during parallel processing',
                'stream': f'❌ Error: Could not process batch - {str(e)}'
            }

    def _handle_cached_result(self, result: Dict):
        """Handle a cached search result."""
        # Add timestamp if not present
        if 'cached_at' not in result:
            result['cached_at'] = datetime.now().isoformat()

        yield {
            'type': 'result',
            'data': result,
            'status': f'Retrieved from cache: {result.get("title", "Unknown")}',
            'thought': 'Using previously analyzed article',
            'stream': f'📚 Cached: {result.get("title", "Unknown")}'
        }
        
        yield {
            'type': 'wiki_summary',
            'data': result.get('text', 'No summary available.'),
            'status': f'Retrieved wiki summary for: {result.get("title", "Unknown")}',
            'thought': f'📖 Retrieved article summary',
            'stream': f'📖 Summary: {result.get("title", "Unknown")}'
        }
        
        if 'analysis' in result:
            yield {
                'type': 'analysis',
                'data': result['analysis'],
                'status': f'Generated analysis for: {result.get("title", "Unknown")}',
                'thought': f'🔍 Generated article analysis',
                'stream': f'🔍 Analysis: {result.get("title", "Unknown")}'
            }
        
        if 'literature_review' in result:
            yield {
                'type': 'literature_review',
                'data': result['literature_review'],
                'status': f'Generated literature review for: {result.get("title", "Unknown")}',
                'thought': f'📚 Generated article review',
                'stream': f'📚 Review for: {result.get("title", "Unknown")}'
            }

    def _generate_result_events(self, result: Dict):
        """Generate standard result events."""
        yield {
            'type': 'wiki_summary',
            'data': result.get('text', 'No summary available.'),
            'status': f'Retrieved wiki summary for: {result.get("title", "Unknown")}',
            'thought': 'Retrieved article summary',
            'stream': f'📖 Summary: {result.get("title", "Unknown")}'
        }
        
        yield {
            'type': 'analysis',
            'data': result['analysis'],
            'status': f'Generated analysis for: {result.get("title", "Unknown")}',
            'thought': 'Generated article analysis',
            'stream': f'🔍 Analysis: {result.get("title", "Unknown")}'
        }
        
        yield {
            'type': 'result',
            'data': result,
            'status': 'Processing complete',
            'thought': f'Finished analyzing {result.get("title", "Unknown")}',
            'stream': f'✅ Analyzed: {result.get("title", "Unknown")}'
        }

    def add_result(self, result: Dict):
        """Add a processed result to the collection."""
        if not hasattr(self, '_results'):
            self._results = []
        self._results.append(result)
        
    def get_all_results(self) -> List[Dict]:
        """Get all collected results."""
        if not hasattr(self, '_results'):
            self._results = []
        return self._results
        
    def clear_results(self):
        """Clear all collected results."""
        if hasattr(self, '_results'):
            self._results = []

    def get_all_results_sorted(self):
        """Get all collected results sorted by score."""
        unique_results = []
        seen_urls = set()
        
        for result in self._all_results:
            url = result.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)
                
        return sorted(unique_results, key=lambda x: float(x['score']), reverse=True)

    def clear_all_results(self):
        """Clear all collected results."""
        self._all_results = []
        self._seen_urls = set()
