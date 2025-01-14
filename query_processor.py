from typing import Dict, List, Any, AsyncGenerator
import logging
from llm_manager import LLMManager

logger = logging.getLogger(__name__)

class QueryProcessor:
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager

    async def decompose_query(self, query: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Break down a complex query into simpler sub-queries."""
        try:
            logger.info(f"Decomposing query: {query}")
            
            yield {
                'type': 'progress',
                'status': 'Decomposing query...',
                'thought': 'Breaking down complex query...',
                'stream': '🔄 Starting query decomposition...'
            }
            
            prompt = f"""Break down this research query into focused sub-queries. Return a JSON array of strings, where each string is a focused sub-query that explores a specific aspect.

Original query: "{query}"

Respond with only a JSON array of strings."""

            sub_queries = await self.llm_manager.get_json_response(prompt)
            
            logger.info(f"Successfully decomposed query into {len(sub_queries)} sub-queries")
            yield {
                'type': 'result',
                'status': 'Decomposition complete',
                'thought': 'Query broken down into components',
                'stream': '✅ Query decomposition complete',
                'data': sub_queries
            }
            
        except Exception as e:
            logger.error(f"Query decomposition failed: {str(e)}", exc_info=True)
            yield {
                'type': 'error',
                'status': 'Decomposition failed',
                'thought': 'Error during decomposition',
                'stream': f'❌ Decomposition error: {str(e)}'
            }

    async def enrich_query(self, query: str, params: Dict[str, str]) -> AsyncGenerator[Dict[str, Any], None]:
        """Enrich a query with additional context and variations."""
        try:
            # Create prompt for query enrichment
            prompt = f"""Given this search query: "{query}"
            Generate an enriched version focusing on {params['focus']} aspects with {params['depth']} analysis.
            Consider:
            1. Key concepts and terminology
            2. Related topics and themes
            3. Historical context if relevant
            4. Technical details if appropriate
            
            Return ONLY the enriched query as a single line of text, no explanation needed."""
            
            logger.info(f"Enriching query with focus: {params['focus']}, depth: {params['depth']}")
            
            # Get enriched query from LLM
            enriched = await self.llm_manager.get_string_response(prompt)
            if not enriched:
                logger.warning("Failed to get enriched query")
                return
                
            # Clean up response
            enriched = enriched.strip().strip('"\'')
            if not enriched:
                logger.warning("Empty enriched query")
                return
                
            logger.info(f"Enriched query: {enriched}")
            
            # Yield the enriched query
            yield {
                'type': 'result',
                'data': {
                    'original_query': query,
                    'enriched_query': enriched,
                    'params': params
                },
                'status': 'Query enriched',
                'thought': f'Generated {params["focus"]} variation',
                'stream': f'✨ Generated variation: {enriched}'
            }
            
        except Exception as e:
            logger.error(f"Error enriching query: {str(e)}", exc_info=True)
            yield {
                'type': 'error',
                'data': {'error': str(e)},
                'status': 'Error enriching query',
                'thought': 'Failed to enrich query',
                'stream': f'❌ Error generating variation: {str(e)}'
            }

    async def refine_query(self, query: str, context: List[Dict]) -> AsyncGenerator[Dict[str, Any], None]:
        """Refine the search query based on initial search results."""
        yield {
            'type': 'progress',
            'status': 'Analyzing initial results...',
            'thought': 'Reviewing search context',
            'stream': '🔍 Analyzing search results...'
        }
        
        context_text = "\n".join(
            f"Title: {result.get('title', '')}\nExcerpt: {result.get('text', '')[:200]}..."
            for result in context[:3]
        )
        
        prompt = f"""Based on these initial search results, refine this query to get more relevant information.
        Consider what might be missing or what aspects need more focus.
        
        Original query: {query}
        
        Initial results:
        {context_text}
        
        Format the response as a single string containing the refined query."""
        
        try:
            yield {
                'type': 'progress',
                'status': 'Refining query...',
                'thought': 'Using initial results to improve search',
                'stream': '🤖 Refining search terms...'
            }
            
            refined_query = await self.llm_manager.get_string_response(prompt)
            
            if refined_query:
                logger.info("Successfully refined query")
                yield {
                    'type': 'result',
                    'status': 'Query refined',
                    'thought': 'Improved query based on results',
                    'stream': '✅ Search terms refined',
                    'data': refined_query
                }
            else:
                logger.warning("Empty refined query, using original")
                yield {
                    'type': 'warning',
                    'status': 'Could not refine query',
                    'thought': 'Using original query instead',
                    'stream': '⚠️ Using original query',
                    'data': query
                }
        except Exception as e:
            logger.error(f"Error refining query: {str(e)}")
            yield {
                'type': 'error',
                'status': 'Error refining query',
                'thought': f'Error: {str(e)}',
                'stream': '❌ Query refinement failed',
                'data': query
            }
