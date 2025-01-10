# search_engine.py

import asyncio
import json
from typing import List, Dict, Any, Optional, AsyncGenerator, TYPE_CHECKING
from data_sources import DataSources
from nlp_utils import preprocess_text, extract_entities, perform_topic_modeling, generate_literature_review
from utils import cache_results, retry_on_error, fetch_gpt_response, fetch_gpt_response_parallel
import logging
import aiohttp
from config import OPENAI_API_KEY
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import json
from pathlib import Path
from datetime import datetime, timedelta
from exporters import SearchResultExporter
from stream_interface import StreamInterface

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
        self.api_key = OPENAI_API_KEY
        self._article_cache = {}  # In-memory cache
        self._cache_dir = Path("article_cache")
        self._cache_dir.mkdir(exist_ok=True)
        self._load_cache()
        self.exporter = SearchResultExporter()
        self._all_results = []
        self._seen_urls = set()  # Track seen URLs to prevent duplicates

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
            if self.data_source:
                logger.debug("Cleaning up data sources...")
                await self.data_source.__aexit__(exc_type, exc_val, exc_tb)
                logger.debug("Data sources cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}", exc_info=True)
            # Don't re-raise the exception to ensure cleanup continues

    async def _call_openai(self, prompt: str, stream: bool = False) -> str:
        """
        Call local LLM using fetch_gpt_response_parallel.
        """
        try:
            logger.info(f"Calling local LLM with prompt: '{prompt}'")
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Call with single prompt (wrapped in list for parallel function)
            responses = await fetch_gpt_response_parallel(
                self.session, 
                [prompt],
                model="phi4",  # Using local LLM
                temperature=0.3,
            )
            
            # Get first (and only) response
            result = responses[0] if responses else ""
            logger.debug(f"Raw LLM Response:\n{result}")
            return result
            
        except Exception as e:
            logger.error(f"Local LLM call failed: {str(e)}", exc_info=True)
            raise

    async def _parse_json_response(self, response: str) -> dict:
        """Parse JSON from LLM response with better error handling."""
        try:
            # Try to find JSON-like content between triple backticks if present
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
            if json_match:
                response = json_match.group(1)
                logger.debug(f"Extracted JSON from markdown: {response}")
            
            # Clean up common formatting issues
            response = response.strip()
            if response.startswith('```') and response.endswith('```'):
                response = response[3:-3].strip()
            
            logger.debug(f"Attempting to parse JSON:\n{response}")
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"JSON Parse Error: {str(e)}")
            logger.error(f"Failed JSON content:\n{response}")
            raise

    async def get_streaming_response(self, prompt: str):
        """
        Get streaming response from OpenAI.
        """
        logger.info(f"Getting streaming response for prompt: '{prompt}'")
        response = await self._call_openai(prompt, stream=True)
        if response:
            logger.debug(f"Received streaming response: {response}")
            return response
        logger.warning("No response generated.")
        return "No response generated."

    async def process_result_batch(self, batch: List[Dict], query: str) -> AsyncGenerator[Dict, None]:
        """
        Process a batch of search results with GPT analysis in parallel.
        
        Args:
            batch: List of search results to process
            query: Original search query
            
        Yields:
            Processed results and progress updates
        """
        try:
            # Filter out already cached articles
            uncached_articles = []
            cached_results = []
            
            for result in batch:
                article_id = str(result.get('pageid', result.get('title', '')))
                cached = self._get_cached_article(article_id)
                if cached:
                    cached_results.append(cached)
                    yield {
                        'type': 'result',
                        'data': cached,
                        'status': f'Retrieved from cache: {result.get("title", "Unknown")}',
                        'thought': 'Using previously analyzed article',
                        'stream': f'📚 Cached: {result.get("title", "Unknown")}'
                    }
                else:
                    uncached_articles.append(result)

            if not uncached_articles:
                return

            # Generate analysis prompts for uncached articles
            prompts = []
            for result in uncached_articles:
                analysis_prompt = f"""Analyze this Wikipedia article excerpt in relation to the query: "{query}"
                
                Focus on:
                1. Key facts and findings relevant to the query
                2. Historical context and developments
                3. Impact and implications
                4. Connections to other relevant topics
                
                Article: {result['text']}
                
                Provide a concise, factual analysis focusing on aspects most relevant to the query."""
                prompts.append(analysis_prompt)
            
            # Start processing notification for uncached articles
            for i, result in enumerate(uncached_articles, 1):
                yield {
                    'type': 'progress',
                    'status': f'Queued article {i}/{len(uncached_articles)}: {result.get("title", "Unknown")}',
                    'thought': f'Preparing to analyze new article...',
                    'stream': f'📋 Queued: {result.get("title", "Unknown")}'
                }
            
            # Process all prompts in parallel with rate limiting
            async with aiohttp.ClientSession() as session:
                # Notify starting GPT analysis
                for i, result in enumerate(uncached_articles, 1):
                    yield {
                        'type': 'progress',
                        'status': f'Starting GPT analysis for article {i}/{len(uncached_articles)}: {result.get("title", "Unknown")}',
                        'thought': 'Sending to GPT for analysis...',
                        'stream': f'🤖 Analyzing: {result.get("title", "Unknown")}'
                    }
                
                analyses = await fetch_gpt_response_parallel(session, prompts)
                
                # Process results as they complete
                for i, (result, analysis) in enumerate(zip(uncached_articles, analyses), 1):
                    try:
                        # Notify starting post-processing
                        yield {
                            'type': 'progress',
                            'status': f'Post-processing article {i}/{len(uncached_articles)}: {result.get("title", "Unknown")}',
                            'thought': 'Processing GPT analysis results...',
                            'stream': f'🔄 Processing: {result.get("title", "Unknown")}'
                        }
                        
                        # Generate literature review for this article
                        try:
                            # Get the article text and title
                            article_text = result.get('text', '')
                            article_title = result.get('title', 'Unknown')
                            
                            if article_text:
                                literature_review = generate_literature_review([article_text])
                                result['literature_review'] = literature_review
                                
                                # Yield the literature review event
                                yield {
                                    'type': 'literature_review',
                                    'data': literature_review,
                                    'status': f'Generated literature review for: {article_title}',
                                    'thought': 'Generated individual article review',
                                    'stream': f'📚 Review for: {article_title}'
                                }
                        except Exception as e:
                            logger.error(f"Error generating literature review for article {result.get('title')}: {str(e)}")
                            result['literature_review'] = "Error generating literature review"
                        
                        # Add analysis to result
                        processed_result = {
                            **result,
                            'analysis': analysis,
                            'content': f"""### Summary\n{result.get('text', 'No summary available.')}\n\n### Analysis\n{analysis}""",
                            'processed_at': datetime.now().isoformat()
                        }
                        
                        # Cache the processed result
                        article_id = str(result.get('pageid', result.get('title', '')))
                        self._cache_article(article_id, processed_result)
                        
                        # Yield the wiki summary event
                        yield {
                            'type': 'wiki_summary',
                            'data': result.get('text', 'No summary available.'),
                            'status': f'Retrieved wiki summary for: {result.get("title", "Unknown")}',
                            'thought': 'Retrieved article summary',
                            'stream': f'📖 Summary: {result.get("title", "Unknown")}'
                        }
                        
                        # Yield the analysis event
                        yield {
                            'type': 'analysis',
                            'data': analysis,
                            'status': f'Generated analysis for: {result.get("title", "Unknown")}',
                            'thought': 'Generated article analysis',
                            'stream': f'🔍 Analysis: {result.get("title", "Unknown")}'
                        }
                        
                        # Yield the processed result
                        yield {
                            'type': 'result',
                            'data': processed_result,
                            'status': f'Completed analysis of article {i}/{len(uncached_articles)}',
                            'thought': f'Finished analyzing {result.get("title", "Unknown")}',
                            'stream': f'✅ Analyzed: {result.get("title", "Unknown")}'
                        }

                    except Exception as e:
                        logger.error(f"Error processing result {result.get('title')}: {str(e)}")
                        yield {
                            'type': 'error',
                            'status': f'Error processing article {i}/{len(uncached_articles)}',
                            'thought': f'Error analyzing {result.get("title", "Unknown")}',
                            'stream': f'❌ Error: Could not analyze {result.get("title", "Unknown")}'
                        }
            
        except Exception as e:
            logger.error(f"Batch processing error: {str(e)}")
            yield {
                'type': 'error',
                'status': 'Batch processing error',
                'thought': 'Error occurred during parallel processing',
                'stream': f'❌ Error: Could not process batch - {str(e)}'
            }

    async def search(
        self,
        query: str,
        max_results: int = 100,
        min_score: float = 0.7,
        max_variations: int = 2,
        chunk_size: int = 300
    ) -> AsyncGenerator:
        """
        Search for relevant content based on the query.
        
        Args:
            query: The search query
            max_results: Maximum number of results to return
            min_score: Minimum similarity score (0-1)
            max_variations: Number of query variations to try
            chunk_size: Size of content chunks to return
            
        Yields:
            Dict containing search progress updates and results
        """
        try:
            # Initialize data source if needed
            if not self.data_source:
                await self.initialize_data_source()
            
            # Clear previous results
            self.clear_results()
            
            # Initialize progress tracking
            total_steps = max_variations * 2  # Enrichment + search for each variation
            progress = 0.0
            progress_step = 1.0 / total_steps if total_steps > 0 else 1.0
            
            # Generate variations through enrichment
            variations = [query]  # Start with original query
            enrichment_params = [
                {'focus': 'broad', 'depth': 'shallow'},
                {'focus': 'specific', 'depth': 'deep'},
                {'focus': 'technical', 'depth': 'medium'},
                {'focus': 'historical', 'depth': 'deep'},
                {'focus': 'contemporary', 'depth': 'medium'}
            ]
            
            # Try different enrichment parameters until we have enough variations
            for params in enrichment_params:
                if len(variations) >= max_variations:
                    break
                    
                yield {
                    'stream': f"✨ Generating variation with {params['focus']} focus...",
                    'progress': min(progress, 0.99)
                }
                
                enriched = None
                async for update in self.enrich_query(query, params):
                    if isinstance(update, dict) and 'data' in update:
                        data = update['data']
                        if isinstance(data, dict) and 'enriched_query' in data:
                            enriched = data['enriched_query']
                            if enriched and enriched not in variations:
                                variations.append(enriched)
                                break
                
                progress = min(progress + progress_step, 0.99)
            
            # Limit to max_variations
            variations = variations[:max_variations]
            
            # Search across variations
            total_results = 0
            
            for i, variation in enumerate(variations, 1):
                yield {
                    'stream': f"✨ Searching variation {i}/{len(variations)}: {variation}",
                    'progress': min(progress, 0.99)
                }
                
                # Search with current variation
                current_batch = []
                async for batch in self.data_source.stream_search_wikipedia(
                    variation,
                    min_score=min_score,
                    max_results=max_results
                ):
                    current_batch.extend(batch)
                    
                    # Process batch when it reaches a reasonable size or is the last batch
                    if len(current_batch) >= 10 or total_results + len(current_batch) >= max_results:
                        async for update in self.process_result_batch(current_batch, variation):
                            if update['type'] == 'result':
                                result = update['data']
                                url = result.get('url', '')
                                if url and url not in self._seen_urls:
                                    self._seen_urls.add(url)
                                    self._all_results.append(result)
                                    total_results += 1
                                    yield {'result': result}
                            elif update['type'] in ['literature_review', 'progress', 'error']:
                                yield update
                                
                            if total_results >= max_results:
                                # Sort final results
                                self._all_results.sort(key=lambda x: float(x['score']), reverse=True)
                                yield {
                                    'stream': f"✅ Search complete! Found {total_results} relevant articles",
                                    'progress': 1.0
                                }
                                return
                        
                        current_batch = []  # Clear the batch after processing
            
            progress = min(progress + progress_step, 0.99)
            yield {
                'stream': f"✨ Found {total_results} unique results so far...",
                'progress': progress
            }
        
            # Sort final results
            self._all_results.sort(key=lambda x: float(x['score']), reverse=True)
        
            # Complete
            yield {
                'stream': f"✅ Search complete! Found {total_results} relevant articles across {len(variations)} query variations",
                'progress': 1.0
            }
        
        except Exception as e:
            logger.error(f"Error in search: {str(e)}", exc_info=True)
            yield {'stream': f"❌ Error during search: {str(e)}"}

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

    async def batch_search(self, queries: List[str], min_score: float = None, max_results: int = 250) -> List[Dict[str, Any]]:
        """
        Process multiple search queries concurrently.
        
        Args:
            queries: List of search queries to process
            min_score: Minimum similarity score threshold
            max_results: Maximum number of results per query
            
        Returns:
            List of search results for each query
        """
        logger.info(f"Starting batch search for {len(queries)} queries")
        tasks = []
        
        async def process_single_query(query: str) -> Dict[str, Any]:
            results = []
            async for update in self.search(query, min_score if min_score is not None else self.min_score, max_results):
                if update.get("type") == "results":
                    results = update.get("data", [])
            return {"query": query, "results": results}
        
        # Create tasks for all queries
        for query in queries:
            tasks.append(process_single_query(query))
        
        # Execute all tasks concurrently
        try:
            results = await asyncio.gather(*tasks)
            logger.info(f"Completed batch search for {len(queries)} queries")
            return results
        except Exception as e:
            logger.error(f"Batch search failed: {str(e)}", exc_info=True)
            raise

    async def stream_batch_search(self, queries: List[str], min_score: float = None, max_results: int = 250) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process multiple search queries concurrently with streaming updates.
        
        Args:
            queries: List of search queries to process
            min_score: Minimum similarity score threshold
            max_results: Maximum number of results per query
            
        Yields:
            Dict containing search progress updates and results for each query
        """
        logger.info(f"Starting streaming batch search for {len(queries)} queries")
        tasks = {query: self.search(query, min_score if min_score is not None else self.min_score, max_results) for query in queries}
        pending = set(tasks.values())
        
        try:
            while pending:
                # Wait for the next result from any query
                done, pending = await asyncio.wait(
                    pending,
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Process completed tasks
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
        """
        Break down a complex query into simpler sub-queries.
        
        Args:
            query: The original search query
            
        Yields:
            Dict containing progress updates and sub-queries
        """
        try:
            logger.info(f"Decomposing query: {query}")
            
            # Initial progress update
            yield {
                'type': 'progress',
                'status': 'Decomposing query...',
                'thought': 'Breaking down complex query...',
                'stream': '🔄 Starting query decomposition...'
            }
            
            # Construct decomposition prompt
            prompt = f"""Break down this research query into focused sub-queries. Return a JSON array of strings, where each string is a focused sub-query that explores a specific aspect.

Original query: "{query}"

Respond with only a JSON array of strings."""

            response = await self._call_openai(prompt)
            if not response:
                raise ValueError("Empty response from LLM")
                
            sub_queries = await self._parse_json_response(response)
            if not isinstance(sub_queries, list):
                raise ValueError(f"Expected list response, got {type(sub_queries)}")
            
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

    async def enrich_query(self, query: str, params: Dict[str, str] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Enrich implicit components of the query with additional context, keywords and related concepts.
        
        Args:
            query: Original search query
            params: Optional enrichment parameters
            
        Yields:
            Dict containing progress updates and enriched data
        """
        logger.info("Enriching query components")
        try:
            # Initial progress update
            yield {
                'type': 'progress',
                'status': 'Enriching query...',
                'thought': 'Adding context and keywords...',
                'stream': '🔄 Starting query enrichment...'
            }

            prompt = f"""
            As an expert in semantic search and knowledge graphs, analyze this query and enrich it with related concepts, 
            synonyms, and contextual information. Consider historical, cultural, and domain-specific relationships.

            Query: "{query}"

            Format your response as a valid JSON object that includes:
            {{
                "enriched_query": "expanded version of the original query",
                "related_concepts": ["list", "of", "related", "concepts"],
                "contextual_expansions": ["expanded", "interpretations"],
                "domain_specific_terms": ["relevant", "domain", "terminology"]
            }}

            Respond only with the JSON object, no other text.
            """
            
            if params:
                prompt += f"\n\nEnrichment parameters: {params}"
            
            raw_response = await self._call_openai(prompt)
            if not raw_response:
                raise ValueError("Empty response from LLM")
                
            enriched = await self._parse_json_response(raw_response)
            if not isinstance(enriched, dict):
                raise ValueError(f"Expected dict response, got {type(enriched)}")
            
            logger.info("Successfully enriched query components")
            yield {
                'type': 'result',
                'status': 'Enrichment complete',
                'thought': 'Query expanded with context',
                'stream': '✅ Query enrichment complete',
                'data': enriched
            }
            
        except Exception as e:
            logger.error(f"Query enrichment failed: {str(e)}", exc_info=True)
            yield {
                'type': 'error',
                'status': 'Query enrichment failed',
                'thought': 'Error during enrichment',
                'stream': f'❌ Error: {str(e)}'
            }

    async def refine_query(self, query: str, context: List[Dict]) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Refine the search query based on initial search results.
        
        Args:
            query: Original search query
            context: List of initial search results
            
        Yields:
            Dict containing progress updates and refined query
        """
        yield {
            'type': 'progress',
            'status': 'Analyzing initial results...',
            'thought': 'Reviewing search context',
            'stream': '🔍 Analyzing search results...'
        }
        
        # Extract relevant information from context
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
            
            response = await self._call_openai(prompt)
            refined_query = response.strip()
            
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

    async def analyze_query(self, query: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Analyze the query to understand its intent and key components.
        
        Args:
            query: Search query to analyze
            
        Yields:
            Dict containing progress updates and analysis results
        """
        try:
            logger.info(f"Analyzing query: {query}")
            
            # Initial progress update
            yield {
                'type': 'progress',
                'status': 'Analyzing query...',
                'thought': 'Understanding query components...',
                'stream': '🔍 Starting query analysis...'
            }
            
            # Construct analysis prompt
            prompt = f"""Analyze this research query and identify its key components. Return a JSON object with:
1. main_topic: The central topic or subject
2. aspects: List of specific aspects or subtopics to explore
3. constraints: Any limitations or specific requirements
4. type: Type of information needed (e.g., historical, technical, comparative)

Query: "{query}"

Respond with only a JSON object."""

            response = await self._call_openai(prompt)
            if not response:
                raise ValueError("Empty response from LLM")
                
            analysis = await self._parse_json_response(response)
            if not isinstance(analysis, dict):
                raise ValueError(f"Expected dict response, got {type(analysis)}")
            
            logger.info("Successfully analyzed query")
            yield {
                'type': 'result',
                'status': 'Analysis complete',
                'thought': 'Query components identified',
                'stream': '✅ Query analysis complete',
                'data': analysis
            }
            
        except Exception as e:
            logger.error(f"Query analysis failed: {str(e)}", exc_info=True)
            yield {
                'type': 'error',
                'status': 'Analysis failed',
                'thought': 'Error during analysis',
                'stream': f'❌ Analysis error: {str(e)}'
            }

    async def generate_research_plan(self, query: str, analysis: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate a structured research plan based on query analysis.
        
        Args:
            query: Original search query
            analysis: Query analysis results
            
        Yields:
            Dict containing progress updates and research plan
        """
        yield {
            'type': 'progress',
            'status': 'Planning research approach...',
            'thought': 'Creating structured research plan',
            'stream': '📋 Developing research strategy...'
        }
        
        prompt = f"""Create a structured research plan for investigating this query.
        Consider the analysis provided and outline specific steps to gather comprehensive information.
        
        Query: {query}
        
        Analysis:
        - Topic: {analysis.get('topic')}
        - Key Aspects: {', '.join(analysis.get('aspects', []))}
        - Information Type: {analysis.get('info_type')}
        - Time Context: {analysis.get('time_context')}
        - Challenges: {', '.join(analysis.get('challenges', []))}
        
        Format the response as a JSON array of objects, each with keys:
        - step_number: integer
        - description: string
        - focus_areas: array of strings
        - expected_outcomes: string"""
        
        try:
            yield {
                'type': 'progress',
                'status': 'Generating research steps...',
                'thought': 'Creating detailed research plan',
                'stream': '🤖 Planning research steps...'
            }
            
            response = await self._call_openai(prompt)
            plan = await self._parse_json_response(response)
            
            if isinstance(plan, list):
                logger.info(f"Generated research plan with {len(plan)} steps")
                yield {
                    'type': 'result',
                    'status': 'Research plan ready',
                    'thought': f'Created {len(plan)} research steps',
                    'stream': '✅ Research plan complete',
                    'data': plan
                }
            else:
                logger.warning("Invalid research plan format")
                default_plan = [{"step_number": 1, "description": query, "focus_areas": [], "expected_outcomes": "General results"}]
                yield {
                    'type': 'warning',
                    'status': 'Could not generate detailed plan',
                    'thought': 'Using basic research approach',
                    'stream': '⚠️ Using simplified plan',
                    'data': default_plan
                }
        except Exception as e:
            logger.error(f"Error generating research plan: {str(e)}")
            default_plan = [{"step_number": 1, "description": query, "focus_areas": [], "expected_outcomes": "General results"}]
            yield {
                'type': 'error',
                'status': 'Error generating plan',
                'thought': f'Error: {str(e)}',
                'stream': '❌ Plan generation failed',
                'data': default_plan
            }

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

    def _load_cache(self):
        """Load cached articles from disk"""
        try:
            cache_file = self._cache_dir / "article_cache.json"
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    # Filter out expired entries (older than 7 days)
                    current_time = datetime.now()
                    self._article_cache = {
                        k: v for k, v in cached_data.items()
                        if datetime.fromisoformat(v['processed_at']) > current_time - timedelta(days=7)
                    }
        except Exception as e:
            logger.error(f"Error loading cache: {str(e)}")
            self._article_cache = {}

    def _save_cache(self):
        """Save article cache to disk"""
        try:
            cache_file = self._cache_dir / "article_cache.json"
            with open(cache_file, 'w') as f:
                json.dump(self._article_cache, f)
        except Exception as e:
            logger.error(f"Error saving cache: {str(e)}")

    def _get_cached_article(self, article_id: str) -> Optional[Dict]:
        """Get article from cache if it exists and is not expired"""
        if article_id in self._article_cache:
            cached = self._article_cache[article_id]
            # Check if cache is still valid (7 days)
            current_time = datetime.now()
            if datetime.fromisoformat(cached['processed_at']) > current_time - timedelta(days=7):
                return cached
            else:
                del self._article_cache[article_id]
        return None

    def _cache_article(self, article_id: str, data: Dict):
        """Cache article data"""
        self._article_cache[article_id] = data
        # Periodically save cache to disk (every 10 new articles)
        if len(self._article_cache) % 10 == 0:
            self._save_cache()

    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Deduplicate results based on URL and sort by score"""
        seen_urls = set()
        unique_results = []
        
        # Sort by score first to keep highest scoring versions
        for result in sorted(results, key=lambda x: x.get('score', 0), reverse=True):
            url = result.get('url', '')
            title = result.get('title', '').lower()  # Case-insensitive title matching
            
            # Use both URL and title as unique identifiers
            if url and url not in seen_urls and title not in seen_urls:
                seen_urls.add(url)
                seen_urls.add(title)  # Also track titles to catch duplicates with different URLs
                unique_results.append(result)
        
        return unique_results

    def _merge_results(self, existing_results: List[Dict], new_results: List[Dict]) -> List[Dict]:
        """Merge new results with existing ones, maintaining uniqueness and order by score"""
        # Combine lists and deduplicate
        all_results = existing_results + new_results
        return self._deduplicate_results(all_results)
