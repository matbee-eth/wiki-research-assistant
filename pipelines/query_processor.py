import string
from typing import Dict, Any, List, Optional, Union
import logging
import json
import asyncio
import re
from llm_manager import LLMManager

logger = logging.getLogger(__name__)

class QueryProcessor:
    """Processes queries through various transformations and enrichments."""
    
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager

    async def _decompose_single_query(self, query: string, config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Decompose a single query into multiple sub-queries."""
        logger.info(f"Decomposing query: {query}")
        
        prompt = """Break down this research query into focused sub-queries. Return a JSON array of strings, where each string is a focused sub-query that explores a specific aspect.

Original query: "{}"

Respond with only a JSON array of strings.""".format(query)
        
        response = await self.llm_manager.get_string_response(prompt=prompt)
        sub_queries = await self._parse_json_response(response)
        
        return sub_queries

    async def decompose_queries(self, data_items: List[Dict[str, Any]], config: Dict[str, Any] = {}) -> List[Dict[str, Any]]:
        """Map each query to multiple sub-queries."""
        tasks = [self._decompose_single_query(item, config) for item in data_items]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        
        # Create new list with all results
        processed_results = []
        processed_results.extend(data_items)  # Keep original items
        for result in results:
            processed_results.extend(result)  # Add sub-queries
                
        return processed_results

    async def _enrich_single_query(self, query: string, config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Enrich a single query into multiple variations."""
        logger.info(f"Enriching query: {query}")
        
        prompt = """As an expert in semantic search and knowledge graphs, analyze this query and generate variations of it to explore different aspects.
        Consider historical, cultural, and domain-specific relationships.
        Return a JSON array of strings, where each string is a variation that might yield relevant results.

Original query: "{}"

Respond with only a JSON array of strings.""".format(query)
        
        response = await self.llm_manager.get_string_response(prompt=prompt)
        enriched_queries = await self._parse_json_response(response)
            
        return enriched_queries

    async def enrich_queries(self, data_items: List[Dict[str, Any]], config: Dict[str, Any] = {}) -> List[Dict[str, Any]]:
        """Map each query to multiple enriched variations."""
        tasks = [self._enrich_single_query(item, config) for item in data_items]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        
        # Flatten results
        processed_results = []
        for result in results:
            processed_results.extend(result)  # Add all variations
                
        return processed_results

    async def _analyze_single_query(self, query: string, config: Dict[str, Any] = None):
        """
        Analyze a single query to identify key components.
        Internal helper method for analyze_queries.
        """
        prompt = f"""
        Step 1. 
        Complete a step-by-step analysis of this query. Analyze this research query and identify its key components.
            Consider:
            - Topic or subject
            - Time period or temporal constraints
            - Geographic scope
            - Required evidence types
            - Specific claims to verify

        Query: "{query}"

        Step 2.
            Return a JSON object with the following structure:
            <nitpick>
            "claims_to_verify" field entries should not use words like "it", "this", "that", "they", "them" to refer to the query or topic, specify the topic in place of these vague terms.
            </nitpick>
            {[{
                "topic": "subject",
                "temporal_scope": "time period",
                "geographic_scope": "location",
                "evidence_types": ["list", "of", "evidence", "types"],
                "claims_to_verify": ["list", "of", "claims", "that", "include", "topic", "in", "description"]
            },
            {
                "topic": "second subject",
                "temporal_scope": "time period for second subject",
                "geographic_scope": "location",
                "evidence_types": ["list", "of", "evidence", "types"],
                "claims_to_verify": ["list", "of", "claims", "that", "include", "topic", "in", "description"]
            }]}
        """
        
        raw_response = await self.llm_manager.get_string_response(prompt=prompt)
        analysis = await self._parse_json_response(raw_response)
        logger.info(f"Query analysis for {query}: {analysis}")
        
        # Ensure analysis is a list
        if isinstance(analysis, dict):
            analysis = [analysis]
        elif not isinstance(analysis, list):
            analysis = [analysis] if analysis else []
            
        claims = [claim for item in analysis for claim in item.get('claims_to_verify', [])]
        return claims

    async def analyze_queries(self, data_items: List[Dict[str, Any]], config: Dict[str, Any] = {}) -> List[Dict[str, Any]]:
        """
        Analyze multiple queries to identify key components in parallel.
        
        Args:
            data_items: List of queries or dictionaries containing queries
            config: Optional configuration parameters
            
        Returns:
            List of dictionaries containing query analysis and metadata
        """
        logger.info(f"Analyzing {len(data_items)} queries")
        tasks = [self._analyze_single_query(item, config) for item in data_items]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        
        # Handle any exceptions
        processed_results = []
        processed_results.extend(data_items)
        for result in results:
            processed_results.extend(result)

        return processed_results

    async def _parse_json_response(self, response: str) -> Union[List[str], Dict[str, Any]]:
        """Parse a JSON response from the LLM."""
        try:
            if not response:
                return {}
                
            # Function to try parsing JSON from a string
            def try_parse_json(text: str) -> Optional[Union[List, Dict]]:
                try:
                    parsed = json.loads(text.strip())
                    if isinstance(parsed, (list, dict)):
                        return parsed
                except json.JSONDecodeError:
                    return None

            # First try to find JSON in code blocks
            code_blocks = re.finditer(r'```(?:json)?\s*([\s\S]*?)```', response)
            for block in code_blocks:
                content = block.group(1).strip()
                parsed = try_parse_json(content)
                if parsed is not None:
                    return parsed

            # If no valid JSON in code blocks, try to find JSON arrays/objects directly
            json_patterns = [
                r'\[\s*{[\s\S]*}\s*\]',  # JSON array of objects
                r'\{[\s\S]*\}',          # JSON object
                r'\[[\s\S]*\]'           # JSON array
            ]
            
            for pattern in json_patterns:
                matches = re.finditer(pattern, response)
                for match in matches:
                    content = match.group()
                    parsed = try_parse_json(content)
                    if parsed is not None:
                        return parsed

            # If still no valid JSON found, log the response and return empty
            logger.error(f"No valid JSON found in response: {response}")
            return {}
            
        except Exception as e:
            logger.error(f"Error parsing JSON response: {str(e)}")
            logger.error(f"Original response was: {response}")
            return {}
