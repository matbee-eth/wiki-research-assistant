import string
from typing import Dict, Any, List, Optional, Union, AsyncGenerator
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

    async def _decompose_single_query(self, item: Dict[str, Any], config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Decompose a single query into multiple sub-queries."""
        
        system_prompt = """You are a query decomposition expert.
Your task is to break down research queries into focused semantic search queries.
You must return ONLY a JSON array of strings, where each string is a focused sub-query that explores a specific aspect.
Adhere to the provided analysis information regarding topic, time period, geographic scope, and evidence types.
Consider variations of the original query that might yield more relevant results."""

        # Combine analysis and query into a single prompt
        prompt = f"""Analysis: {item.get('analysis', '')}
Query: {item.get('query', '')}"""

        response = await self.llm_manager.get_string_response(
            prompt=prompt,
            system_prompt=system_prompt
        )
        sub_queries = await self._parse_json_response(response)
        # Map each sub-query to the same data in 'item' but update the 'query' field
        decomposed_items = [
            {**item, 'query': sub_query}
            for sub_query in sub_queries
        ]
        return decomposed_items

    async def decompose_queries(self, data_items: List[Dict[str, Any]], config: Dict[str, Any] = {}) -> AsyncGenerator[Dict[str, Any], None]:
        """Map each query to multiple sub-queries."""
        for item in data_items:
            decomposed = await self._decompose_single_query(item, config)
            for result in decomposed:
                yield result

    async def _enrich_single_query(self, item: Dict[str, Any], config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Enrich a single query into multiple variations."""
        
        system_prompt = """You are an expert in semantic search and knowledge graphs.
Your task is to analyze queries and generate variations to explore different aspects.
Consider historical, cultural, and domain-specific relationships.
You must return ONLY a JSON array of strings, where each string is a variation that might yield relevant results."""

        response = await self.llm_manager.get_string_response(
            prompt=item.get('query', ''),
            system_prompt=system_prompt
        )
        enriched_queries = await self._parse_json_response(response)
            
        return enriched_queries

    async def enrich_queries(self, data_items: List[Dict[str, Any]], config: Dict[str, Any] = {}) -> AsyncGenerator[Dict[str, Any], None]:
        """Map each query to multiple enriched variations."""
        for item in data_items:
            enriched = await self._enrich_single_query(item, config)
            for result in enriched:
                yield result

    async def _analyze_single_query(self, query: Dict[str, Any], config: Dict[str, Any] = None):
        """
        Analyze a single query to identify key components.
        Internal helper method for analyze_queries.
        """
        system_prompt = """You are a query analysis expert.
Your task is to analyze research queries and identify their key components in a structured format.
You must follow these steps:

1. Analyze the query considering:
   - Topic or subject
   - Time period or temporal constraints
   - Geographic scope
   - Required evidence types
   - Specific, non-vague, and direct claims to verify against any possibly relevant documents. Used for semantic filtering.
   - Mandatory claims to verify. Think: "This article contains information about Italy or Spain" or "This article is about Roads in Feudal Japan"

2. Return a JSON object with this structure using the analysis given:
   [{
     "topic": "subject",
     "temporal_scope": "time period",
     "geographic_scope": "location",
     "evidence_types": ["list", "of", "evidence", "types"],
     "claims_to_verify": ["list", "of", "claims", "that", "include", "topic", "geographic", "temporal", "in", "description"],
     "mandatory_claims_to_verify": ["list", "of", "claims", "that", "include", "topic", "geographic", "temporal", "in", "description"]
   }]

Important:
- Claims must not use words like "it", "this", "that", "they", "them" to refer to the query or topic
- Specify the topic explicitly in place of vague terms
- Claims must strictly adhere to the query, topic, geographic and timeline constraints"""

        raw_response = await self.llm_manager.get_string_response(
            prompt=query,
            system_prompt=system_prompt
        )
        analysis = await self._parse_json_response(raw_response)
        
        # Ensure analysis is a list
        if isinstance(analysis, dict):
            analysis = [analysis]
        elif not isinstance(analysis, list):
            analysis = [analysis] if analysis else []
        
        # Map each entry in the analysis list to a dictionary with query and analysis
        mapped_results = [{"query": query, "analysis": entry} for entry in analysis]
        
        return mapped_results
        
    
    async def analyze_queries(self, data_items: List[Dict[str, Any]], config: Dict[str, Any] = {}) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Analyze multiple queries to identify key components in parallel.
        
        Args:
            data_items: List of queries or dictionaries containing queries
            config: Optional configuration parameters
            
        Yields:
            Dictionaries containing query analysis and metadata
        """
        tasks = []
        for item in data_items:
            if isinstance(item, str):
                query = item
            elif isinstance(item, dict):
                query = item.get('query', '')
            else:
                continue

            if not query:
                continue

            tasks.append(self._analyze_single_query({"query": query}, config))

        if not tasks:
            return

        try:
            analyses = await asyncio.gather(*tasks)
            for item, analysis_list in zip(data_items, analyses):
                if not analysis_list:
                    continue
                    
                # Preserve the original item data and add analysis
                for analysis in analysis_list:
                    result = item.copy() if isinstance(item, dict) else {"query": item}
                    result['analysis'] = analysis.get('analysis', {})
                    yield result

        except Exception as e:
            logger.error(f"Error analyzing queries: {str(e)}")
            return

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
