import string
from typing import AsyncGenerator, Dict, Any, Union, List
import logging
import asyncio
from llm_manager import LLMManager

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Ensure module-level debug logging is enabled

class FactChecker:
    """A processor for validating claims against documents using LLM models."""
    
    def __init__(self, llm_manager: LLMManager):
        """
        Initialize the fact checker with an LLM manager for validation.
        
        Args:
            llm_manager: The LLM manager instance to use for validation
        """
        logger.debug("Initializing FactChecker with LLM manager")
        self.llm_manager = llm_manager

    async def _generate_single_claim(self, data: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a single sentence claim for a single query.
        Internal helper method for generate_claims.
        """
        try:
            logger.debug("Starting single claim generation")
            config = config or {}
            query = data.get('query', '')
            logger.debug(f"Generating claim for query: {query} {data}")
            
            prompt = f"""
Given the following research query, generate a single sentence claim or a Yes/No answerable question that can be fact-checked. The claim should conform the query into a Question or Claim. Do not modify or introduce any additional context to the nature of the query.
Query: {query}

Your response should be in the following format:
<Examples>
<Response Example 1>
Claim: [A factual statement related to the query]
<Response Example 2>
Claim: [Another factual statement related to the query]
<Response Example 3>
Question: [A Yes/No question related to the query]
<Response Example 4>
Question: [Another Yes/No question related to the query]
</Examples>

Your response should ONLY be ONE of the above formats. Only one claim or question should be generated.
"""
            logger.debug(f"Using prompt for claim generation: {prompt}")

            response = await self.llm_manager.get_string_response(
                prompt=prompt,
                model='phi4'
            )
            logger.debug(f"Received response from phi4 model: {response}")
            
            if not response:
                logger.warning("No response received from model for claim generation")
                return {
                    "success": False,
                    "error": "Failed to generate claim - no response from model",
                    "original_data": data
                }
                
            # Add the generated claim to the data dictionary for the validate step
            claim = response.strip()
            # Update the pipeline data
            data['claim'] = claim
                    
            logger.debug(f"Generated claim: {claim}")
            # logger.debug(f"Returning claim generation result: {result}")
            return data
            
        except Exception as e:
            logger.error(f"Claim generation failed: {str(e)}", exc_info=True)
            raise

    async def generate_claims(self, data_items: List[Dict[str, Any]], config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Generate claims for multiple queries in parallel.
        
        Args:
            data_items: List of dictionaries containing queries to generate claims from
            config: Optional configuration parameters
            
        Returns:
            List of dictionaries containing generated claims and metadata
        """
        logger.debug(f"Generating claims for {len(data_items)} items")
        tasks = [self._generate_single_claim(item, config) for item in data_items]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            processed_results.append(result)
        logger.debug(f"Generated {len(processed_results)} claims for {len(data_items)} items")
        return processed_results

    async def _create_validation_prompt(self, claim: str, document: str) -> str:
        """Create the prompt for claim validation."""
        logger.debug(f"Creating validation prompt for claim: {claim}")
        prompt = f"""Document: {document}
Claim: {claim}
"""
        return prompt

    async def _validate_single_claim(self, claim: string, data: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Validate a single claim against its document.
        Internal helper method for validate_claims.
        """
        try:
            document = data.get('document', '')
            logger.debug(f"Validating claim: {claim} against document: {document}")
            if not claim or not document:
                logger.error("No claim or document provided for validation")
                return {
                    'is_valid': False,
                    'error': 'No document provided',
                }
            
            prompt = await self._create_validation_prompt(claim, document)
            validation = await self.llm_manager.get_string_response(prompt=prompt, model="bespoke-minicheck")
            logger.debug(f"Validation result for claim: {claim} - {validation}")
            # Return just the validation result
            return {
                'is_valid': validation == 'Yes',
            }
            
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}", exc_info=True)
            raise

    async def validate_claims(self, data_items: List[Dict[str, Any]], config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Validate claims against their documents.
        Only returns items where the claim is validated as True.
        
        Args:
            data_items: List of dictionaries, each containing a 'results' list to validate
            config: Optional configuration parameters
            
        Returns:
            List of dictionaries that have valid claims
        """
        logger.debug(f"Validating {len(data_items)} items")
        
        tasks = [self._validate_single_claim(item.get('claim', ''), item, config) for item in data_items]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        # Filter valid results while preserving data structure
        valid_results = []
        for i, (item, validation) in enumerate(zip(data_items, results)):
            if isinstance(validation, Exception):
                logger.error(f"Error in validation {i}: {str(validation)}")
                continue
            claim = item.get('claim', '')
            logger.debug(f"validate_claims Validation result for claim: {claim} - {validation}")
                
            if validation.get('is_valid', False):
                logger.debug(f"Valid result for: {item} {claim}")
                valid_results.append(item)
            else:
                logger.debug(f"Filtered out result for: {item} {claim}")
                
        return valid_results