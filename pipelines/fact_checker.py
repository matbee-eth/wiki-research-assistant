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

Your response should be in ONE of the following formats:

Claim: [A claim that can be fact-checked against a document]
Question: [A Yes/No question that can be fact-checked against a document]
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
                
            # Extract the claim from the response
            lines = response.strip().split('\n')
            claim = next((line.split(':', 1)[1].strip() for line in lines if line.startswith(('Claim:', 'Question:'))), '')
            
            # Update the pipeline data with the generated claim
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

    async def validate_claims(self, items: List[Dict[str, Any]], config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Validate claims against source content.
        
        Args:
            items: List of items containing claims and source content
            config: Optional configuration parameters including min_score
            
        Returns:
            List of validated items
        """
        config = config or {}
        
        logger.debug(f"Validating {len(items)} claims")
        validated_items = []
        for item in items:
            logger.debug(f"Validating claim: {item.get('claim', 'No claim provided')} against item: {item}")
            document = item.get('document', '')
            claims_to_verify = item.get('analysis', {}).get('claims_to_verify', [])
            if not claims_to_verify or not document:
                logger.error("No claims or document provided for validation")
                continue
            
            if claims_to_verify:
                # If claims_to_verify is provided, validate each claim in the list
                for claim_to_verify in claims_to_verify:
                    logger.debug(f"Validating claim: {claim_to_verify} against document: {document}")
                    prompt = await self._create_validation_prompt(claim_to_verify, document)
                    validation = await self.llm_manager.get_string_response(prompt=prompt, model="bespoke-minicheck")
                    logger.debug(f"Validation result for claim: {claim_to_verify} - {validation}")
                    if validation == 'Yes':
                        validated_items.append(item)
                        break
            # Return just the validation result
        logger.debug(f"Validated {len(validated_items)} claims out of {len(items)} items")
        return validated_items