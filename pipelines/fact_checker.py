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
            
            system_prompt = """You are a fact-checking assistant that converts research queries into clear, verifiable claims or Yes/No questions.
Your task is to generate a single sentence claim or a Yes/No answerable question that can be fact-checked.
You must conform the query into a Question or Claim without modifying or introducing any additional context.
Any Geographic or Temporal constraints must be included in the claim or question.
You must strictly adhere to the provided query, topic, geographic and timeline constraints.

Your response must be in ONE of these formats:
Claim: [A non-vague, directed, claim that can be fact-checked against a document]
Question: [A non-vague, directed, Yes/No question that can be fact-checked against a document]"""

            logger.debug(f"Using system prompt for claim generation")

            response = await self.llm_manager.get_string_response(
                prompt=query,
                system_prompt=system_prompt,
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
            return data
            
        except Exception as e:
            logger.error(f"Claim generation failed: {str(e)}", exc_info=True)
            raise

    async def generate_claims(self, items: List[Dict[str, Any]], config: Dict[str, Any] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate claims from items.
        
        Args:
            items: List of items to generate claims for
            config: Optional configuration parameters
            
        Yields:
            Items with generated claims
        """
        for item in items:
            try:
                claim = await self._generate_single_claim(item, config)
                if claim:
                    item['claim'] = claim
                    yield item
            except Exception as e:
                logger.error(f"Error generating claim: {str(e)}", exc_info=True)

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

    async def validate_claims(self, items: List[Dict[str, Any]], config: Dict[str, Any] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Validate claims against source content.
        
        Args:
            items: List of items containing claims and source content
            config: Optional configuration parameters including min_score
            
        Yields:
            Validated items
        """
        config = config or {}
        
        logger.debug(f"Validating {len(items)} claims")
        for item in items:
            document = item.get('document', '')
            if not document:
                logger.error("No document provided for validation")
                continue
                
            # Get claims to verify from analysis or use the document content
            claims_to_verify = []
            if 'analysis' in item and isinstance(item['analysis'], dict):
                claims_to_verify = item['analysis'].get('claims_to_verify', [])
            
            # If no claims in analysis, generate a claim from the document
            if not claims_to_verify:
                try:
                    claim = await self._generate_single_claim(item, config)
                    if claim:
                        claims_to_verify = [claim]
                except Exception as e:
                    logger.error(f"Error generating claim: {str(e)}")
                    continue
            
            if not claims_to_verify:
                logger.error("No claims available for validation")
                continue
            
            # Validate all claims
            validations = []
            valid_count = 0
            for claim_to_verify in claims_to_verify:
                try:
                    logger.debug(f"Validating claim: {claim_to_verify}")
                    prompt = await self._create_validation_prompt(claim_to_verify, document)
                    validation = await self.llm_manager.get_string_response(prompt=prompt, model="bespoke-minicheck")
                    logger.debug(f"Validation result for claim: {claim_to_verify} - {validation}")
                    
                    if validation == 'Yes':
                        if valid_count == 0:
                            validations.append((claim_to_verify, validation))
                        valid_count += 1
                except Exception as e:
                    logger.error(f"Error validating claim: {str(e)}")
                    validations.append((claim_to_verify, "Error"))

            # Calculate validation rate
            validation_rate = (valid_count / len(claims_to_verify)) * 100 if claims_to_verify else 0

            # Yield results if at least one validation passes
            if valid_count > 0:
                for claim, validation in validations:
                    validated_item = {**item, 'claim': claim, 'validation': validation, 'validation_rate': validation_rate}
                    yield validated_item