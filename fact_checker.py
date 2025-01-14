from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class FactChecker:
    """A processor for validating claims against documents using bespoke-minicheck model."""
    
    def __init__(self, llm_manager):
        """
        Initialize the fact checker with an LLM manager for validation.
        
        Args:
            llm_manager: The LLM manager instance to use for validation
        """
        self.llm_manager = llm_manager

    async def validate_claim(self, document: str, claim: str) -> Dict[str, Any]:
        """
        Validate if a claim is supported by the given document using bespoke-minicheck model.
        
        Args:
            document (str): The source document text
            claim (str): The claim to validate against the document
            
        Returns:
            Dict[str, Any]: Result containing validation status and explanation
        """
        try:
            prompt = f"""Document: {document}
Claim: {claim}

Based on the document above, is the claim accurate and supported by the document content?
Respond with either 'Yes' or 'No', followed by a brief explanation.

Format your response as:
Verdict: [Yes/No]
Explanation: [Your explanation]"""

            response = await self.llm_manager.get_string_response(
                prompt=prompt,
                model="bespoke-minicheck"  # Always use bespoke-minicheck for fact checking
            )
            
            if not response:
                return {
                    "is_valid": False,
                    "explanation": "Failed to validate claim - no response from validation model",
                    "error": "No response from model"
                }
                
            # Parse response
            lines = response.strip().split('\n')
            verdict = lines[0].replace('Verdict:', '').strip().lower() == 'yes'
            explanation = ' '.join(lines[1:]).replace('Explanation:', '').strip()
            
            return {
                "is_valid": verdict,
                "explanation": explanation,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error validating claim: {str(e)}")
            return {
                "is_valid": False,
                "explanation": "Error occurred during validation",
                "error": str(e)
            }
