import json
import logging
import aiohttp
import re
from typing import Dict, List, Any, Optional
from utils import fetch_gpt_response, fetch_gpt_response_parallel

class LLMManager:
    def __init__(self):
        """Initialize LLM manager."""
        self.session = None
        self.logger = logging.getLogger(__name__)

    async def __aenter__(self):
        """Initialize aiohttp session."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close aiohttp session."""
        try:
            if self.session and not self.session.closed:
                self.logger.debug("Closing LLM manager session...")
                await self.session.close()
                self.session = None
                self.logger.debug("Session closed")
        except Exception as e:
            self.logger.error(f"Error closing LLM manager session: {str(e)}", exc_info=True)

    async def get_response(self, prompt: str, max_tokens: int = None, model: str = "phi4", temperature: float = 0.9, stream: bool = False) -> str:
        """Get response from LLM."""
        try:
            if not self.session:
                raise RuntimeError("LLMManager must be used as an async context manager")
            return await fetch_gpt_response(self.session, prompt, max_tokens, model, temperature, stream)
        except Exception as e:
            self.logger.error(f"Error getting LLM response: {str(e)}")
            return None

    async def get_json_response(self, prompt: str, max_tokens: int = None) -> Dict:
        """Get JSON response from LLM."""
        try:
            response = await self.get_response(prompt, max_tokens)
            if not response:
                return {}
                
            # Try to find JSON-like content between triple backticks if present
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
            if json_match:
                response = json_match.group(1)
                self.logger.debug(f"Extracted JSON from markdown: {response}")
            
            # Clean up common formatting issues
            response = response.strip()
            if response.startswith('```') and response.endswith('```'):
                response = response[3:-3].strip()
            
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                # If direct parsing fails, try to extract JSON from the text
                # Look for {...} or [...] patterns
                json_pattern = r'({[^}]*}|\[[^\]]*\])'
                matches = re.findall(json_pattern, response)
                if matches:
                    for potential_json in matches:
                        try:
                            return json.loads(potential_json)
                        except json.JSONDecodeError:
                            continue
                
                self.logger.warning(f"Could not parse JSON from response: {response}")
                return {}
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Error decoding JSON response: {str(e)}")
            return {}
        except Exception as e:
            self.logger.error(f"Error getting JSON response: {str(e)}")
            return {}

    async def get_string_response(self, prompt: str, max_tokens: int = None, model: str = "phi4", temperature: float = 0.3, stream: bool = False) -> str:
        """
        Get string response from LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            max_tokens: Maximum number of tokens in the response
            model: The model to use (default: phi4)
            temperature: Temperature parameter for generation (default: 0.9)
            stream: Whether to stream the response (default: False)
            
        Returns:
            String response from the LLM
        """
        try:
            return await self.get_response(prompt, max_tokens, model=model, temperature=temperature, stream=stream)
        except Exception as e:
            self.logger.error(f"Error getting string response: {str(e)}")
            return ""

    async def get_parallel_responses(self, prompts: List[str], max_tokens: int = None) -> List[str]:
        """Get parallel responses from LLM."""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            return await fetch_gpt_response_parallel(self.session, prompts, max_tokens)
        except Exception as e:
            self.logger.error(f"Error getting parallel responses: {str(e)}")
            return [""] * len(prompts)
