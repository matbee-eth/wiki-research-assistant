# utils.py

import logging
from typing import Callable
from functools import lru_cache, wraps
import time
import asyncio
import aiohttp
import json
from config import OPENAI_API_KEY, OPENAI_API_BASE
from datetime import datetime
import streamlit as st
import base64

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG,
    filename='research_assistant.log',
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Add console handler for development
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# Get root logger and add console handler
root_logger = logging.getLogger()
root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)

def cache_results(func: Callable) -> Callable:
    """
    Cache function results to avoid redundant API calls.
    """
    cache = {}
    
    @wraps(func)
    async def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = await func(*args, **kwargs)
        return cache[key]
    
    return wrapper

def retry_on_error(func: Callable) -> Callable:
    """
    Retry function on error with exponential backoff.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                retry_count += 1
                if retry_count == max_retries:
                    logger.error(f"Failed after {max_retries} retries: {str(e)}")
                    raise
                wait_time = 2 ** retry_count
                logger.warning(f"Attempt {retry_count} failed. Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
    
    return wrapper

def log_function_call(func: Callable) -> Callable:
    """
    Decorator to log function entry, exit, and execution time.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        func_name = func.__name__
        logger.debug(f"Entering {func_name} - Args: {args}, Kwargs: {kwargs}")
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"Exiting {func_name} - Execution time: {execution_time:.2f}s")
            return result
        except Exception as e:
            logger.error(f"Error in {func_name}: {str(e)}", exc_info=True)
            raise
    return wrapper

# Simple in-memory cache using LRU
@cache_results
async def get_cached_results(query: str):
    return None  # Placeholder for caching logic

# Rate limiting semaphore for OpenAI API calls
OPENAI_SEMAPHORE = asyncio.Semaphore(5)  # Limit concurrent API calls

@log_function_call
@retry_on_error
async def fetch_gpt_response(session, prompt, max_tokens, model="phi4", temperature=0.3, stream=False):
    """
    Fetch response from GPT API.
    
    Args:
        session: aiohttp ClientSession
        prompt: Input prompt
        model: Model to use
        temperature: Temperature parameter
        max_tokens: Maximum tokens to generate
        stream: Whether to stream the response
        
    Returns:
        str: Generated response
    """
    url = f"{OPENAI_API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if stream:
        payload["stream"] = stream
    
    logger.debug(f"Making API request to {url}")
    logger.debug(f"Request payload: {json.dumps(payload, indent=2)}")
    
    try:
        async with session.post(url, headers=headers, json=payload) as response:
            logger.debug(f"API response status: {response.status}")
            
            if response.status == 200:
                if not stream:
                    data = await response.json()
                    logger.debug(f"API response data: {json.dumps(data, indent=2)}")
                    content = data['choices'][0]['message']['content'].strip()
                    logger.debug(f"Extracted content: {content}")
                    return content
                    
                async for line in response.content:
                    if line:
                        try:
                            json_response = json.loads(line.decode('utf-8').strip().strip('data: '))
                            logger.debug(f"Stream response chunk: {json.dumps(json_response, indent=2)}")
                            if json_response.get("choices"):
                                content = json_response["choices"][0].get("delta", {}).get("content", "")
                                if content:
                                    logger.debug(f"Stream content chunk: {content}")
                                    return content
                        except json.JSONDecodeError:
                            logger.warning("Failed to parse stream response chunk")
                            continue
            elif response.status == 429:
                logger.warning("Rate limit exceeded. Retrying...")
                await asyncio.sleep(1)
                return None
            else:
                error_text = await response.text()
                logger.error(f"GPT API Error {response.status}: {error_text}")
                return None
    except Exception as e:
        logger.error(f"Error in fetch_gpt_response: {str(e)}", exc_info=True)
        return None

@log_function_call
@retry_on_error
async def fetch_gpt_response_parallel(session, prompts, max_tokens=None, model="phi4", temperature=0.3):
    """
    Fetch responses from GPT API in parallel with rate limiting.
    
    Args:
        session: aiohttp ClientSession
        prompts: List of input prompts
        model: Model to use
        temperature: Temperature parameter
        max_tokens: Maximum tokens to generate
        
    Returns:
        List[str]: Generated responses
    """
    async def process_single_prompt(prompt):
        async with OPENAI_SEMAPHORE:
            return await fetch_gpt_response(session, prompt, max_tokens, model, temperature)
            
    tasks = [process_single_prompt(prompt) for prompt in prompts]
    try:
        responses = await asyncio.gather(*tasks)
        return responses
    except Exception as e:
        logger.error(f"Error in parallel GPT processing: {str(e)}", exc_info=True)
        raise

def export_results(results):
    """Export search results to a markdown file."""
    if not results:
        return
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"research_results_{timestamp}.md"
    
    content = ["# Research Results\n"]
    content.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    for i, result in enumerate(results, 1):
        content.append(f"## {i}. {result['title']}")
        content.append(f"Score: {result['score']:.2f}\n")
        
        if 'content' in result:
            content.append(result['content'])
        
        if 'url' in result:
            content.append(f"\nSource: [{result['url']}]({result['url']})")
        
        content.append("\n---\n")
    
    markdown_content = "\n".join(content)
    
    # Use base64 encoding for the download link
    b64 = base64.b64encode(markdown_content.encode()).decode()
    href = f'<a href="data:text/markdown;base64,{b64}" download="{filename}">📥 Download Results</a>'
    st.markdown(href, unsafe_allow_html=True)
