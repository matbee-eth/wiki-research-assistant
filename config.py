# config.py

import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# OpenAI API configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")

OPENAI_API_BASE = os.getenv('OPENAI_API_BASE', 'https://ollama2.matbee.com/v1')
WIKIPEDIA_API_URL = 'https://en.wikipedia.org/w/api.php'
WIKIDATA_API_URL = 'https://www.wikidata.org/w/api.php'
NEWS_API_KEY = os.getenv('NEWS_API_KEY')  # For real-time news integration
