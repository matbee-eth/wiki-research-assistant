# app.py

import streamlit as st
import asyncio
import logging
from stream_interface import StreamInterface
from dotenv import load_dotenv
import os
from typing import List, Dict, Any, AsyncGenerator
import pandas as pd
from pathlib import Path
import base64
from pipeline import ExecutionMode, Pipeline, batch
from llm_manager import LLMManager
from pipelines.query_processor import QueryProcessor
from pipelines.fact_checker import FactChecker
from data_sources import DataSources

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('research_assistant.log'),
        logging.StreamHandler()
    ]
)

# Disable watchdog and inotify debug logs
logging.getLogger('watchdog').setLevel(logging.WARNING)
logging.getLogger('watchdog.observers.inotify_buffer').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

def get_download_link(file_path: str, label: str) -> str:
    """Generate a download link for a file."""
    with open(file_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    extension = Path(file_path).suffix[1:]  # Remove the dot
    mime_types = {
        'html': 'text/html',
        'pdf': 'application/pdf',
        'parquet': 'application/octet-stream',
        'md': 'text/markdown'
    }
    mime_type = mime_types.get(extension, 'application/octet-stream')
    href = f'data:{mime_type};base64,{b64}'
    return f'<a href="{href}" download="{Path(file_path).name}" target="_blank">{label}</a>'

def export_results(results: List[dict]):
    """Export search results to a CSV file."""
    df = pd.DataFrame(results)
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="search_results.csv" target="_blank">Download Results</a>'
    st.markdown(href, unsafe_allow_html=True)

def create_pipeline(llm_manager: LLMManager) -> Pipeline:
    """
    Create and configure the search pipeline.
    
    Args:
        llm_manager: LLM manager instance for query processing
        
    Returns:
        Configured pipeline instance
    """
    # Create pipeline components
    query_processor = QueryProcessor(llm_manager)
    fact_checker = FactChecker(llm_manager)
    data_sources = DataSources()
    data_sources.initialize()  # Ensure embeddings are initialized
    

    # Create pipeline
    pipeline = Pipeline()
    
    # Add steps with array-based methods
    pipeline.add_map("analyze", query_processor.analyze_queries)
    pipeline.add_map("decompose", query_processor.decompose_queries)
    pipeline.add_map("enrich", query_processor.enrich_queries)
    
    # Create an async wrapper for stream_search_wikipedia to handle arrays
    async def search_batch(items: List[Dict[str, Any]], config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        results = []
        for item in items:
            # Collect all results from the generator
            item_results = []
            async for result in data_sources.stream_search_wikipedia(item, config):
                item_results.extend(result.get('results', []))
            # Update the item with all collected results
            item['results'] = item_results
            results.append(item)
        return results
    
    pipeline.add_map("search", search_batch)
    pipeline.add_map("generate_claims", fact_checker.generate_claims)
    pipeline.add_filter("validate_claims", fact_checker.validate_claims)  # Changed to filter
    
    return pipeline

class SearchRequestProcessor:
    def __init__(self):
        """Initialize the search request processor."""
        self.llm_manager = None
        self.pipeline = None

    async def __aenter__(self):
        """Initialize async resources."""
        self.llm_manager = LLMManager()
        await self.llm_manager.__aenter__()
        self.pipeline = create_pipeline(self.llm_manager)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup async resources."""
        if self.llm_manager:
            await self.llm_manager.__aexit__(exc_type, exc_val, exc_tb)
            self.llm_manager = None

    async def process_search_request(self, request: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """Process a search request."""
        try:
            query = request.get('query')
            if not query:
                yield {
                    'type': 'error',
                    'message': 'No query provided'
                }
                return
                
            min_score = request.get('min_score', 0.7)
            logger.info(f"Processing search request: {query}")
            
            # Structure the initial data for the pipeline as a single-item array
            pipeline_data = [{
                'query': query,  # Original query string
                'config': {     # Configuration for all steps
                    'min_score': min_score,
                    'search': {
                        'min_score': min_score
                    }
                },
                'results': [],  # Will hold search results
                'metadata': {}  # Additional metadata
            }]
            
            # Use the pipeline
            async for result in self.pipeline.execute(pipeline_data):
                logger.info(f"Got pipeline result: {result.get('type')} - {result.get('step')}")
                logger.info(f"Got pipeline result data: {result}")
                # Extract the first item since we're only processing one query
                if result.get('data') and isinstance(result['data'], list) and len(result['data']) > 0:
                    result['data'] = result['data'][0]
                yield result
                
        except Exception as e:
            logger.error(f"Error processing search request: {str(e)}", exc_info=True)
            yield {
                'type': 'error',
                'message': f"Error processing request: {str(e)}"
            }

async def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Research Assistant", layout="wide")
    st.title("Research Assistant")
    
    # Sidebar configuration
    st.sidebar.title("Settings")
    st.sidebar.header("Pipeline Settings")
    st.sidebar.header("Advanced Settings")
    min_score = st.sidebar.slider("Minimum Score", 0.0, 1.0, 0.7,
        help="Minimum relevance score for search results")
    
    # <SEARCH>
    query = st.text_area("Enter your research query:", height=100)
    search_button = st.button("Search")
    
    # <SEARCH STATUS> and <SEARCH RESULTS>
    
    if search_button and query:
        if not query:
            st.error("Please enter a query")
            return
            
        progress_placeholder = st.empty()
        
        # Create interface for streaming updates
        interface = StreamInterface(
            progress_placeholder=progress_placeholder,
        )
        
        # Run search with progress updates
        async with SearchRequestProcessor() as processor:
            await interface.stream_research_progress(
                processor.process_search_request({
                    'query': query,
                    'min_score': min_score
                })
            )

async def run_tests() -> bool:
    """Run test suite."""
    logger.info("Running tests...")
    item_results = []
    config = {
        "limit": 1,
    }  # Add any necessary config here
    query = "Any laws that regulate who can own a firearm in canada"

    # Test LLM Manager initialization
    test_llm = LLMManager()
    logger.info(" LLM Manager initialization")

    async with test_llm as llm:
        # Test QueryProcessor
        query_processor = QueryProcessor(llm)
        logger.info(" QueryProcessor initialization")

        # Test DataSources
        test_sources = DataSources()
        logger.info(" DataSources initialization")
        
        test_sources.initialize()  # Ensure embeddings are initialized

        # Test FactChecker
        fact_checker = FactChecker(llm)
        logger.info(" FactChecker initialization")
        
        # Test Pipeline creation
        # Implement Pydantic in "Pipeline" class to validate output types
        logger.info(" Pipeline creation")
        query_generator_pipeline = Pipeline(initial_items=[query], config={"query": query})
        query_generator_pipeline.add_map("analyze", query_processor.analyze_queries)
        # query_generator_pipeline.add_map("decompose", query_processor.decompose_queries)
        # query_generator_pipeline.add_map("enrich", query_processor.enrich_queries)

        # wikipedia_query_pipeline = Pipeline()
        query_generator_pipeline.add_map("search", test_sources.stream_search_wikipedia, execution_mode=ExecutionMode.ALL)
        query_generator_pipeline.add_map("generate_claims", fact_checker.generate_claims, execution_mode=ExecutionMode.ALL)
        query_generator_pipeline.add_filter("validate_claims", fact_checker.validate_claims, execution_mode=ExecutionMode.ALL)
        # query_generator_pipeline.add_pipeline("wikipedia", wikipedia_query_pipeline)
        
        # Process the queue
        async for processed_result in query_generator_pipeline.process_queue():
            if (processed_result.get('is_final') == True):
                item_results.append(processed_result)

        logger.info("All tests passed successfully!")
        logger.info(f"Item results: {item_results}")
        return True

if __name__ == "__main__":
    # Check if we're running under Streamlit
    if not os.environ.get('STREAMLIT_SCRIPT_MODE'):
        logger.info("Note: For the full UI experience, run with 'streamlit run app.py'")
        logger.info("Running in test mode instead...")
        success = asyncio.run(run_tests())
        exit(0 if success else 1)
    else:
        asyncio.run(main())
