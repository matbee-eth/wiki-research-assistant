# app.py

from concurrent.futures import process
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
from chat_thread import ChatThreadManager

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

def create_pipeline(llm_manager: LLMManager, config: Dict[str, Any] = None) -> Pipeline:
    """
    Create and configure the search pipeline.
    
    Args:
        llm_manager: LLM manager instance for query processing
        config: Configuration parameters for pipeline steps
        
    Returns:
        Configured pipeline instance
    """
    # Create pipeline components
    query_processor = QueryProcessor(llm_manager)
    fact_checker = FactChecker(llm_manager)
    data_sources = DataSources()
    data_sources.initialize()  # Ensure embeddings are initialized
    
    # Create pipeline with initial config
    pipeline = Pipeline(config=config or {})

    # Add steps with array-based methods
    pipeline.add_map("analyze", query_processor.analyze_queries, execution_mode=ExecutionMode.IMMEDIATE)
    pipeline.add_map("decompose", query_processor.decompose_queries, execution_mode=ExecutionMode.IMMEDIATE)
    pipeline.add_map("generate_claims", fact_checker.generate_claims, execution_mode=ExecutionMode.IMMEDIATE)

    # pipeline.add_map("enrich", query_processor.enrich_queries)
    
    pipeline.add_map("search", data_sources.stream_search_wikipedia, execution_mode=ExecutionMode.IMMEDIATE)
    pipeline.add_map("validate_claims", fact_checker.validate_claims, execution_mode=ExecutionMode.IMMEDIATE)
    
    return pipeline

class SearchRequestProcessor:
    def __init__(self):
        """Initialize the search request processor."""
        self.llm_manager = None
        self.pipeline = None
        self.chat_manager = None
    
    async def __aenter__(self):
        """Initialize async resources."""
        self.llm_manager = LLMManager()
        await self.llm_manager.__aenter__()
        self.chat_manager = st.session_state.get('chat_manager')
        if not self.chat_manager:
            self.chat_manager = ChatThreadManager()
            st.session_state.chat_manager = self.chat_manager
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
                st.session_state.processing = False
                return
                
            # Extract configuration parameters
            config = {
                'min_score': request.get('min_score', 0.85),
                'max_results': request.get('max_results', 10),
            }
            
            # Create pipeline with config only when processing a request
            self.pipeline = create_pipeline(self.llm_manager, config)
            self.pipeline.append({"query": query})
            
            # Get current chat thread
            thread = self.chat_manager.get_thread("default")
            
            # Add user message to thread
            thread.add_message(
                role="user",
                content=request.get("query", ""),
                metadata={"type": "search_request"}
            )
            
            async for processed_result in self.pipeline.process_queue():
                logger.info(f"Processed result: {processed_result} - {processed_result.get('step')}")
                
                # Store pipeline results in thread
                if processed_result.get("final_result"):
                    thread.add_message(
                        role="assistant",
                        content=processed_result.get("response", ""),
                        pipeline_results=processed_result,
                        metadata={"type": "pipeline_result"}
                    )
                yield processed_result
            
            # Set processing to False after pipeline finishes
            st.session_state.processing = False
                
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            thread.add_message(
                role="system",
                content=f"Error processing request: {str(e)}",
                metadata={"type": "error"}
            )
            yield {
                'type': 'error',
                'message': f"Error processing request: {str(e)}"
            }
            st.session_state.processing = False

async def main():
    st.set_page_config(page_title="Research Assistant", layout="wide")
    st.title("Research Assistant")
    
    # Initialize session state
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'messages' not in st.session_state:
        st.session_state.messages = {}
    
    # Sidebar configuration
    st.sidebar.title("Settings")
    st.sidebar.header("Pipeline Settings")
    min_score = st.sidebar.slider("Minimum Score", 0.0, 1.0, 0.72, 0.01)
    
    # <SEARCH>
    query = st.text_area("Enter your research query:", height=100)
    
    # <SEARCH STATUS> and <SEARCH RESULTS>
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.session_state.processing:
            st.markdown("""
                <div style="margin-top: 1rem;">
                    <div class="spinner"></div>
                    <span style="margin-left: 8px;">Researching...</span>
                </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if not st.session_state.processing:
            if st.button("Search", type="primary", use_container_width=True):
                st.session_state.processing = True
    
    # Show interface if processing
    if st.session_state.processing and query:
        try:
            # Create interface for streaming updates
            interface = StreamInterface()
            
            # Create search request with min_score
            request = {
                'query': query,
                'min_score': min_score  # Add min_score to request
            }
            
            # Run search with progress updates
            async with SearchRequestProcessor() as processor:
                await interface.stream_research_progress(
                    processor.process_search_request(request)
                )
            
            # Reset processing state when done
            st.session_state.processing = False
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.session_state.processing = False
            
    elif st.session_state.processing and not query:
        st.warning("Please enter a query to start research.")
        st.session_state.processing = False

async def run_tests() -> bool:
    """Run tests for the pipeline."""
    
    query = "mulsim attacks in london england"
    item_results = []
    
    # Initialize the query generator pipeline
    query_generator_pipeline = Pipeline()
    
    # Test LLM Manager initialization
    test_llm = LLMManager()
    # logger.info(" LLM Manager initialization")

    async with test_llm as llm:
        # Test QueryProcessor
        query_processor = QueryProcessor(llm)
        # logger.info(" QueryProcessor initialization")

        # Test DataSources
        test_sources = DataSources()
        # logger.info(" DataSources initialization")
        
        test_sources.initialize()  # Ensure embeddings are initialized

        # Test FactChecker
        fact_checker = FactChecker(llm)
        # logger.info(" FactChecker initialization")
        
        # Test Pipeline creation
        # Implement Pydantic in "Pipeline" class to validate output types
        # logger.info(" Pipeline creation")
        query_generator_pipeline.add_map("analyze", query_processor.analyze_queries, execution_mode=ExecutionMode.IMMEDIATE)
        # query_generator_pipeline.add_map("decompose", query_processor.decompose_queries)
        # query_generator_pipeline.add_map("enrich", query_processor.enrich_queries)

        # wikipedia_query_pipeline = Pipeline()
        query_generator_pipeline.add_map("search", test_sources.stream_search_wikipedia, execution_mode=ExecutionMode.IMMEDIATE)
        # query_generator_pipeline.add_map("generate_claims", fact_checker.generate_claims, execution_mode=ExecutionMode.ALL)
        query_generator_pipeline.add_filter("validate_claims", fact_checker.validate_claims, execution_mode=ExecutionMode.IMMEDIATE)
        # query_generator_pipeline.add_pipeline("wikipedia", wikipedia_query_pipeline)
        
        query_generator_pipeline.append({"query": query})
        # Process the queue
        async for processed_result in query_generator_pipeline.process_queue():
            # if (processed_result.get('is_final') == True):
            try:
                item_results.append({ "summary": processed_result.get('data', [["t"]])[0]["summary"], "step": processed_result.get('step') })
            except Exception as e:
                item_results.append({ "summary": processed_result.get('data'), "step": processed_result.get('step') })
                logger.error(f"Error processing result: {e}")

        # logger.info("All tests passed successfully!")
        logger.info(f"Item results: {item_results}")
        return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    args = parser.parse_args()

    if args.test:
        asyncio.run(run_tests())
    else:
        asyncio.run(main())
