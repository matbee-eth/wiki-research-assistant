# app.py

import streamlit as st
import asyncio
import logging
from search_engine import SearchEngine
from stream_interface import StreamInterface
from dotenv import load_dotenv
import os
from typing import List
import pandas as pd
from pathlib import Path
import base64

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('research_assistant.log'),
        logging.StreamHandler()
    ]
)

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

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Research Assistant", layout="wide")
    
    # Title and description
    st.title("🔍 Semantic Research Assistant")
    st.markdown("Enter your research query below to search across Wikipedia and other sources.")
    
    # Initialize search engine and interface
    search_engine = SearchEngine()
    
    # Create main columns for layout
    col1, col2 = st.columns([5, 1])
    
    with col1:
        # Search input
        query = st.text_input(
            "Enter your query",
            placeholder="e.g., What are the major theories of consciousness?",
            key="search_input"
        )
        
        # Search parameters
        col_params1, col_params2 = st.columns(2)
        with col_params1:
            max_results = st.slider("Maximum Results", min_value=10, max_value=500, value=100, step=10)
            min_score = st.slider("Minimum Score", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
        with col_params2:
            max_variations = st.slider("Query Variations", min_value=1, max_value=5, value=2, step=1)
            chunk_size = st.slider("Content Length", min_value=100, max_value=1000, value=300, step=100)
            
        search_clicked = st.button("🔍 Search", key="search_button")
    
    # Show export button in col2 if we have results
    with col2:
        st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
        if search_engine.get_all_results():
            st.download_button(
                label="📥 Export Results",
                data=search_engine.get_export_data(),
                file_name="research_results.md",
                mime="text/markdown",
                key="export_button"
            )
    
    # Handle search if button clicked
    if search_clicked and query:
        # Run search with parameters
        search_generator = search_engine.search(
            query=query,
            min_score=min_score,
            max_variations=max_variations,
            chunk_size=chunk_size
        )
        
        # Create interface and run search
        interface = StreamInterface()
        asyncio.run(interface.stream_research_progress(search_generator))

if __name__ == "__main__":
    main()
