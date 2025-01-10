import streamlit as st
import logging
from typing import Dict, List, TYPE_CHECKING, AsyncGenerator, Any
import plotly.graph_objects as go
import asyncio
import datetime
import base64

if TYPE_CHECKING:
    from search_engine import SearchEngine

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class StreamInterface:
    def __init__(self):
        """Initialize the interface."""
        if 'results' not in st.session_state:
            st.session_state.results = []
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'results_count' not in st.session_state:
            st.session_state.results_count = 0
        if 'displayed_results' not in st.session_state:
            st.session_state.displayed_results = set()
            
        # Add styles
        st.markdown("""
            <style>
                div[data-testid="stMarkdownContainer"] > .progress-log {
                    height: 300px;
                    overflow-y: auto;
                    border: 1px solid rgba(250, 250, 250, 0.2);
                    padding: 10px;
                    border-radius: 5px;
                    background-color: rgba(17, 17, 17, 0.7);
                    margin-bottom: 20px;
                    color: #fafafa;
                    font-family: "Source Sans Pro", sans-serif;
                }
                .progress-log::-webkit-scrollbar {
                    width: 8px;
                }
                .progress-log::-webkit-scrollbar-track {
                    background: rgba(250, 250, 250, 0.1);
                    border-radius: 4px;
                }
                .progress-log::-webkit-scrollbar-thumb {
                    background: rgba(250, 250, 250, 0.3);
                    border-radius: 4px;
                }
                .progress-log::-webkit-scrollbar-thumb:hover {
                    background: rgba(250, 250, 250, 0.4);
                }
                .log-entry {
                    margin: 4px 0;
                    line-height: 1.5;
                    color: #fafafa;
                }
                .log-section {
                    margin-top: 12px;
                    padding: 8px;
                    background-color: rgba(250, 250, 250, 0.05);
                    border-radius: 4px;
                    border-left: 3px solid #00ff88;
                }
                .log-entry code {
                    background-color: rgba(250, 250, 250, 0.1);
                    padding: 2px 4px;
                    border-radius: 3px;
                    font-family: monospace;
                }
                .export-link {
                    text-decoration: none;
                    padding: 0.5rem 1rem;
                    border-radius: 0.3rem;
                    background-color: #0066cc;
                    color: white !important;
                    font-weight: 500;
                    display: inline-block;
                    text-align: center;
                    width: 100%;
                    margin-top: 5px;
                }
                .export-link:hover {
                    background-color: #0052a3;
                    color: white !important;
                    text-decoration: none;
                }
            </style>
        """, unsafe_allow_html=True)
        
        # Create empty containers
        self.progress_container = st.container()
        self.results_container = st.container()
        
        self.min_score = 0.8  # Default min_score
        self.theme = {
            'background': '#1E1E1E',
            'text': '#E0E0E0',
            'accent1': '#00FF88',
            'accent2': '#FF4081',
            'accent3': '#7E57C2'
        }

    def set_container(self, container):
        """Set the main Streamlit container."""
        self.container = container

    def _format_message(self, message: str, container: Any = None):
        """Format and display a message."""
        if container:
            container.markdown(message)
        else:
            st.markdown(message)

    def add_message(self, message: str, container: Any = None):
        """Add a message to the progress log."""
        formatted_message = self._format_message(message, container)
        st.session_state.messages.append(formatted_message)
        self._update_progress_log()

    def _update_progress_log(self):
        """Update the progress log display."""
        if not st.session_state.messages:
            return
            
        # Create a formatted log with emojis for different message types
        log_entries = []
        for msg in st.session_state.messages:
            # Add spacing between sections
            if any(section in msg for section in ["📋 Research Plan:", "🔍 Analysis Results:", "🔄 Query Processing:", "🚀 Starting Search:"]):
                # Format section headers with special styling
                log_entries.append(f'<div class="log-entry log-section">{msg}</div>')
            else:
                # Format code snippets and queries
                log_entries.append(f'<div class="log-entry">{msg}</div>')
        
        log_html = f"""
        <div class="progress-log" id="progress-log">
            {''.join(log_entries)}
        </div>
        <script>
            window.setTimeout(function() {{
                var log = document.getElementById('progress-log');
                if (log) {{
                    log.scrollTop = log.scrollHeight;
                }}
            }}, 100);
        </script>
        """
        
        self.progress_log.markdown(log_html, unsafe_allow_html=True)

    async def stream_research_progress(
        self, 
        search_generator: AsyncGenerator,
        progress_bar: Any,
        progress_log: Any,
        results_container: Any,
        search_engine: Any
    ):
        """Stream research progress updates."""
        try:
            # Create results area once
            results_area = results_container.container()
            results_area.markdown("### 📊 Search Results")
            
            # Create progress log area
            progress_area = progress_log.container()
            progress_messages = []
            
            # Create a container for each result and track displayed sources
            result_containers = {}
            displayed_sources = set()
            
            # Process search updates
            async for update in search_generator:
                if isinstance(update, dict):
                    if 'stream' in update:
                        # Add message to list and update display
                        progress_messages.append(update['stream'])
                        with progress_area:
                            st.empty()  # Clear previous
                            for msg in progress_messages:
                                st.markdown(msg)
                    
                    if 'progress' in update:
                        progress_bar.progress(update['progress'])
                    
                    if 'result' in update:
                        # Get all current results
                        all_results = search_engine.get_all_results()
                        
                        # Update or create containers for each result
                        with results_area:
                            for result in all_results:
                                title = result['title']
                                score = float(result['score'])
                                source_url = result.get('url', '')
                                
                                # Skip if we've already displayed this source
                                if source_url in displayed_sources:
                                    continue
                                    
                                # Create container if not exists
                                if title not in result_containers:
                                    result_containers[title] = results_area.expander(
                                        f"{title} (Score: {score:.2f})"
                                    )
                                
                                # Update content
                                with result_containers[title]:
                                    if 'text' in result:
                                        st.markdown("### Summary")
                                        st.markdown(result['text'])
                                        
                                    if 'analysis' in result:
                                        st.markdown("\n### Analysis")
                                        st.markdown(result['analysis'])
                                        
                                    if 'literature_review' in result and result['literature_review']:
                                        st.markdown("\n### Literature Review")
                                        st.markdown(result['literature_review'])
                                        
                                    if source_url:
                                        st.markdown(f"\nSource: [{source_url}]({source_url})")
                                        displayed_sources.add(source_url)
                    
                    elif 'wiki_summary' in update:
                        # Handle wiki summary update
                        with progress_area:
                            st.markdown("### 📖 Wiki Summary")
                            st.markdown(update['data'])
                            
                    elif 'analysis' in update:
                        # Handle analysis update
                        with progress_area:
                            st.markdown("### 🔍 Analysis")
                            st.markdown(update['data'])
                            
                    elif 'literature_review' in update:
                        # Handle literature review update
                        with progress_area:
                            st.markdown("### 📚 Literature Review")
                            st.markdown(update['data'])
        
            # Show export button if we have results
            if search_engine.get_all_results():
                self.show_export_button()
                        
        except Exception as e:
            error_msg = f"❌ Error during search: {str(e)}"
            st.error(error_msg)
            progress_messages.append(error_msg)
            with progress_area:
                st.empty()
                for msg in progress_messages:
                    st.markdown(msg)

    def clear_results(self):
        """Clear the current results."""
        st.session_state.results = []
        
    def get_results(self):
        """Get current results, filtered and sorted."""
        if not hasattr(st.session_state, 'results') or not st.session_state.results:
            return []
            
        # Filter and sort results, ensuring uniqueness by title
        seen_titles = set()
        valid_results = []
        
        for item in st.session_state.results:
            if isinstance(item, dict) and 'title' in item and 'score' in item:
                title = item['title']
                if title not in seen_titles:
                    seen_titles.add(title)
                    valid_results.append(item)
        
        # Sort by score and return top results
        return sorted(valid_results, key=lambda x: float(x['score']), reverse=True)

    def add_results(self, results):
        """Add results to session state."""
        if not hasattr(st.session_state, 'results'):
            st.session_state.results = []
        st.session_state.results.extend(results)

    async def handle_batch_search(self, search_engine, queries: List[str], min_score: float = 0.8):
        """Handle batch search queries with parallel processing and streaming updates."""
        try:
            # Initialize progress tracking
            progress = {query: {"status": "pending", "results": []} for query in queries}
            
            # Initialize progress display
            self.add_message("Processing queries in parallel...")
            
            queries_completed = 0
            total_queries = len(queries)
            
            # Process queries with parallel streaming updates
            async def process_query(query: str):
                nonlocal queries_completed
                try:
                    async for update in search_engine.search(query, min_score, self):
                        if 'stream' in update:
                            self.add_message(f"[{query}] {update['stream']}")
                        if 'data' in update:
                            progress[query]["results"] = update['data']
                            progress[query]["status"] = "completed"
                            queries_completed += 1
                            
                            # Update progress bar
                            current_progress = queries_completed / total_queries
                            self.add_message(f"Completed {queries_completed}/{total_queries} queries")
                            
                except Exception as e:
                    progress[query]["status"] = "failed"
                    progress[query]["error"] = str(e)
                    self.add_message(f"❌ Error processing query '{query}': {str(e)}")
            
            # Create tasks for all queries
            tasks = [process_query(query) for query in queries]
            await asyncio.gather(*tasks)
            
            # Combine results from all completed queries
            all_results = []
            for query, query_progress in progress.items():
                if query_progress["status"] == "completed":
                    all_results.extend(query_progress["results"])
            
            # Sort combined results by score
            st.session_state.results = sorted(all_results, key=lambda x: x.get('score', 0), reverse=True)
            
            # Display final results
            self._display_results()
            
        except Exception as e:
            logger.error(f"Error in batch search: {str(e)}", exc_info=True)
            self.add_message(f"❌ Error in batch search: {str(e)}")

    async def handle_search(self, query: str):
        """Handle search request and display results."""
        if not query:
            return
            
        # Show query header
        self.add_message(f"🔍 Query: {query}")
        
        try:
            async with SearchEngine(min_score=self.min_score) as engine:
                async for update in engine.search(query):
                    progress = update.get('progress', 0)
                    status = update.get('status', '')
                    stream = update.get('stream', '')
                    
                    # Update stats based on stream message
                    if 'stream' in update:
                        self.add_message(stream)
                    
                    # Display results if any
                    if 'results' in update:
                        st.session_state.results = update['results']
                        self._display_results()
                        
        except Exception as e:
            logger.error(f"Error in search: {str(e)}", exc_info=True)
            self.add_message(f"❌ Error in search: {str(e)}")

    def visualize_search_graph(self, nodes: List[Dict], edges: List[Dict]):
        """Create an interactive force-directed graph of search results"""
        fig = go.Figure()
        
        # Add nodes
        node_x = [node['x'] for node in nodes]
        node_y = [node['y'] for node in nodes]
        
        fig.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            marker=dict(
                size=20,
                color=[self.theme['accent1'] if node['type'] == 'query' else self.theme['accent2'] for node in nodes],
                line=dict(width=2, color=self.theme['accent3'])
            ),
            text=[node['label'] for node in nodes],
            hoverinfo='text',
            name='nodes'
        ))
        
        # Add edges
        for edge in edges:
            fig.add_trace(go.Scatter(
                x=[nodes[edge['source']]['x'], nodes[edge['target']]['x']],
                y=[nodes[edge['source']]['y'], nodes[edge['target']]['y']],
                mode='lines',
                line=dict(width=1, color=self.theme['text']),
                hoverinfo='none',
                showlegend=False
            ))
            
        fig.update_layout(
            plot_bgcolor=self.theme['background'],
            paper_bgcolor=self.theme['background'],
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            font=dict(color=self.theme['text'])
        )
        
        self.container.plotly_chart(fig, use_container_width=True)

    def get_export_data(self, results: List[Dict]) -> str:
        """Generate markdown export data from results."""
        if not results:
            return ""
            
        markdown = ["# Research Results\n"]
        
        for result in results:
            title = result.get('title', 'Untitled')
            content = result.get('content', '')
            url = result.get('url', '')
            score = float(result.get('score', 0))
            
            markdown.extend([
                f"\n## {title} (Score: {score:.2f})\n",
                f"{content}\n",
                f"Source: [{url}]({url})\n"
            ])
            
        return "\n".join(markdown)

    def export_results(self, format: str):
        """Export results in the specified format."""
        # This method is no longer needed as export functionality is now handled in _display_results
        pass

    def show_progress(self, progress_bar: Any, progress_log: Any):
        """Show the progress section."""
        with self.progress_container:
            st.markdown("### 🔄 Search Progress")
            # Create progress bar
            self.progress_bar = progress_bar
            # Create log container
            self.progress_log = progress_log

    def show_results_section(self, results_container: Any):
        """Show the results section."""
        with results_container:
            st.markdown("### 📊 Search Results")

    def _display_results(self, container: Any = None):
        """Display search results."""
        results = self.get_results()
        if not results:
            return
            
        target = container if container else st
        
        with target:
            st.markdown("### 📊 Search Results")
            
            # Display each result in its own expander
            for result in results:
                title = result['title']
                score = float(result['score'])
                
                with st.expander(f"{title} (Score: {score:.2f})"):
                    if 'content' in result:
                        st.markdown(result['content'])
                    if 'literature_review' in result and result['literature_review']:
                        st.markdown("\n### Literature Review")
                        st.markdown(result['literature_review'])
                    if 'url' in result:
                        st.markdown(f"\nSource: [{result['url']}]({result['url']})")

    def show_export_button(self):
        """Show the export button."""
        results = self.get_results()
        if not results:
            return
            
        # Create export content
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"research_results_{timestamp}.md"
        
        content = ["# Research Results\n"]
        content.append(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        for i, result in enumerate(results, 1):
            content.append(f"## {i}. {result['title']}")
            content.append(f"Score: {result['score']:.2f}\n")
            
            if 'content' in result:
                content.append(result['content'])
            
            if 'url' in result:
                content.append(f"\nSource: [{result['url']}]({result['url']})")
            
            content.append("\n---\n")
        
        markdown_content = "\n".join(content)
        b64 = base64.b64encode(markdown_content.encode()).decode()
        href = f'<a href="data:text/markdown;base64,{b64}" download="{filename}" class="export-link">📥 Export Results</a>'
        st.markdown(href, unsafe_allow_html=True)

    def has_results(self) -> bool:
        """Check if there are any results."""
        return bool(st.session_state.results)
