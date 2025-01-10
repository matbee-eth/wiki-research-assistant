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
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
        logger.debug(f"Formatting message: {message[:100]}...")
        if container:
            logger.debug(f"Using provided container for message display")
            container.markdown(message)
        else:
            logger.debug(f"Using default st.markdown for message display")
            st.markdown(message)

    def add_message(self, message: str, container: Any = None):
        """Add a message to the progress log."""
        logger.info(f"Adding message to progress log: {message[:100]}...")
        formatted_message = self._format_message(message, container)
        st.session_state.messages.append(formatted_message)
        logger.debug("Message added to session state")
        self._update_progress_log()

    def _update_progress_log(self):
        """Update the progress log display with enhanced categorization and styling."""
        logger.debug(f"Updating progress log with {len(st.session_state.messages)} messages")
        if not st.session_state.messages:
            logger.debug("No messages to display in progress log")
            return
            
        # Create a formatted log with emojis and styling for different message types
        log_entries = []
        for msg in st.session_state.messages:
            entry_class = "log-entry"
            
            # Categorize and style different types of log messages
            if "📋 Research Plan:" in msg:
                entry_class += " log-section plan-section"
            elif "🔍 Analysis:" in msg:
                entry_class += " log-section analysis-section"
            elif "📊 Analysis Results:" in msg:
                entry_class += " log-section analysis-results"
            elif "🔄 Query Processing:" in msg:
                entry_class += " log-section query-section"
            elif "🚀 Starting Search:" in msg:
                entry_class += " log-section search-section"
            elif "⚡ Progress Update:" in msg:
                entry_class += " progress-update"
            elif "📈 Score:" in msg or "relevance score:" in msg.lower():
                entry_class += " score-entry"
            elif "⚠️" in msg:
                entry_class += " warning-entry"
            elif "✅" in msg:
                entry_class += " success-entry"
            
            log_entries.append(f'<div class="{entry_class}">{msg}</div>')
        
        # Enhanced CSS styling for different log categories
        css = """
        <style>
            .progress-log {
                max-height: 400px;
                overflow-y: auto;
                padding: 10px;
                background: rgba(0,0,0,0.05);
                border-radius: 5px;
            }
            .log-entry {
                margin: 5px 0;
                padding: 5px;
                border-radius: 3px;
            }
            .log-section {
                font-weight: bold;
                padding: 8px;
                margin: 10px 0;
                border-left: 3px solid #7E57C2;
            }
            .analysis-section {
                border-left-color: #00BFA5;
            }
            .analysis-results {
                border-left-color: #00B8D4;
                background: rgba(0,184,212,0.1);
            }
            .score-entry {
                color: #00BFA5;
            }
            .warning-entry {
                color: #FFA726;
            }
            .success-entry {
                color: #66BB6A;
            }
            .progress-update {
                font-style: italic;
                color: #9E9E9E;
            }
        </style>
        """
        
        log_html = f"""
        {css}
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

    async def stream_research_progress(self, search_generator):
        """Stream research progress and results."""
        # Initialize session state if needed
        if 'result_containers' not in st.session_state:
            st.session_state.result_containers = {}
        if 'displayed_results' not in st.session_state:
            st.session_state.displayed_results = set()

        try:
            # Create layout containers
            progress_log = st.container()
            results_container = st.container()
            
            # Create progress bar
            progress_bar = progress_log.progress(0.0)
            
            # Add header first
            progress_log.markdown("### 🔄 Progress Log")
            
            # Create ONE container with the messages inside
            progress_container = progress_log.container()
            progress_container.markdown("""
                <style>
                    .progress-container {
                        max-height: 300px;
                        overflow-y: auto;
                        padding: 10px;
                        border: 1px solid rgba(250, 250, 250, 0.1);
                        border-radius: 4px;
                        margin-top: 1em;
                    }
                </style>
            """, unsafe_allow_html=True)
            
            progress_placeholder = progress_container.empty()
            progress_messages = []
            
            # Create results area in a separate container
            results_area = results_container.container()
            results_area.markdown("### 📈 Results Timeline")
            timeline_placeholder = results_area.empty()
            results_area.markdown("### 📚 Detailed Results")
            
            async def update_progress(message):
                progress_messages.append(message)
                progress_placeholder.markdown(
                    '<div class="progress-container">' + 
                    '\n\n'.join(progress_messages) + 
                    '</div>', 
                    unsafe_allow_html=True
                )

            # Process search updates
            async def process_stream(stream: AsyncGenerator[Dict, None]):
                """Process a stream of results."""
                logger.info("Starting to process stream")
                try:
                    async for item in stream:
                        logger.debug(f"Received stream item of type: {item.get('type')}")
                        
                        if item.get('type') == 'result':
                            logger.info(f"Processing result: {item.get('data', {}).get('title', 'Unknown')}")
                            self._process_result(item)
                            
                        elif item.get('type') in ['progress', 'processing', 'thought']:
                            logger.debug(f"Processing status update: {item.get('status', '')}")
                            if 'stream' in item:
                                self.add_message(item['stream'])
                            if 'status' in item:
                                self.add_message(item['status'])
                                
                        elif item.get('type') == 'error':
                            logger.error(f"Received error in stream: {item.get('error', 'Unknown error')}")
                            self.add_message(f"⚠️ Error: {item.get('error')}")
                            
                        else:
                            logger.warning(f"Unknown stream item type: {item.get('type')}")
                            
                except Exception as e:
                    logger.error(f"Error processing stream: {str(e)}", exc_info=True)
                    self.add_message(f"⚠️ Error processing results: {str(e)}")

            # Process search updates
            async for update in search_generator:
                try:
                
                    logger.debug(f"Received update: {update}")
                    
                    # Handle stream messages (main progress updates)
                    
                    if 'thought' in update:
                        await update_progress(update['thought'])
                    
                    elif 'stream' in update:
                        await update_progress(update['stream'])
                    
                    # Handle progress bar updates
                    if 'progress' in update:
                        progress_bar.progress(update['progress'])
                        
                    # Handle search results
                    if 'result' in update:
                        result = update['result']
                        logger.debug(f"Processing result: {result}")
                        
                        if isinstance(result, dict) and 'url' in result:
                            if result['url'] not in st.session_state.displayed_results:
                                st.session_state.displayed_results.add(result['url'])
                                st.session_state.results.append(result)
                                
                                # Update the timeline in its placeholder
                                timeline_fig = self.create_timeline_visualization(st.session_state.results)
                                timeline_placeholder.plotly_chart(timeline_fig, use_container_width=True)
                                
                                # Create expander for result
                                with results_area.expander(f"📄 {result.get('title', 'Untitled')}", expanded=False):
                                    st.markdown(f"**Source:** {result['url']}")
                                    if 'content' in result:
                                        st.markdown(result['content'])
                except Exception as e:
                    logger.error(f"Error processing update: {e}", exc_info=True)
                    logger.error(f"Update was: {update}")
        
            # Show export button if we have results
            if st.session_state.result_containers:
                self.show_export_button()
                        
        except Exception as e:
            error_msg = f"❌ Error during search: {str(e)}"
            st.error(error_msg)
            progress_placeholder.write(error_msg)

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
                x=[nodes[edge['source']]['x'], nodes[edge['source']]['x']],
                y=[nodes[edge['source']]['y'], nodes[edge['source']]['y']],
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

    def create_timeline_visualization(self, results: List[Dict]) -> go.Figure:
        """Create a timeline visualization of search results."""
        # Sort results by date if available, otherwise use order of discovery
        sorted_results = sorted(results, key=lambda x: x.get('date', ''))
        
        # Prepare data for timeline
        titles = [r.get('title', 'Untitled') for r in sorted_results]
        texts = [f"{r.get('title', 'Untitled')}<br>Score: {r.get('score', 0):.2f}" for r in sorted_results]
        
        # Create timeline figure
        fig = go.Figure()
        
        # Add timeline events
        fig.add_trace(go.Scatter(
            x=list(range(len(sorted_results))),
            y=[1] * len(sorted_results),
            mode='markers+text',
            marker=dict(
                size=20,
                color=self.theme['accent1'],
                line=dict(color=self.theme['accent2'], width=2)
            ),
            text=titles,
            textposition="top center",
            hovertext=texts,
            hoverinfo='text'
        ))
        
        # Update layout
        fig.update_layout(
            showlegend=False,
            plot_bgcolor=self.theme['background'],
            paper_bgcolor=self.theme['background'],
            font=dict(color=self.theme['text']),
            height=200,
            margin=dict(l=20, r=20, t=50, b=20),
            yaxis=dict(
                showgrid=False,
                showticklabels=False,
                range=[0.5, 1.5]
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                title='Results Timeline'
            )
        )
        
        return fig

    def _display_results(self, container: Any = None):
        """Display search results."""
        if container is None:
            container = self.results_container

        results = self.get_results()
        if not results:
            container.warning("No results to display.")
            return

        # Add timeline visualization
        container.markdown("### 📈 Results Timeline")
        timeline_fig = self.create_timeline_visualization(results)
        container.plotly_chart(timeline_fig, use_container_width=True)

        # Display individual results
        container.markdown("### 📚 Detailed Results")
        for result in results:
            title = result['title']
            score = float(result['score'])
            
            with container.expander(f"{title} (Score: {score:.2f})"):
                if 'content' in result:
                    container.markdown(result['content'])
                if 'literature_review' in result and result['literature_review']:
                    container.markdown("\n### Literature Review")
                    container.markdown(result['literature_review'])
                if 'url' in result:
                    container.markdown(f"\nSource: [{result['url']}]({result['url']})")

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
