import streamlit as st
import logging
from typing import Dict, List, TYPE_CHECKING, AsyncGenerator, Any, Optional
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
    """Interface for streaming research results."""
    
    def __init__(
        self, 
        progress_placeholder: Optional[st.empty] = None,
        results_container: Optional[st.container] = None
    ):
        """Initialize the interface."""
        # Create status columns
        cols = st.columns(3)
        self.step_col = cols[0].empty()
        self.count_col = cols[1].empty()
        self.complete_col = cols[2].empty()
        
        # Create progress bar
        self.progress_bar = st.empty()
        
        # Create results container
        self.results_container = st.container()
        
        # Initialize state
        self._progress = 0.0
        self._results = {}
        self._result_containers = {}
        self._total_steps = 0
        self._current_step_index = 0
        self._current_step_progress = 0.0
        self._current_step = None
        
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
            elif "❌" in msg:
                entry_class += " error-entry"
            elif "🔍 Checking" in msg:
                entry_class += " fact-check-entry"
            
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
            .error-entry {
                color: #FF5252;
            }
            .fact-check-entry {
                color: #7E57C2;
                font-style: italic;
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

    def _update_status(self, step: str = "", count: int = 0, complete: bool = False):
        """Update the status display."""
        self.step_col.markdown(f"**Step:** {step}")
        
        # For validate_claim step, show Valid/Invalid counts alongside total results
        if step == 'validate_claim':
            # Only count definitively valid/invalid results
            valid_count = len([r for r in self._results.values() 
                             if r.get('validation', {}).get('is_valid', False)])
            invalid_count = len([r for r in self._results.values() 
                               if r.get('validation', {}).get('is_valid') is False])  # Explicitly False
            
            self.count_col.markdown(
                f"**Results:** {count}\n"
                f"**Valid:** {valid_count}\n"
                f"**Invalid:** {invalid_count}"
            )
        else:
            self.count_col.markdown(f"**Results:** {count}")
            
        if complete:
            self.complete_col.markdown("✅ Search Complete!")
        else:
            self.complete_col.empty()
            
    def _update_progress(self, progress: float = 0):
        """Update just the progress bar."""
        if progress is not None:
            self._progress = min(progress, 0.99)
        self.progress_bar.progress(self._progress)

    async def stream_research_progress(self, pipeline_generator: AsyncGenerator[Dict[str, Any], None]):
        """
        Stream progress updates and results from the pipeline.
        
        Args:
            pipeline_generator: Async generator yielding pipeline updates
        """
        try:
            self._progress = 0.0
            self._results = {}  # Track results by ID
            self._result_containers = {}  # Track containers by ID
            self._current_step_index = 0
            self._current_step_progress = 0.0
            self._step_progress = {}  # Track progress for each step
            
            # Create results area
            results_area = self.results_container.container()
            results_area.markdown("### 📚 Search Results")
            
            async for update in pipeline_generator:
                update_type = update.get('type')
                logger.debug(f"Received pipeline update type: {update_type} from step: {update.get('step', 'unknown')}, {update.get('data', 'unknown')}")
                pipeline_data = update.get('data', {})
                step = update.get('step')
                if update_type == 'progress':
                    step = update.get('step', '')
                    total_steps = update.get('total_steps', 0)
                    sub_step = update.get('sub_step', '')
                    total = update.get('total', 0)
                    processed = update.get('processed', 0)
                    status = update.get('status', '')
                    
                    if total_steps > 0:
                        self._total_steps = total_steps
                    
                    if step:
                        # Track progress for this step
                        if step not in self._step_progress:
                            self._step_progress[step] = {
                                'processed': 0,
                                'total': total if total > 0 else 1,
                                'sub_step': sub_step
                            }
                        
                        step_info = self._step_progress[step]
                        if total > 0:
                            step_info['total'] = total
                        if processed >= 0:
                            step_info['processed'] = processed
                            
                        # Calculate step progress
                        self._current_step_progress = step_info['processed'] / step_info['total']
                        
                        # Calculate overall progress
                        if self._total_steps > 0:
                            total_progress = sum(
                                info['processed'] / info['total'] 
                                for info in self._step_progress.values()
                            ) / self._total_steps
                            
                            self._update_progress(total_progress)
                    
                elif update_type == 'result':
                    logger.debug(f"Processing result update - step: {step} - data: {pipeline_data}")
                    
                    # Handle validation updates
                    if isinstance(pipeline_data, dict) and 'validation' in pipeline_data:
                        validation = pipeline_data.get('validation')
                        logger.debug(f"Found validation update: {validation} {update}")
                        # Try to get result ID from the data
                        metadata = update.get('metadata', {})
                        result_id = (metadata.get('article_id') or 
                                    pipeline_data.get('url') or 
                                    pipeline_data.get('title') or 
                                    pipeline_data.get('id'))
                        
                        logger.debug(f"Found result ID for validation: {result_id}")
                        
                        if result_id and result_id in self._results:
                            logger.debug(f"Processing validation update for {result_id}: {validation}")
                            
                            # Update validation status and result data
                            self._results[result_id].update(pipeline_data)
                            
                            # Remove if explicitly invalid
                            if validation.get('is_valid') is False:
                                logger.debug(f"Removing invalid result {result_id}")
                                if result_id in self._result_containers:
                                    self._result_containers[result_id].empty()
                                    del self._result_containers[result_id]
                                del self._results[result_id]
                                logger.debug(f"Removed invalid result: {result_id}")
                            else:
                                # Update display for non-invalid results
                                if result_id in self._result_containers:
                                    self._display_result(
                                        {'data': self._results[result_id]},
                                        self._result_containers[result_id]
                                    )
                                    logger.debug(f"Updated display for result: {result_id}")
                        else:
                            logger.debug(f"No matching result found for validation update: {result_id}")
                    
                    # Handle non-validation updates
                    if pipeline_data:
                        # Process the pipeline data
                        if isinstance(pipeline_data, dict):
                            # Extract any results from pipeline data
                            results = []
                            if 'results' in pipeline_data:
                                results = pipeline_data['results']
                            elif 'original_data' in pipeline_data and isinstance(pipeline_data['original_data'], dict):
                                results = pipeline_data['original_data'].get('results', [])
                            elif pipeline_data.get('metadata') or pipeline_data.get('url') or pipeline_data.get('title'):
                                # This looks like a record itself
                                results = [pipeline_data]
                            
                            # Process each result that has an ID
                            for result in results:
                                if not isinstance(result, dict):
                                    continue
                                    
                                # Get result ID
                                metadata = result.get('metadata', {})
                                result_id = metadata.get('article_id') or result.get('url') or result.get('title') or result.get('id')
                                
                                if not result_id:
                                    continue
                                
                                # Create or update result
                                if result_id not in self._results:
                                    self._results[result_id] = result
                                    with results_area:
                                        self._result_containers[result_id] = st.empty()
                                    logger.debug(f"Added new result: {result_id}")
                                else:
                                    # Don't update if already marked invalid
                                    existing_validation = self._results[result_id].get('validation', {})
                                    if existing_validation.get('is_valid') is False:
                                        logger.debug(f"Skipping update for invalid result: {result_id}")
                                        continue
                                        
                                    self._results[result_id].update(result)
                                    logger.debug(f"Updated existing result: {result_id}")
                                
                                # Display the result
                                if result_id in self._result_containers:
                                    self._display_result(
                                        {'data': self._results[result_id]},
                                        self._result_containers[result_id]
                                    )
                            
                            # Update progress and status
                            self._progress = min(self._progress + 0.05, 0.95)
                            self._update_progress(self._progress)
                            
                            # Only count results that aren't explicitly invalid
                            valid_count = len([r for r in self._results.values() 
                                             if r.get('validation', {}).get('is_valid') is not False])
                            self._update_status(step=update.get('step', ''), count=valid_count)
                        
                elif update_type == 'error':
                    error_msg = update.get('data', 'Unknown error')
                    st.error(f"❌ Error: {error_msg}")
                    logger.error(f"Pipeline error: {error_msg}")
                    
            # Complete
            valid_count = len([r for r in self._results.values() 
                             if r.get('validation', {}).get('is_valid') is not False])
            self._update_status(step="Complete", count=valid_count, complete=True)
            self._update_progress(1.0)
                
        except Exception as e:
            logger.error(f"Error streaming results: {str(e)}", exc_info=True)
            st.error(f"❌ Error: {str(e)}")
            
        finally:
            # Clear progress
            self.progress_bar.empty()

    def clear_results(self):
        """Clear the current results."""
        st.session_state.results = []
        
    def get_results(self):
        """Get current results, filtered and sorted."""
        # Filter out invalid results
        valid_results = [
            result for result in self._results.values()
            if not result.get('validation') or result.get('validation', {}).get('is_valid') is not False
        ]
        
        # Sort by score if available
        return sorted(
            valid_results,
            key=lambda x: float(x.get('score', 0)),
            reverse=True
        )

    def add_results(self, results):
        """Add results to session state."""
        if not hasattr(st.session_state, 'results'):
            st.session_state.results = []
        st.session_state.results.extend(results)


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

    def _display_result(self, result: Dict[str, Any], container: Optional[st.container] = None) -> None:
        """Display a single search result."""
        data = result.get('data', {})
        title = data.get('title', 'Untitled')
        url = data.get('url', '')
        score = data.get('score', 0)
        metadata = data.get('metadata', {})
        validation = data.get('validation', {})
        
        # Create markdown for result
        markdown = f"### [{title}]({url.replace(' ', '%20')})\n"
        if metadata:
            markdown += f"**ID:** {metadata.get('article_id', 'Unknown')}\n"
            
        # Add validation status if available
        if validation:
            is_valid = validation.get('is_valid', False)
            explanation = validation.get('explanation', '')
            status = "✅ Valid" if is_valid else "❌ Invalid"
            markdown += f"**Validation:** {status}\n"
            if explanation:
                markdown += f"**Explanation:** {explanation}\n"
        
        # Display the result
        if container:
            container.markdown(markdown)
        else:
            st.markdown(markdown)

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
