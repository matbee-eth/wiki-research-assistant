import streamlit as st
import logging
from typing import Dict, List, TYPE_CHECKING, AsyncGenerator, Any, Optional
import plotly.graph_objects as go
import asyncio
import datetime
import base64
import html
from chat_thread import ChatThreadManager, ChatThread
from llm_manager import LLMManager

if TYPE_CHECKING:
    from search_engine import SearchEngine

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class StreamInterface:
    """Interface for streaming updates from the pipeline."""
    
    def __init__(self, *args, **kwargs):
        """Initialize the interface."""
        # Initialize session state
        if "messages" not in st.session_state:
            st.session_state.messages = {}
        if "chat_manager" not in st.session_state:
            st.session_state.chat_manager = ChatThreadManager()
            # Create a default thread
            st.session_state.chat_manager.create_thread("default")
        if "results" not in st.session_state:
            st.session_state.results = []
        if "chat_processing" not in st.session_state:
            st.session_state.chat_processing = False
        if "chat_messages_queue" not in st.session_state:
            st.session_state.chat_messages_queue = []
        
        # Initialize state
        self._progress = 0.0
        self._results = {}  # Track results by ID
        self._result_containers = {}  # Track containers by ID
        self._total_steps = 0
        self._current_step = None
        self._step_counts = {}  # Track counts for each step
        self._step_order = []  # Track step order
        self._step_containers = {}  # Track containers for each step
        self._step_expanders = {}  # Track expanders for each step
        
        # Initialize main containers in specific order
        self._main_container = st.container()
        
        # Progress section (always at top)
        self._progress_section = self._main_container.container()
        self._status_cols = self._progress_section.columns(3)
        self._step_col = self._status_cols[0].empty()
        self._count_col = self._status_cols[1].empty()
        self._complete_col = self._status_cols[2].empty()
        self._progress_bar = self._progress_section.empty()
        self._progress_container = self._progress_section.empty()
        self._progress_log = self._progress_section.empty()
        
        # Results section (below progress)
        self._results_container = self._main_container.container()
        self._log_container = self._main_container.container()
        
        # Chat section (below results)
        self._chat_container = self._main_container.container()
        
        # Add loading animations
        self.add_loading_animations()

    def set_container(self, container):
        """Set the main Streamlit container."""
        self.container = container

    def _format_message(self, message: str, container: Optional[st.container] = None) -> str:
        """Format a message for display."""
        if not message:
            return ""
            
        # Add timestamp to message
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        message = f"[{timestamp}] {message}"
        
        # Only write to container if explicitly provided
        if container is not None:
            st.markdown(message, unsafe_allow_html=True)
        return message  # Return the formatted message

    def add_message(self, message: str, step: str, container: Optional[st.container] = None):
        """Add a message to the progress log for a specific step."""
        # logger.info(f"Adding message to progress log for step {step}: {message[:100]}...")
        formatted_message = self._format_message(message, container)  # Don't pass container here
        if formatted_message:
            if step not in st.session_state.messages:
                st.session_state.messages[step] = []
            st.session_state.messages[step].append(formatted_message)
            # logger.debug(f"Message added to session state for step {step}")
            
            # Update the progress log in a non-blocking way
            if hasattr(self, '_progress_log'):
                try:
                    with self._progress_log:
                        self._update_progress_log()
                except Exception as e:
                    logger.error(f"Error updating progress log: {str(e)}")

    def _update_progress_log(self):
        """Update the progress log display with enhanced categorization and styling."""
        if not hasattr(self, '_progress_log'):
            return
            
        css = """
        <style>
            .log-container {
                max-height: 400px;
                overflow-y: auto;
                padding: 10px;
                background: #F8F9FA;
                border-radius: 4px;
                margin: 10px 0;
            }
            .log-section {
                margin: 10px 0;
                padding: 8px;
                background: white;
                border-radius: 4px;
                box-shadow: 0 1px 2px rgba(0,0,0,0.1);
            }
            .log-section h4 {
                margin: 0 0 8px 0;
                color: #1565C0;
            }
            .log-message {
                margin: 4px 0;
                padding: 4px 0;
                border-bottom: 1px solid #E9ECEF;
            }
            .log-message:last-child {
                border-bottom: none;
            }
        </style>
        """
        
        sections = []
        if hasattr(st.session_state, 'messages'):
            # Process messages in reverse order of steps
            for step in reversed(list(st.session_state.messages.keys())):
                messages = st.session_state.messages[step]
                if messages:
                    section = [
                        f'<div class="log-section">',
                        f'<h4>Step: {step}</h4>',
                        '<div class="log-messages">',
                    ]
                    
                    # Add messages in chronological order
                    for msg in messages:
                        section.append(f'<div class="log-message">{msg}</div>')
                    
                    section.extend(['</div>', '</div>'])
                    sections.append('\n'.join(section))
        
        # Join all sections with dividers
        log_text = "\n".join(sections)
        
        # Update the progress log
        try:
            self._progress_log.markdown(f"{css}\n<div class='log-container'>{log_text}</div>", unsafe_allow_html=True)
        except Exception as e:
            logger.error(f"Error rendering progress log: {str(e)}")

    def _update_status(self, step: str = "", count: int = 0, complete: bool = False):
        """Update the status display."""
        if step:
            self._step_col.markdown(f"**Step**: {step}")
        if count > 0:
            self._count_col.markdown(f"**Total Items**: {count}")
        if complete:
            self._complete_col.markdown("✅ **Complete**")
        else:
            self._complete_col.markdown("🔄 **Processing**")

    def _update_progress(self, progress: float):
        """Update the progress bar."""
        if progress is not None:
            self._progress = min(progress, 0.99)
        self._progress_bar.progress(self._progress)

    def _display_progress(self, step: str, progress: float, message: str = None):
        """Display progress with animation."""
        progress_html = f"""
        <div class="progress-wrapper" style="position: relative; z-index: 1000;">
            <div class="progress-header">
                <div class="spinner"></div>
                <div class="progress-title">{step}</div>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {progress * 100}%;"></div>
            </div>
            {f'<div class="progress-message loading">{message}</div>' if message else ''}
        </div>
        """
        self._progress_container.markdown(progress_html, unsafe_allow_html=True)

    async def cleanup(self):
        """Clean up resources."""
        # Clear containers
        if hasattr(self, '_main_container'):
            self._main_container.empty()
        if hasattr(self, '_progress_container'):
            self._progress_container.empty()
        if hasattr(self, '_results_container'):
            self._results_container.empty()
        if hasattr(self, '_log_container'):
            self._log_container.empty()
        if hasattr(self, '_progress_bar'):
            self._progress_bar.empty()
        if hasattr(self, '_progress_log'):
            self._progress_log.empty()
        if hasattr(self, '_step_col'):
            self._step_col.empty()
        if hasattr(self, '_count_col'):
            self._count_col.empty()
        if hasattr(self, '_complete_col'):
            self._complete_col.empty()
        
        # Reset state
        self._progress = 0.0
        self._results = {}
        self._result_containers = {}
        self._total_steps = 0
        self._current_step = None
        self._step_counts = {}
        self._step_order = []
        self._step_containers = {}
        self._step_expanders = {}
        self._expected_data_counts = {}
        self._received_data_counts = {}

    async def stream_research_progress(self, pipeline_generator: AsyncGenerator[Dict[str, Any], None]):
        """Stream progress updates and results from the pipeline."""
        try:
            # Initialize state for new search
            self._progress = 0.0
            self._results = {}
            self._result_containers = {}
            self._total_steps = 0
            self._current_step = None
            self._step_counts = {}
            self._step_order = []
            self._step_containers = {}
            self._step_expanders = {}
            self._expected_data_counts = {}  # Track expected data counts per step
            self._received_data_counts = {}  # Track received data counts per step
            
            async for update in pipeline_generator:
                if update.get('type') == 'error':
                    st.error(update.get('message', 'An error occurred'))
                    continue
                
                step = update.get('step', '')
                progress = update.get('progress', 0)
                message = update.get('message', '')
                expected_count = update.get('total_items', 0)  # Get expected total items if provided
                
                # Update expected count if provided
                if expected_count > 0:
                    self._expected_data_counts[step] = expected_count
                
                # Initialize received count for new step
                if step not in self._received_data_counts:
                    self._received_data_counts[step] = 0
                
                # Display progress with animation
                self._display_progress(step, progress, message)
                
                # Create or get step container
                if step not in self._step_counts:
                    self._step_counts[step] = 0
                    self._total_steps += 1
                    self._step_order.insert(0, step)  # Insert new steps at the beginning
                    
                    # Clear results container
                    self._results_container.empty()
                    main_container = self._results_container.container()
                    
                    # Recreate all step containers in current order
                    for current_step in self._step_order:
                        # Either create new expander or reuse existing one
                        if current_step in self._step_expanders:
                            self._step_expanders[current_step].label = f"Results"
                            # Collapse previous steps, expand current step
                            self._step_expanders[current_step].expanded = (current_step == step)
                        else:
                            main_container.markdown(f"#### Step: {current_step}")
                            # Only expand the current step
                            self._step_expanders[current_step] = main_container.expander(
                                "Results", 
                                expanded=(current_step == step)
                    )
                
                        if not current_step in self._step_containers:
                            self._step_containers[current_step] = self._step_expanders[current_step].container()
                
                    self._step_counts[step] += 1
                
                # Update current step and progress
                if self._current_step != step:
                    self._current_step = step
                    logger.info(f"Starting step: {step}")
                    self.add_message("▶️ *Starting processing...*", step)
                    
                    # Collapse previous steps when switching to a new one
                    for prev_step, expander in self._step_expanders.items():
                        if prev_step != step:
                            expander.expanded = False
                
                # Calculate overall progress
                step_index = len(self._step_order) - self._step_order.index(step) - 1
                self._progress = (step_index / max(self._total_steps, 1)) + (1 / max(self._total_steps, 1) * 0.5)
                self._update_progress(self._progress)
                
                # Handle the data which is now always a list
                data_list = update.get('data', [])
                if not isinstance(data_list, list):
                    data_list = [data_list]
                
                # Update received count
                self._received_data_counts[step] += len(data_list)
                
                
                # Create a fresh container for this batch of results
                batch_container = self._step_containers[step]
                
                # Process items in reverse order so newest appears first
                for item in data_list:
                    if isinstance(item, dict) and (item.get('article_id') or item.get('url') or item.get('title') or item.get('id')):
                        # Get result ID from various possible sources
                        result_id = (
                            item.get('article_id') or 
                            item.get('url') or 
                            item.get('title') or 
                            item.get('id')
                        )
                        
                        if result_id:
                            self._results[result_id] = item
                            self._result_containers[result_id] = batch_container
                            self._display_result({'data': item}, batch_container)
                            
                            # Add result message to step log without container
                            title = item.get('title', result_id)
                            self.add_message(f"📄 **Processed**: {step} - {title}", step)
                    elif isinstance(item, dict) and (item.get('query') or item.get('content')):
                        # Handle string/simple content
                        content = item.get('query') or item.get('content')
                        if content:
                            self._results[content] = {'content': content}
                            self._display_result({'data': {'content': content}}, batch_container)
                    else:
                        # Handle non-dict items (e.g. strings, numbers)
                        result_id = str(item)
                        if result_id:
                            self._results[result_id] = {'content': item}
                            self._display_result({'data': {'content': item}}, batch_container)
                            self.add_message(f"🖋️ **Processed**: {step} - {result_id}", step)
                
                # Update progress display
                result_count = len(self._results)
                self._update_status(
                    step=f"{step} ({len(data_list)} items)",
                    count=result_count,
                    complete=update.get('is_final', False) and update.get('progress', 0) == 1.0
                )
                
                if update.get('is_final', False) and update.get('progress', 0) == 1.0:
                    # Step is complete
                    logger.info(f"Step {step} completed")
                    self.add_message(f"✅ **Step completed** with {self._step_counts[step]} items processed", step)
                    # Collapse the expander when step is complete
                    self._step_expanders[step].expanded = False
                    if step == self._step_order[0]:  # Check first step since order is reversed
                        self.add_message("✨ **All processing completed!**", step)
                        self._update_progress(1.0)
                        self.display_chat_history()
                
        except Exception as e:
            logger.error(f"Error in pipeline processing: {str(e)}", exc_info=True)
            if self._current_step:
                self.add_message(f"❌ **Error**: {str(e)}", self._current_step)
            st.error(f"Error streaming progress: {str(e)}")
            raise
        finally:
            await self.cleanup()

    def add_results(self, results):
        """Add results to session state."""
        if not hasattr(st.session_state, 'results'):
            st.session_state.results = []
            
        # Handle both single result and list of results
        if isinstance(results, dict):
            st.session_state.results.append(results)
            self._results[results.get('id', str(len(self._results)))] = results
        elif isinstance(results, list):
            st.session_state.results.extend(results)
            for result in results:
                self._results[result.get('id', str(len(self._results)))] = result

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

    def get_export_data(self, results: List[Dict]) -> str:
        """Generate markdown export data from results."""
        if not results:
            return ""
            
        markdown = ["# Research Results\n"]
        
        for result in results:
            title = result.get('title', '')
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
        with self._progress_container:
            st.markdown("### 🔄 Search Progress")
            # Create progress bar
            self._progress_bar = progress_bar
            # Create log container
            self._progress_log = progress_log

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

    def has_results(self) -> bool:
        """Check if there are any results."""
        return bool(st.session_state.results)

    def _display_result(self, result: Dict[str, Any], container: st.container):
        """Display a search result in the provided container."""
        if not result or not isinstance(result, dict):
            return

        data = result.get('data', {})
        
        # Handle string/simple content
        if isinstance(data, str) or (isinstance(data, dict) and 'content' in data and isinstance(data['content'], str)):
            content = data if isinstance(data, str) else data['content']
            
            # Detect if content looks like a search query
            is_query = any(keyword in content.lower() for keyword in ['search', 'query', 'find', 'look for'])
            icon = '🔍' if is_query else '📝'
            
            markdown = f"""
<style>
.simple-result {{
    background: white;
    border-radius: 12px;
    margin: 16px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    overflow: hidden;
}}
.simple-header {{
    display: flex;
    align-items: center;
    background: #f3f6fc;
    padding: 12px 16px;
    border-bottom: 1px solid #e8eaed;
}}
.simple-icon {{
    font-size: 1.2em;
    margin-right: 12px;
}}
.simple-type {{
    color: #5f6368;
    font-size: 0.85em;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
.simple-content {{
    padding: 16px;
    color: #202124;
    line-height: 1.6;
    font-size: 1.1em;
}}
.query-style {{
    color: #1a73e8;
    font-weight: 500;
}}
</style>

<div class="simple-result">
    <div class="simple-header">
        <div class="simple-icon">{icon}</div>
        <div class="simple-type">{is_query and "Search Query" or "Analysis"}</div>
    </div>
    <div class="simple-content{is_query and ' query-style' or ''}">
        {content}
    </div>
</div>
"""
            container.markdown(markdown, unsafe_allow_html=True)
            return
            
        # Handle complex dictionary data
        title = data.get('title', '')
        url = data.get('url', '')
        score = data.get('score', 0)
        document = data.get('document', '')
        summary = data.get('summary', '')
        query = data.get('query', '')
        claim = data.get('claim', '')
        article_id = data.get('article_id', '')
        validation_rate = data.get('validation_rate')
        
        # Build the result box
        parts = []
        parts.append('<div class="result-box">')
        
        # Header section with relevance score
        parts.append('<div class="result-header">')
        if score:
            score_color = '#4CAF50' if score > 0.8 else '#FFC107' if score > 0.6 else '#FF5722'
            parts.append(f'<div class="relevance-score" style="background-color: {score_color}">')
            parts.append(f'<div class="score-value">{score:.0%}</div>')
            parts.append('<div class="score-label">relevant</div>')
            parts.append('</div>')
        
        # Title and source
        parts.append('<div class="header-content">')
        if url:
            parts.append(f'<h3 class="title"><a href="{url.replace(" ", "%20")}" target="_blank">{title or "Untitled"}</a></h3>')
        elif title:
            parts.append(f'<h3 class="title">{title}</h3>')
        if article_id:
            parts.append(f'<div class="source-id">Source ID: {article_id}</div>')
        parts.append('</div>')  # Close header-content
        parts.append('</div>')  # Close result-header
            
        # Query and Claim section
        if query or claim or validation_rate is not None:
            parts.append('<div class="search-context">')
            if query:
                parts.append('<div class="query-box">')
                parts.append('<div class="query-label">Search Query</div>')
                parts.append(f'<div class="query-text">{query}</div>')
                parts.append('</div>')
            if claim and claim is str:
                parts.append('<div class="claim-box">')
                parts.append('<div class="claim-label">Question/Claim</div>')
                parts.append(f'<div class="claim-text">{claim}</div>')
                parts.append('</div>')
            if validation_rate is not None:
                validation_color = '#4CAF50' if validation_rate >= 80 else '#FFC107' if validation_rate >= 50 else '#FF5722'
                parts.append(f'<div class="validation-rate" style="color: {validation_color}">')
                parts.append(f'Validation Rate: {validation_rate:.0f}%')
                parts.append('</div>')
            parts.append('</div>')
            
        # Document content section
        if summary:
            parts.append('<div class="summary-section">')
            parts.append('<div class="summary-label">Summary</div>')
            parts.append(f'<div class="summary-text">{summary}</div>')
            parts.append('</div>')
        elif document:
            parts.append('<div class="document-section">')
            parts.append('<div class="document-content">')
            parts.append(document)
            parts.append('</div>')
            parts.append('</div>')
            
        parts.append('</div>')  # Close result-box
        
        # Add CSS
        css = """
        <style>
        .result-box {
            background: white;
            border-radius: 12px;
            padding: 0;
            margin: 16px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .result-header {
            display: flex;
            align-items: stretch;
            background: #f8f9fa;
            border-bottom: 1px solid #e8eaed;
            padding: 16px;
        }
        .relevance-score {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            min-width: 60px;
            height: 60px;
            border-radius: 8px;
            margin-right: 16px;
            color: white;
            padding: 8px;
        }
        .score-value {
            font-size: 1.2em;
            font-weight: bold;
        }
        .score-label {
            font-size: 0.8em;
            opacity: 0.9;
        }
        .header-content {
            flex-grow: 1;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .title {
            margin: 0 0 4px 0;
            color: #1a73e8;
            font-size: 1.2em;
            line-height: 1.3;
        }
        .title a {
            color: inherit;
            text-decoration: none;
        }
        .title a:hover {
            text-decoration: underline;
        }
        .source-id {
            color: #5f6368;
            font-size: 0.9em;
        }
        .search-context {
            padding: 16px;
            background: #f3f6fc;
            border-bottom: 1px solid #e8eaed;
        }
        .query-box, .claim-box {
            margin-bottom: 8px;
        }
        .query-box:last-child, .claim-box:last-child {
            margin-bottom: 0;
        }
        .query-label, .claim-label {
            font-size: 0.85em;
            text-transform: uppercase;
            color: #5f6368;
            margin-bottom: 4px;
            letter-spacing: 0.5px;
        }
        .query-text {
            color: #3c4043;
            font-size: 1em;
        }
        .claim-text {
            color: #1a73e8;
            font-size: 1.1em;
            font-weight: 500;
            margin-bottom: 4px;
        }
        .validation-rate {
            font-size: 0.9em;
            font-weight: 500;
            margin-top: 4px;
        }
        .document-section {
            padding: 16px;
        }
        .document-content {
            color: #202124;
            line-height: 1.6;
            font-size: 1em;
        }
        </style>
        """
        
        # Combine CSS and content
        markdown = f"{css}\n{''.join(parts)}"
        container.markdown(markdown, unsafe_allow_html=True)

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

    def add_loading_animations(self):
        """Add loading animations and progress indicators."""
        st.markdown("""
        <style>
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.3; }
            100% { opacity: 1; }
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .loading {
            animation: pulse 1.5s ease-in-out infinite;
        }
        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(0,0,0,.1);
            border-radius: 50%;
            border-top-color: #1a73e8;
            animation: spin 1s linear infinite;
            margin-right: 8px;
            vertical-align: middle;
        }
        .processing-indicator {
            display: flex;
            align-items: center;
            padding: 8px 16px;
            background: #f3f6fc;
            border-radius: 8px;
            color: #1a73e8;
            font-weight: 500;
            margin: 8px 0;
        }
        .processing-indicator .spinner {
            margin-right: 12px;
        }
        .progress-wrapper {
            background: #f3f6fc;
            border-radius: 8px;
            padding: 16px;
            margin: 16px 0;
            position: sticky;
            top: 0;
            z-index: 1000;
            backdrop-filter: blur(10px);
        }
        .progress-header {
            display: flex;
            align-items: center;
            margin-bottom: 8px;
        }
        .progress-title {
            font-weight: 500;
            color: #1a73e8;
            margin-left: 8px;
        }
        .progress-bar {
            height: 6px;
            background: #e8eaed;
            border-radius: 3px;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            background: #1a73e8;
            border-radius: 3px;
            transition: width 0.3s ease;
        }
        .progress-message {
            margin-top: 8px;
            color: #5f6368;
        }
        .stButton > button:disabled {
            opacity: 0.7;
            cursor: not-allowed;
        }
        .stButton > button[data-loading="true"] {
            position: relative;
            padding-right: 40px;
        }
        .stButton > button[data-loading="true"]::after {
            content: '';
            position: absolute;
            width: 16px;
            height: 16px;
            right: 12px;
            top: 50%;
            transform: translateY(-50%);
            border: 2px solid rgba(255,255,255,.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 1s linear infinite;
        }
        </style>
        """, unsafe_allow_html=True)

    def show_processing_indicator(self):
        """Show a processing indicator with spinner."""
        st.markdown("""
            <div class="processing-indicator">
                <div class="spinner"></div>
                <span>Processing your request...</span>
            </div>
        """, unsafe_allow_html=True)

    def display_chat_history(self):
        """Display the chat history in the chat container."""
        with self._chat_container:
            st.header("Chat History")
            
            # Get the default thread
            thread = st.session_state.chat_manager.get_thread("default")
            
            # Process any queued messages
            if st.session_state.chat_messages_queue:
                for msg_data in st.session_state.chat_messages_queue:
                    thread.add_message(**msg_data)
                st.session_state.chat_messages_queue = []
            
            # Display messages with different styling based on role
            for msg in thread.messages:
                with st.container():
                    # Different background colors for different roles
                    if msg.role == "user":
                        st.markdown("""
                        <div style='background-color: #f0f2f6; padding: 10px; border-radius: 10px; margin: 5px 0;'>
                            <strong>You:</strong><br>{0}
                        </div>
                        """.format(msg.content), unsafe_allow_html=True)
                    elif msg.role == "assistant":
                        st.markdown("""
                        <div style='background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin: 5px 0;'>
                            <strong>Assistant:</strong><br>{0}
                        </div>
                        """.format(msg.content), unsafe_allow_html=True)
                    elif msg.role == "system":
                        st.markdown("""
                        <div style='background-color: #fff3e0; padding: 10px; border-radius: 10px; margin: 5px 0;'>
                            <strong>System:</strong><br>{0}
                        </div>
                        """.format(msg.content), unsafe_allow_html=True)
                    
                    # If there are pipeline results, show them in an expander
                    if msg.pipeline_results:
                        with st.expander("View Pipeline Results"):
                            st.json(msg.pipeline_results)
            
            # Add chat input
            chat_input = st.text_input("Chat with Assistant", key="chat_input")
            
            # Show processing indicator if chat is being processed
            if st.session_state.chat_processing:
                st.info("Processing your message...")
            
            # Handle send button click
            if st.button("Send", key="send_chat", disabled=st.session_state.chat_processing):
                if chat_input:
                    # Queue the user message
                    st.session_state.chat_messages_queue.append({
                        'role': "user",
                        'content': chat_input,
                        'metadata': {"type": "chat"}
                    })
                    
                    # Get context from previous messages and results
                    context = self.get_chat_context(thread)
                    
                    # Set processing flag
                    st.session_state.chat_processing = True
                    
                    # Process chat with LLM
                    asyncio.create_task(self.process_chat_message(thread, chat_input, context))
                    
                    # Clear input (using session state to persist across reruns)
                    st.session_state.chat_input = ""

    def get_chat_context(self, thread) -> str:
        """Get context for chat from thread history and results."""
        context_parts = []
        
        # Add recent results if available
        results = self.get_results()
        if results:
            context_parts.append("Recent search results:")
            for result in results[:3]:  # Include top 3 results
                title = result.get('title', '')
                content = result.get('content', '')
                if title and content:
                    context_parts.append(f"- {title}: {content[:200]}...")
        
        # Add recent messages
        messages = thread.messages[-5:]  # Last 5 messages
        if messages:
            context_parts.append("\nRecent conversation:")
            for msg in messages:
                context_parts.append(f"{msg.role}: {msg.content}")
        
        return "\n".join(context_parts)

    async def process_chat_message(self, thread, user_message: str, context: str):
        """Process a chat message with the LLM."""
        try:
            # Create system prompt with context
            system_prompt = f"""You are a helpful research assistant. Use the following context to inform your responses:
            
{context}

Respond in a helpful and informative way, using the context when relevant."""
            
            # Get LLM response
            async with LLMManager() as llm:
                response = await llm.get_response(
                    prompt=user_message,
                    system_prompt=system_prompt,
                    temperature=0.7
                )
            
            if response:
                # Queue the assistant response
                st.session_state.chat_messages_queue.append({
                    'role': "assistant",
                    'content': response,
                    'metadata': {"type": "chat_response"}
                })
            else:
                # Queue error message
                st.session_state.chat_messages_queue.append({
                    'role': "system",
                    'content': "Failed to get response from assistant",
                    'metadata': {"type": "error"}
                })
                
        except Exception as e:
            logger.error(f"Error processing chat message: {str(e)}")
            # Queue error message
            st.session_state.chat_messages_queue.append({
                'role': "system",
                'content': f"Error: {str(e)}",
                'metadata': {"type": "error"}
            })
        finally:
            # Clear processing flag
            st.session_state.chat_processing = False
