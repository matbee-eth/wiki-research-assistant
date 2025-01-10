# visualization.py

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
import logging
import aiohttp
import json
import html
import re
import asyncio
from dateutil import parser
from utils import fetch_gpt_response, log_function_call
import numpy as np
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def create_custom_theme():
    """Create a custom modern theme for visualizations."""
    return {
        'background_color': '#1a1a1a',
        'text_color': '#ffffff',
        'primary_color': '#00ff88',
        'secondary_color': '#ff3366',
        'accent_color': '#9933ff',
        'font_family': 'Inter, sans-serif',
        'timeline_colors': px.colors.qualitative.Set3
    }

@log_function_call
async def extract_timeline_events(session: aiohttp.ClientSession, text: str, context: str = "") -> List[Dict[str, Any]]:
    """Extract timeline events from text using LLM."""
    try:
        prompt = f"""
        Extract timeline events from the following text. Format as a JSON array of events.
        Each event should have: date (YYYY-MM-DD), date_str (human readable), description (event details).
        Only include events with clear dates or time periods. Be precise and factual.
        For ancient dates before 1677, use the original date in date_str but set date to 1677-01-01.

        Text: {text}
        Context: {context}

        Example format:
        [
            {{
                "date": "1776-07-04",
                "date_str": "July 4, 1776",
                "description": "US Declaration of Independence signed"
            }},
            {{
                "date": "1677-01-01",
                "date_str": "81 BC",
                "description": "Ancient event description"
            }}
        ]

        Respond ONLY with the JSON array, no other text.
        """
        
        response = await fetch_gpt_response(session, prompt)
        logger.debug(f"Raw LLM response for event extraction:\n{response}")
        
        # Clean up response to extract JSON
        json_str = response.strip()
        if '```json' in json_str:
            json_str = json_str.split('```json')[1].split('```')[0].strip()
        elif '```' in json_str:
            json_str = json_str.split('```')[1].strip()
            
        logger.debug(f"Cleaned JSON string:\n{json_str}")
        events = json.loads(json_str)
        
        if not isinstance(events, list):
            logger.warning(f"Expected list of events, got {type(events)}")
            events = []
        
        # Validate and clean events
        valid_events = []
        for event in events:
            try:
                if all(k in event for k in ['date', 'date_str', 'description']):
                    # Normalize date to supported range
                    event['date'] = normalize_date(event['date'])
                    valid_events.append(event)
                else:
                    logger.warning(f"Skipping invalid event: {event}")
            except (ValueError, TypeError) as e:
                logger.warning(f"Error processing event {event}: {str(e)}")
                
        logger.info(f"Successfully extracted {len(valid_events)} events")
        return valid_events
        
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM response as JSON: {str(e)}")
        logger.debug("Attempting regex-based extraction as fallback")
        
        # Improved regex pattern for date extraction
        patterns = [
            r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})[:\s]+([^.!?\n]+[.!?])',  # YYYY-MM-DD
            r'(\d{1,2}[-/]\d{1,2}[-/]\d{4})[:\s]+([^.!?\n]+[.!?])',  # DD-MM-YYYY
            r'([A-Z][a-z]+ \d{1,2},? \d{4})[:\s]+([^.!?\n]+[.!?])',  # Month DD, YYYY
            r'(\d{4})[:\s]+([^.!?\n]+[.!?])'  # YYYY only
        ]
        
        events = []
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                try:
                    date_str = match.group(1)
                    desc = match.group(2).strip()
                    
                    # Normalize date to supported range
                    normalized_date = normalize_date(date_str)
                    
                    events.append({
                        'date': normalized_date,
                        'date_str': date_str,
                        'description': desc
                    })
                except (ValueError, TypeError) as e:
                    logger.warning(f"Error parsing date {date_str}: {str(e)}")
                    
        logger.debug(f"Found {len(events)} potential events using regex")
        return events
        
    except Exception as e:
        logger.error(f"Error extracting timeline events: {str(e)}", exc_info=True)
        return []

def validate_date(date_str: str) -> bool:
    """Validate if a date is within pandas' supported range (1677-2262)."""
    try:
        date = pd.to_datetime(date_str)
        min_date = pd.Timestamp.min
        max_date = pd.Timestamp.max
        return min_date <= date <= max_date
    except Exception:
        return False

def normalize_date(date_str: str) -> str:
    """Normalize dates to be within pandas' supported range."""
    try:
        date = pd.to_datetime(date_str)
        min_date = pd.Timestamp.min
        max_date = pd.Timestamp.max
        
        if date < min_date:
            logger.warning(f"Date {date_str} before minimum supported date, using 1677-01-01")
            return "1677-01-01"
        elif date > max_date:
            logger.warning(f"Date {date_str} after maximum supported date, using 2262-01-01")
            return "2262-01-01"
        return date.strftime("%Y-%m-%d")
    except Exception as e:
        logger.error(f"Error normalizing date {date_str}: {str(e)}")
        return "1677-01-01"

@log_function_call
async def visualize_timeline(results: List[Dict[str, Any]]):
    """
    Create an interactive and visually stunning timeline visualization of events.
    
    Args:
        results: List of search results containing text content
    """
    try:
        if not results:
            logger.info("No results to visualize timeline.")
            if 'st' in globals():
                st.info("No timeline events could be extracted from the results.")
            return
            
        logger.info(f"Starting timeline visualization for {len(results)} results")
        theme = create_custom_theme()
        
        # Create aiohttp session for API calls
        async with aiohttp.ClientSession() as session:
            all_events = []
            for i, result in enumerate(results, 1):
                try:
                    logger.debug(f"Processing result {i}/{len(results)}: {result['title']}")
                    
                    # Use available content fields
                    text = f"{result['title']}\n\n"
                    if 'content' in result:
                        text += result['content']
                        logger.debug(f"Using content field for result {i} ({len(result['content'])} chars)")
                    elif 'summary' in result:
                        text += result['summary']
                        logger.debug(f"Using summary field for result {i} ({len(result['summary'])} chars)")
                    else:
                        logger.warning(f"No content or summary found for result {i}")
                        continue
                    
                    # Add key events if available
                    if 'key_events' in result:
                        text += f"\n\n{result['key_events']}"
                        logger.debug(f"Added key_events for result {i}")
                    
                    events = await extract_timeline_events(session, text, context=result['title'])
                    logger.debug(f"Extracted {len(events)} events from result {i}")
                    
                    # Add source and metadata to each event
                    for event in events:
                        event['source_title'] = result['title']
                        event['statement'] = event['description']
                        event['category'] = classify_event(event['description'])
                    
                    all_events.extend(events)
                    logger.debug(f"Added {len(events)} events from result {i} to timeline")
                except Exception as e:
                    logger.error(f"Error processing timeline for result {i}: {str(e)}", exc_info=True)
                    continue
        
        if not all_events:
            logger.warning("No timeline events could be extracted from the results")
            if 'st' in globals():
                st.error("⚠️ No timeline events could be extracted from the results.")
            return

        logger.info(f"Creating visualization for {len(all_events)} total events")
        
        # Create DataFrame and sort by date
        df = pd.DataFrame(all_events)
        
        # Convert date column to datetime and handle old dates
        df['date'] = df['date'].apply(lambda x: pd.to_datetime(normalize_date(str(x))))
        df = df.sort_values('date')
        logger.debug(f"Created DataFrame with shape: {df.shape}")
        
        # Create alternating positions for events with varying distances
        df['Position'] = [1 if i % 2 == 0 else -1 for i in range(len(df))]
        df['Position'] *= np.random.uniform(0.8, 1.2, size=len(df))  # Add some randomness
        
        # Create the main figure
        fig = go.Figure()
        
        # Add decorative background elements
        add_background_elements(fig, df['date'].min(), df['date'].max(), theme)
        
        # Add central timeline with gradient
        add_timeline_base(fig, df['date'].min(), df['date'].max(), theme)
        
        # Add events with animations and hover effects
        add_timeline_events(fig, df, theme)
        
        # Update layout with modern styling
        update_layout(fig, theme)
        
        # Add interactive controls if streamlit is available
        if 'st' in globals():
            add_timeline_controls(df)
        
        # Display timeline if streamlit is available
        logger.info("Displaying timeline visualization")
        if 'st' in globals():
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            add_supplementary_visualizations(df, theme)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error in visualize_timeline: {str(e)}", exc_info=True)
        if 'st' in globals():
            st.error(f"⚠️ Error creating timeline visualization: {str(e)}")
        return None

def classify_event(description: str) -> str:
    """Classify event into categories based on content."""
    categories = {
        'Legal': ['court', 'trial', 'judge', 'lawsuit', 'legal', 'charged', 'convicted'],
        'Investigation': ['investigation', 'FBI', 'police', 'evidence', 'probe'],
        'Personal': ['born', 'died', 'married', 'moved', 'relocated'],
        'Business': ['company', 'business', 'founded', 'investment', 'financial'],
        'Social': ['met', 'attended', 'party', 'relationship', 'friend'],
    }
    
    description = description.lower()
    for category, keywords in categories.items():
        if any(keyword in description for keyword in keywords):
            return category
    return 'Other'

def add_background_elements(fig, start_date, end_date, theme):
    """Add decorative background elements to the timeline."""
    try:
        # Normalize dates to supported range
        start_date = pd.to_datetime(normalize_date(str(start_date)))
        end_date = pd.to_datetime(normalize_date(str(end_date)))
        
        # Add year markers
        for year in pd.date_range(start_date, end_date, freq='Y'):
            fig.add_shape(
                type="line",
                x0=year,
                x1=year,
                y0=-2,
                y1=2,
                line=dict(color=theme['text_color'], width=0.5, dash="dot"),
                opacity=0.3,
                layer="below"
            )
    except Exception as e:
        logger.error(f"Error adding background elements: {str(e)}")

def add_timeline_base(fig, start_date, end_date, theme):
    """Add the central timeline with gradient effect."""
    fig.add_trace(go.Scatter(
        x=[start_date, end_date],
        y=[0, 0],
        mode='lines',
        line=dict(
            color=theme['primary_color'],
            width=3,
            shape='spline'
        ),
        hoverinfo='skip'
    ))

def add_timeline_events(fig, df, theme):
    """Add events to the timeline with animations and hover effects."""
    categories = df['category'].unique()
    colors = dict(zip(categories, theme['timeline_colors']))
    
    for i, row in df.iterrows():
        y_pos = row['Position'] * 0.15
        
        # Add event point
        fig.add_trace(go.Scatter(
            x=[row['date']],
            y=[y_pos],
            mode='markers+text',
            marker=dict(
                size=12,
                color=colors[row['category']],
                line=dict(color=theme['text_color'], width=1),
                symbol='circle'
            ),
            text=[row['date_str']],
            textposition='middle right' if y_pos > 0 else 'middle left',
            hovertemplate=(
                f"<b>{row['date_str']}</b><br>" +
                f"{row['statement']}<br><br>" +
                f"<i>Source: {row['source_title']}</i><br>" +
                f"Category: {row['category']}<extra></extra>"
            ),
            showlegend=False
        ))
        
        # Add connecting line to timeline
        fig.add_trace(go.Scatter(
            x=[row['date'], row['date']],
            y=[0, y_pos],
            mode='lines',
            line=dict(
                color=colors[row['category']],
                width=1,
                dash='dot'
            ),
            hoverinfo='skip',
            showlegend=False
        ))

def update_layout(fig, theme):
    """Update the figure layout with modern styling."""
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=theme['background_color'],
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(
            text="Interactive Timeline of Events",
            font=dict(
                family=theme['font_family'],
                size=24,
                color=theme['text_color']
            ),
            x=0.5,
            y=0.95
        ),
        showlegend=True,
        legend=dict(
            title="Event Categories",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            title="Date",
            titlefont=dict(family=theme['font_family']),
            tickfont=dict(family=theme['font_family'])
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-2, 2]
        ),
        height=600,
        margin=dict(l=50, r=50, t=100, b=50),
        hovermode='closest'
    )

def add_timeline_controls(df):
    """Add interactive controls for timeline manipulation."""
    st.sidebar.markdown("### Timeline Controls")
    
    # Convert dates to datetime for the date picker
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    
    # Date range selector
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Category filter
    categories = ['All'] + list(df['category'].unique())
    selected_categories = st.sidebar.multiselect(
        "Filter by Category",
        categories,
        default=['All']
    )

def add_supplementary_visualizations(df, theme):
    """Add additional visualizations to provide more insights."""
    st.markdown("### Event Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Event frequency over time
        df_time = df.copy()
        # Convert date column to datetime
        df_time['date'] = pd.to_datetime(df_time['date'])
        # Set date as index for resampling
        df_time.set_index('date', inplace=True)
        # Resample by year
        events_by_time = df_time.resample('Y').size()
        
        fig_freq = px.line(
            events_by_time,
            title="Event Frequency Over Time",
            template='plotly_dark'
        )
        st.plotly_chart(fig_freq, use_container_width=True)
    
    with col2:
        # Event categories distribution
        category_counts = df['category'].value_counts()
        fig_cats = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title="Event Categories Distribution",
            template='plotly_dark'
        )
        st.plotly_chart(fig_cats, use_container_width=True)

def clean_html_markup(text: str) -> str:
    """
    Clean HTML markup and convert to markdown-friendly format.
    """
    if not text:
        return ""
        
    # Unescape HTML entities
    text = html.unescape(text)
    
    # Handle data-mw-anchor attributes
    text = re.sub(r'<h2 data-mw-anchor="[^"]*">', '## ', text)
    text = re.sub(r'<h3 data-mw-anchor="[^"]*">', '### ', text)
    
    # Convert HTML tags to markdown
    replacements = [
        (r'<p>', '\n\n'),
        (r'</p>', ''),
        (r'<b>', '**'),
        (r'</b>', '**'),
        (r'<i>', '*'),
        (r'</i>', '*'),
        (r'<h2>', '## '),
        (r'</h2>', '\n\n'),
        (r'<h3>', '### '),
        (r'</h3>', '\n\n'),
        (r'<ul>', '\n'),
        (r'</ul>', '\n'),
        (r'<li>', '* '),
        (r'</li>', '\n'),
        (r'<div[^>]*>', ''),
        (r'</div>', '\n'),
        (r'<span[^>]*>', ''),
        (r'</span>', '')
    ]
    
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    
    # Handle italics in citations
    text = re.sub(r'([A-Za-z. ]+) v\. ([A-Za-z. ]+)', r'*\1 v. \2*', text)
    
    # Clean up multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def visualize_results(results: List[Dict[str, Any]]) -> None:
    """
    Display search results in a structured format.
    
    Args:
        results: List of search results with metadata
    """
    for result in results:
        if result.get('is_synthesis'):
            st.markdown("## 📊 Research Synthesis")
            st.markdown(result['content'])
            continue
            
        st.markdown(f"## 📚 {result['title']} (Score: {result['score']:.2f})")
        
        # Create columns for layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Main content
            if result.get('analytical_summary'):
                st.markdown("### 📝 Analysis")
                st.markdown(result['analytical_summary'])
            
            if result.get('key_events'):
                st.markdown("### 📅 Key Events")
                st.markdown(result['key_events'])
            
            st.markdown("### 📄 Full Content")
            with st.expander("Show full content"):
                st.markdown(result['content'])
        
        with col2:
            # Show entities
            if result.get('entities'):
                st.markdown("### 🔍 Key Entities")
                for entity, label in result['entities']:
                    st.markdown(f"- **{entity}** ({label})")
            
            # Show topics
            if result.get('topics'):
                st.markdown("### 📌 Topics")
                for topic in result['topics']:
                    st.markdown(f"- {', '.join(topic)}")
        
        # Add source link
        if result.get('url'):
            st.markdown(f"[View Source]({result['url']})")
        st.markdown("---")
    
    # Create word cloud from all content
    if any(result.get('content') for result in results):
        st.subheader("Word Cloud")
        # Use raw text for word cloud
        all_text = ' '.join([result.get('content', '') for result in results])
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=100
        ).generate(all_text)
        
        # Display word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
        plt.close()
