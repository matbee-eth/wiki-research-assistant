# Semantic Research Assistant

A powerful research tool that leverages AI to search, analyze, and generate literature reviews from Wikipedia articles.
[Screencast from 2025-01-15 12:35:52 PM (trimmed).webm](https://github.com/user-attachments/assets/bd288b53-90ef-48e4-9200-5a2b7339fbb3)

## Features

- **Semantic Search**: Search Wikipedia articles using natural language queries with semantic understanding
- **Real-time Analysis**: Get instant analysis of articles as they are processed
- **Individual Literature Reviews**: Generate detailed literature reviews for each article
- **Interactive UI**: Built with Streamlit for a modern, responsive interface
- **Caching System**: Smart caching of processed articles for improved performance
- **Progress Tracking**: Real-time updates on search and analysis progress

## Components

### 1. Search Engine (`search_engine.py`)
- Core search functionality with semantic understanding
- Parallel processing of search results
- Article caching and management
- Integration with OpenAI compatible APIs for analysis

### 2. NLP Utils (`nlp_utils.py`)
- Natural language processing utilities
- Literature review generation
- Entity extraction and analysis

### 3. Stream Interface (`stream_interface.py`)
- Streamlit-based user interface
- Real-time progress updates
- Interactive result display
- Export functionality

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
export OPENAI_API_KEY=your_api_key
export OPENAI_API_BASE=your_api_base
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Enter your research query in the search box
2. The system will:
   - Search for relevant Wikipedia articles
   - Analyze each article's content
   - Generate individual literature reviews
   - Display results with summaries and analyses

3. Results include:
   - Article summary
   - Detailed analysis
   - Literature review
   - Source links

## Caching
- Cached articles are stored in `article_cache/article_cache.json`

## Technical Details

### Search Process
1. Query processing and semantic search
2. Parallel article processing
3. LLM-powered analysis
4. Literature review generation
5. Real-time result streaming

### Result Types
- Wiki summaries
- Article analyses
- Literature reviews
- Progress updates

## Contributing

Feel free to submit issues and pull requests for:
- New features
- Bug fixes
- Documentation improvements
- Performance enhancements

## License

MIT License - See LICENSE file for details
