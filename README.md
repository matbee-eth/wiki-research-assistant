# Wiki Research Assistant

A sophisticated research assistant powered by AI that helps users conduct thorough research using Wikipedia data with fact-checking capabilities.

[Screencast from 2025-01-17 01:54:51 PM.webm](https://github.com/user-attachments/assets/75748da0-f24f-4095-94f9-bbda81a75427)

## Key Features

- **Advanced Query Processing**: Intelligent decomposition of complex research queries into focused sub-queries for comprehensive coverage
- **Real-time Fact Checking**: Automated claim generation and validation against source documents
- **Interactive Streaming Interface**: Real-time progress updates and interactive chat interface built with Streamlit
- **Pipeline Architecture**: Modular and extensible pipeline system for processing research queries
- **LLM Integration**: Leverages state-of-the-art language models for query analysis and fact checking

## Technical Architecture

### Core Components

1. **Query Processing Pipeline**
   - Query analysis and decomposition
   - Semantic search optimization
   - Sub-query generation for comprehensive coverage

2. **Fact Checking System**
   - Automated claim generation from queries
   - Real-time claim validation against source documents
   - Confidence scoring and evidence tracking

3. **Stream Interface**
   - Real-time progress tracking
   - Interactive chat interface
   - Dynamic result visualization
   - Downloadable search results

4. **Data Sources**
   - Wikipedia integration
   - Efficient search and retrieval
   - Document processing and analysis

## Technology Stack

- **Backend**: Python with asyncio for concurrent processing
- **Frontend**: Streamlit for interactive UI
- **LLM Integration**: Custom LLM manager for model interactions
- **Data Visualization**: Plotly for interactive charts and visualizations
- **Logging**: Comprehensive logging system with configurable levels

## System Requirements

- Python 3.8+
- Required environment variables (see `.env.example`)
- Sufficient memory for LLM operations

## Usage

1. Set up environment variables
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python app.py`
4. Access the interface through your web browser

## Pipeline Flow

1. Query Analysis → Decomposition → Search → Fact Checking
2. Real-time result streaming and updates
3. Interactive chat interface for refinements
4. Export capabilities for search results

## Error Handling

- Comprehensive logging system
- Graceful error recovery
- User-friendly error messages
- Debug mode for development

## Search Capabilities

- Semantic search
- Query decomposition
- Real-time results streaming
- Fact validation
- Result export functionality

## Chat Interface

- Interactive query refinement
- Real-time response streaming
- Thread management
- Progress visualization

## Data Management

- Efficient data processing
- Result caching
- Export functionality
- Session state management

## Contributing

Feel free to submit issues and pull requests for:
- New features
- Bug fixes
- Documentation improvements
- Performance enhancements

## License

MIT License - See LICENSE file for details
