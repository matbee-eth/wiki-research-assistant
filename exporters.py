"""
Export search results in various formats.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Union

import pyarrow as pa
import pyarrow.parquet as pq
import markdown
from bs4 import BeautifulSoup

class SearchResultExporter:
    """Handles export of search results in various formats."""
    
    def __init__(self):
        self.export_dir = Path("exports")
        self.export_dir.mkdir(exist_ok=True)
        
    def _generate_filename(self, query: str, format: str) -> str:
        """Generate a filename based on query and format."""
        # Sanitize query for filename
        safe_query = "".join(c if c.isalnum() else "_" for c in query)[:50]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{safe_query}_{timestamp}.{format.lower()}"
        
    def _format_result_html(self, results: List[Dict[str, Any]], query: str) -> str:
        """Format search results as HTML."""
        html = f"""
        <html>
        <head>
            <title>Search Results: {query}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 2em; }}
                .result {{ margin-bottom: 2em; padding: 1em; border: 1px solid #ddd; }}
                .title {{ color: #2c5282; margin-bottom: 0.5em; }}
                .score {{ color: #718096; font-size: 0.9em; }}
                .metadata {{ color: #4a5568; margin: 0.5em 0; }}
                .content {{ line-height: 1.5; }}
            </style>
        </head>
        <body>
            <h1>Search Results for: {query}</h1>
            <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        """
        
        for result in results:
            html += f"""
            <div class="result">
                <h2 class="title">{result.get('title', 'Untitled')}</h2>
                <div class="score">Relevance Score: {result.get('score', 'N/A')}</div>
                <div class="metadata">
                    <p>Source: {result.get('source', 'Unknown')}</p>
                    <p>URL: <a href="{result.get('url', '#')}">{result.get('url', 'No URL')}</a></p>
                </div>
                <div class="content">{result.get('content', 'No content available')}</div>
            </div>
            """
            
        html += "</body></html>"
        return html
        
    def _format_result_markdown(self, results: List[Dict[str, Any]], query: str) -> str:
        """Format search results as Markdown."""
        md = f"# Search Results for: {query}\n\n"
        md += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        for result in results:
            md += f"## {result.get('title', 'Untitled')}\n\n"
            md += f"**Relevance Score**: {result.get('score', 'N/A')}\n\n"
            md += f"**Source**: {result.get('source', 'Unknown')}\n\n"
            md += f"**URL**: {result.get('url', 'No URL')}\n\n"
            md += f"### Content\n\n{result.get('content', 'No content available')}\n\n"
            md += "---\n\n"
            
        return md

    def export(self, results: List[Dict[str, Any]], query: str, format: str) -> str:
        """
        Export search results in the specified format.
        
        Args:
            results: List of search result dictionaries
            query: Original search query
            format: Export format (html, pdf, parquet, markdown)
            
        Returns:
            Path to the exported file
        """
        format = format.lower()
        filename = self._generate_filename(query, format)
        output_path = self.export_dir / filename
        
        if format == "html":
            html_content = self._format_result_html(results, query)
            output_path.write_text(html_content, encoding='utf-8')
            
        elif format == "pdf":
            # For PDF export, we'll generate HTML and let the frontend handle the conversion
            # This avoids system-level dependencies and compatibility issues
            html_content = self._format_result_html(results, query)
            pdf_html_path = self.export_dir / f"{filename}.html"
            pdf_html_path.write_text(html_content, encoding='utf-8')
            return str(pdf_html_path)
            
        elif format == "parquet":
            # Convert results to a format suitable for Parquet
            data = {
                'query': [query] * len(results),
                'title': [r.get('title', '') for r in results],
                'score': [r.get('score', 0.0) for r in results],
                'source': [r.get('source', '') for r in results],
                'url': [r.get('url', '') for r in results],
                'content': [r.get('content', '') for r in results],
                'export_time': [datetime.now()] * len(results)
            }
            table = pa.Table.from_pydict(data)
            pq.write_table(table, str(output_path))
            
        elif format == "markdown":
            md_content = self._format_result_markdown(results, query)
            output_path.write_text(md_content, encoding='utf-8')
            
        else:
            raise ValueError(f"Unsupported export format: {format}")
            
        return str(output_path)
