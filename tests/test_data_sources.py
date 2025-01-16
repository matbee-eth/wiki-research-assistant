import pytest
import asyncio
from data_sources import DataSources

@pytest.mark.asyncio
async def test_get_full_wikipedia_page():
    async with DataSources() as ds:
        # Test with page title
        result = await ds.get_full_wikipedia_page("Python (programming language)")
        
        print("\n" + "="*120)
        print("TEMPLATES")
        print("="*120)
        
        # if "templates" in result:
        #     for i, template in enumerate(result["templates"], 1):
        #         print(f"\nTemplate {i}:")
        #         print(template)
        
        print("\n" + "="*120)
        print("SECTIONS")
        print("="*120)
        
        for section in result["sections"]:
            level_indent = "  " * section["level"]
            print(f"\n{level_indent}Level {section['level']}: {section['title']}")
            content_preview = section["content"][:200] + "..." if len(section["content"]) > 200 else section["content"]
            print(f"{level_indent}Preview: {content_preview}")
        
        print("\n" + "="*120)
        print(f"Total templates: {len(result.get('templates', []))}")
        print(f"Total sections: {len(result['sections'])}")
        print(f"Total content length: {len(result['content']):,} characters")
        print("="*120 + "\n")
        
        # Check basic metadata
        assert result is not None
        assert isinstance(result, dict)
        assert "title" in result
        assert "content" in result
        assert "pageid" in result
        assert "url" in result
        
        # Verify content is substantial (full article should be much longer than intro)
        assert len(result["content"]) > 1000
        
        # Check specific content markers that should be in a full Python article
        content = result["content"].lower()
        assert "programming language" in content
        assert "guido van rossum" in content
        
        # Test with non-existent page
        with pytest.raises(ValueError, match="Page '.*' not found"):
            await ds.get_full_wikipedia_page("ThisPageDefinitelyDoesNotExist12345")
        
        # Test with empty input
        with pytest.raises(ValueError, match="Page identifier cannot be empty"):
            await ds.get_full_wikipedia_page("")
        
        # Test with page ID
        # Using Python article's page ID
        id_result = await ds.get_full_wikipedia_page(23862)
        assert id_result is not None
        assert isinstance(id_result, dict)
        assert "title" in id_result
        assert len(id_result["content"]) > 1000

@pytest.mark.asyncio
async def test_get_full_wikipedia_page_categories():
    async with DataSources() as ds:
        result = await ds.get_full_wikipedia_page("Python (programming language)")
        
        # Check categories
        assert "categories" in result
        assert isinstance(result["categories"], list)
        assert len(result["categories"]) > 0
        
        # At least one of these categories should be present
        expected_categories = [
            "Category:Class-based programming languages",
            "Category:Articles with example Python (programming language) code"
        ]
        found_category = any(cat in result["categories"] for cat in expected_categories)
        assert found_category, "Expected category not found in article categories"

@pytest.mark.asyncio
async def test_wiki_rendering():
    """Test wiki markup rendering functionality."""
    data_sources = DataSources()
    
    # Test sample with various wiki markup elements
    wiki_content = """
==Introduction==
This is a '''bold''' and ''italic'' text example.

===Section 1===
* List item 1
* List item 2
# Ordered item 1
# Ordered item 2

====Subsection====
Here's a [[Wikipedia|link]] and an external [https://example.com Example Link].

{{Template|param=value}}
<ref>Reference text</ref>

==References==
Some text with a <ref>Another reference</ref> and a {{citation needed}} template.
"""
    
    # Create a mock article data structure
    article_data = {
        'content': wiki_content,
        'sections': [
            {
                'title': 'Test Section',
                'content': "This is a '''test''' section with a [[link]]",
                'level': 2
            }
        ]
    }
    
    # Render the content
    rendered = data_sources.render_wiki_content(article_data)
    
    print("\nTESTING WIKI RENDERING")
    print("="*120)
    print("\nOriginal Wiki Content:")
    print("-"*80)
    print(wiki_content)
    print("\nRendered HTML Content:")
    print("-"*80)
    print(rendered['content'])
    print("\nRendered Section Content:")
    print("-"*80)
    print(rendered['sections'][0]['content'])
    
    # Verify the rendering quality
    assert '<h2>Introduction</h2>' in rendered['content'], "Header not properly rendered"
    assert '<strong>' in rendered['content'], "Bold text not properly rendered"
    assert '<em>' in rendered['content'], "Italic text not properly rendered"
    assert '<ul>' in rendered['content'], "Unordered list not properly rendered"
    assert '<ol>' in rendered['content'], "Ordered list not properly rendered"
    assert '<li>' in rendered['content'], "List items not properly rendered"
    assert '<a href=' in rendered['content'], "Links not properly rendered"
    assert '{{Template|param=value}}' not in rendered['content'], "Template not properly removed"
    assert '<ref>' not in rendered['content'], "References not properly removed"
    
    # Verify section rendering
    section_content = rendered['sections'][0]['content']
    assert '<strong>test</strong>' in section_content, "Section formatting not properly rendered"
    assert '<a href=' in section_content, "Section links not properly rendered"
