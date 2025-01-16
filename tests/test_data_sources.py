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
