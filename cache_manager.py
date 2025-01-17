import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import timedelta

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self, cache_dir: str = "cache", cache_expiry: int = 86400):
        """Initialize cache manager.
        
        Args:
            cache_dir: Directory to store cache files
            cache_expiry: Cache expiry time in seconds (default 24 hours)
        """
        self.cache_dir = cache_dir
        self.cache_expiry = cache_expiry
        self._article_cache = {}  # In-memory cache
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(exist_ok=True)
        self._load_cache()

    def _load_cache(self):
        """Load cached articles from disk"""
        try:
            cache_file = self._cache_dir / "article_cache.json"
            if cache_file.exists():
                with open(cache_file, "r") as f:
                    self._article_cache = json.load(f)
                logger.info(f"Loaded {len(self._article_cache)} cached articles")
        except Exception as e:
            logger.error(f"Error loading cache: {str(e)}")
            self._article_cache = {}

    def _save_cache(self):
        """Save article cache to disk"""
        try:
            cache_file = self._cache_dir / "article_cache.json"
            with open(cache_file, "w") as f:
                json.dump(self._article_cache, f)
            logger.info(f"Saved {len(self._article_cache)} articles to cache")
        except Exception as e:
            logger.error(f"Error saving cache: {str(e)}")

    def cache_article(self, article_id: str, article_data: Dict[str, Any]) -> None:
        """Cache an article's data."""
        try:
            if not article_id or not article_data:
                return
                
            # Add cache metadata
            article_data['cached_at'] = datetime.now().isoformat()
            
            # Create cache directory if it doesn't exist
            if not os.path.exists(self._cache_dir):
                os.makedirs(self._cache_dir)
                
            # Cache file path
            cache_file = os.path.join(self._cache_dir, f"{article_id}.json")
            
            # Write to cache file
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(article_data, f, ensure_ascii=False, indent=2)
                
            logger.debug(f"Cached article {article_id}")
            
        except Exception as e:
            logger.error(f"Error caching article {article_id}: {str(e)}")

    def get_cached_article(self, article_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a cached article if it exists and is not expired."""
        try:
            if not article_id:
                return None
                
            cache_file = os.path.join(self._cache_dir, f"{article_id}.json")
            
            # Check if cache exists
            if not os.path.exists(cache_file):
                return None
                
            # Read cache file
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                
            # Check cache expiration
            cached_at = cached_data.get('cached_at')
            if cached_at:
                cached_time = datetime.fromisoformat(cached_at)
                age = datetime.now() - cached_time
                
                if age.total_seconds() > self.cache_expiry:
                    logger.debug(f"Cache expired for article {article_id}")
                    return None
                    
            return cached_data
            
        except Exception as e:
            logger.error(f"Error retrieving cached article {article_id}: {str(e)}")
            return None
