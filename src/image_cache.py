import os
import hashlib
import requests
from PIL import Image
from typing import Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageCache:
    """
    Image caching service that downloads, resizes, and caches apartment images.
    """
    
    def __init__(self, cache_dir: str = None):
        """
        Initialize the image cache.
        
        Args:
            cache_dir: Directory to store cached images. Defaults to '../image-cache' relative to this file.
        """
        if cache_dir is None:
            # Default to image-cache directory in project root
            base_dir = os.path.dirname(os.path.dirname(__file__))
            cache_dir = os.path.join(base_dir, "image-cache")
        
        self.cache_dir = cache_dir
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self):
        """Create cache directory if it doesn't exist."""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.info(f"Created cache directory: {self.cache_dir}")
    
    def _get_cache_filename(self, url: str) -> str:
        """
        Generate cache filename from URL using MD5 hash.
        
        Args:
            url: Image URL
            
        Returns:
            Cache filename with .jpg extension
        """
        url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
        return f"{url_hash}.jpg"
    
    def _get_cache_path(self, url: str) -> str:
        """
        Get full cache file path for a URL.
        
        Args:
            url: Image URL
            
        Returns:
            Full path to cached image file
        """
        filename = self._get_cache_filename(url)
        return os.path.join(self.cache_dir, filename)
    
    def _download_image(self, url: str, output_path: str) -> bool:
        """
        Download image from URL and save to output path.
        
        Args:
            url: Image URL to download
            output_path: Path to save the downloaded image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Save the raw image temporarily
            temp_path = output_path + ".temp"
            with open(temp_path, 'wb') as f:
                f.write(response.content)
            
            # Open and resize the image
            with Image.open(temp_path) as img:
                # Convert to RGB if necessary (handles RGBA, P, etc.)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize while maintaining aspect ratio
                img.thumbnail((256, 256), Image.Resampling.LANCZOS)
                
                # Save as JPEG
                img.save(output_path, 'JPEG', quality=85, optimize=True)
            
            # Remove temporary file
            os.remove(temp_path)
            
            logger.info(f"Downloaded and cached image: {url}")
            return True
            
        except requests.RequestException as e:
            logger.error(f"Failed to download image {url}: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to process image {url}: {e}")
            # Clean up temporary file if it exists
            temp_path = output_path + ".temp"
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False
    
    def get_local_image_path(self, url: str) -> Optional[str]:
        """
        Get local path for an image URL. Downloads and caches if not already cached.
        
        Args:
            url: Image URL
            
        Returns:
            Local file path if successful, None if failed
        """
        if not url or not url.strip():
            logger.warning("Empty or invalid URL provided")
            return None
        
        cache_path = self._get_cache_path(url)
        
        # Return cached path if file already exists
        if os.path.exists(cache_path):
            logger.debug(f"Using cached image: {cache_path}")
            return cache_path
        
        # Download and cache the image
        logger.info(f"Downloading image from: {url}")
        if self._download_image(url, cache_path):
            return cache_path
        else:
            return None
    
    def clear_cache(self):
        """Remove all cached images."""
        if os.path.exists(self.cache_dir):
            for filename in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            logger.info("Cache cleared")
    
    def get_cache_stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        if not os.path.exists(self.cache_dir):
            return {"total_files": 0, "total_size_mb": 0}
        
        total_files = 0
        total_size = 0
        
        for filename in os.listdir(self.cache_dir):
            file_path = os.path.join(self.cache_dir, filename)
            if os.path.isfile(file_path):
                total_files += 1
                total_size += os.path.getsize(file_path)
        
        return {
            "total_files": total_files,
            "total_size_mb": round(total_size / (1024 * 1024), 2)
        }


# Global instance for easy access
image_cache = ImageCache()


def get_cached_image_path(url: str) -> Optional[str]:
    """
    Convenience function to get cached image path.
    
    Args:
        url: Image URL
        
    Returns:
        Local file path if successful, None if failed
    """
    return image_cache.get_local_image_path(url)


# Example usage and testing
if __name__ == "__main__":
    # Test the image cache
    test_url = "https://wunderflatsng.blob.core.windows.net/imagesproduction/-wMzbcLRXiXh1Hn_NiBoh-large.jpg"
    
    print("Testing Image Cache...")
    print(f"Cache directory: {image_cache.cache_dir}")
    
    # Get cache stats before
    stats_before = image_cache.get_cache_stats()
    print(f"Cache stats before: {stats_before}")
    
    # Test downloading an image
    local_path = image_cache.get_local_image_path(test_url)
    if local_path:
        print(f"Successfully cached image at: {local_path}")
        print(f"File exists: {os.path.exists(local_path)}")
        
        # Test getting the same image again (should use cache)
        local_path2 = image_cache.get_local_image_path(test_url)
        print(f"Second call returned same path: {local_path == local_path2}")
    else:
        print("Failed to cache image")
    
    # Get cache stats after
    stats_after = image_cache.get_cache_stats()
    print(f"Cache stats after: {stats_after}")
