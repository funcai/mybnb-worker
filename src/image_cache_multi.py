import os
import hashlib
import subprocess
import tempfile
from PIL import Image
from typing import List, Optional, Dict
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageCacheMulti:
    """
    Parallel image caching service that downloads, resizes, and caches multiple apartment images simultaneously.
    """
    
    def __init__(self, cache_dir: str = None):
        """
        Initialize the parallel image cache.
        
        Args:
            cache_dir: Directory to store cached images. Defaults to '../image-cache' relative to this file.
        """
        if cache_dir is None:
            # Default to image-cache directory in project root
            base_dir = os.path.dirname(os.path.dirname(__file__))
            cache_dir = os.path.join(base_dir, "image-cache")
        
        self.cache_dir = cache_dir
        self._cached_urls = set()  # In-memory cache of URLs that are already cached
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self):
        """Create cache directory if it doesn't exist."""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.info(f"Created cache directory: {self.cache_dir}")
    
    def _is_cached_in_memory(self, url: str) -> bool:
        """
        Check if URL is in the in-memory cache.
        
        Args:
            url: Image URL
            
        Returns:
            True if URL is known to be cached, False otherwise
        """
        return url in self._cached_urls
    
    def _add_to_memory_cache(self, url: str):
        """
        Add URL to the in-memory cache.
        
        Args:
            url: Image URL that has been successfully cached
        """
        self._cached_urls.add(url)
    
    def _check_file_exists_and_cache(self, url: str) -> bool:
        """
        Check if file exists on disk and update in-memory cache.
        
        Args:
            url: Image URL
            
        Returns:
            True if file exists, False otherwise
        """
        cache_path = self._get_cache_path(url)
        if os.path.exists(cache_path):
            self._add_to_memory_cache(url)
            return True
        return False
    
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
    
    def _download_images_parallel(self, url_to_temp_path: Dict[str, str], max_parallel: int = 32) -> Dict[str, bool]:
        """
        Download multiple images in parallel using curl.
        
        Args:
            url_to_temp_path: Dictionary mapping URLs to temporary file paths
            max_parallel: Maximum number of parallel downloads
            
        Returns:
            Dictionary mapping URLs to success status (True/False)
        """
        if not url_to_temp_path:
            return {}
        
        # Build curl command with parallel downloads
        curl_cmd = [
            'curl',
            '--parallel',
            '--parallel-max', str(max_parallel),
            '--fail',  # Fail silently on HTTP errors
            '--silent',  # Silent mode
            '--show-error',  # Show errors even in silent mode
            '--location',  # Follow redirects
            '--max-time', '30',  # 30 second timeout per URL
            '--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        ]
        
        # Add URL and output pairs
        for url, temp_path in url_to_temp_path.items():
            curl_cmd.extend(['-o', temp_path, url])
        
        logger.info(f"Starting parallel download of {len(url_to_temp_path)} images with max {max_parallel} concurrent downloads")
        
        try:
            # Run curl command
            result = subprocess.run(curl_cmd, capture_output=True, text=True, timeout=300)  # 5 minute total timeout
            
            # Check which downloads succeeded by checking if temp files exist and have content
            success_status = {}
            for url, temp_path in url_to_temp_path.items():
                if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                    success_status[url] = True
                    logger.debug(f"Successfully downloaded: {url}")
                else:
                    success_status[url] = False
                    logger.warning(f"Failed to download: {url}")
            
            successful_count = sum(success_status.values())
            logger.info(f"Parallel download completed: {successful_count}/{len(url_to_temp_path)} successful")
            
            if result.stderr:
                logger.debug(f"curl stderr: {result.stderr}")
            
            return success_status
            
        except subprocess.TimeoutExpired:
            logger.error("Parallel download timed out")
            return {url: False for url in url_to_temp_path.keys()}
        except Exception as e:
            logger.error(f"Parallel download failed: {e}")
            return {url: False for url in url_to_temp_path.keys()}
    
    def _process_downloaded_image(self, temp_path: str, final_path: str) -> bool:
        """
        Process a downloaded image: resize and save as optimized JPEG.
        
        Args:
            temp_path: Path to temporary downloaded file
            final_path: Final cache path for processed image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Open and resize the image
            with Image.open(temp_path) as img:
                # Convert to RGB if necessary (handles RGBA, P, etc.)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize while maintaining aspect ratio
                img.thumbnail((256, 256), Image.Resampling.LANCZOS)
                
                # Save as JPEG
                img.save(final_path, 'JPEG', quality=85, optimize=True)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to process image {temp_path}: {e}")
            return False
    
    def get_cached_image_paths(self, urls: List[str], max_parallel: int = 32) -> Dict[str, Optional[str]]:
        """
        Get local paths for multiple image URLs. Downloads and caches in parallel if not already cached.
        
        Args:
            urls: List of image URLs
            max_parallel: Maximum number of parallel downloads (default: 32)
            
        Returns:
            Dictionary mapping URLs to local file paths (None if failed)
        """
        if not urls:
            return {}
        
        # Filter out empty/invalid URLs
        valid_urls = [url for url in urls if url and url.strip()]
        if not valid_urls:
            logger.warning("No valid URLs provided")
            return {url: None for url in urls}
        
        result = {}
        urls_to_download = []
        
        # Check which images are already cached
        for url in valid_urls:
            # First check in-memory cache to avoid filesystem calls
            if self._is_cached_in_memory(url):
                cache_path = self._get_cache_path(url)
                logger.debug(f"Using cached image (from memory): {cache_path}")
                result[url] = cache_path
            # If not in memory cache, check filesystem and update memory cache
            elif self._check_file_exists_and_cache(url):
                cache_path = self._get_cache_path(url)
                logger.debug(f"Using cached image (from disk): {cache_path}")
                result[url] = cache_path
            else:
                urls_to_download.append(url)
        
        # Handle invalid URLs
        for url in urls:
            if not url or not url.strip():
                result[url] = None
        
        if not urls_to_download:
            logger.info(f"All {len(valid_urls)} images already cached")
            return result
        
        logger.info(f"Need to download {len(urls_to_download)} images")
        
        # Create temporary directory for downloads
        with tempfile.TemporaryDirectory() as temp_dir:
            # Prepare URL to temp path mapping
            url_to_temp_path = {}
            for url in urls_to_download:
                temp_filename = self._get_cache_filename(url) + ".temp"
                temp_path = os.path.join(temp_dir, temp_filename)
                url_to_temp_path[url] = temp_path
            
            # Download images in parallel
            download_status = self._download_images_parallel(url_to_temp_path, max_parallel)
            
            # Process successfully downloaded images
            for url in urls_to_download:
                temp_path = url_to_temp_path[url]
                final_path = self._get_cache_path(url)
                
                if download_status.get(url, False):
                    # Process and cache the image
                    if self._process_downloaded_image(temp_path, final_path):
                        result[url] = final_path
                        self._add_to_memory_cache(url)  # Add to in-memory cache
                        logger.info(f"Successfully cached image: {url}")
                    else:
                        result[url] = None
                        logger.error(f"Failed to process image: {url}")
                else:
                    result[url] = None
                    logger.error(f"Failed to download image: {url}")
        
        successful_count = sum(1 for path in result.values() if path is not None)
        logger.info(f"Batch processing completed: {successful_count}/{len(valid_urls)} images successfully cached")
        
        return result
    


# Global instance for easy access
image_cache_multi = ImageCacheMulti()


def get_cached_image_paths(urls: List[str], max_parallel: int = 32) -> Dict[str, Optional[str]]:
    """
    Convenience function to get cached image paths for multiple URLs.
    
    Args:
        urls: List of image URLs
        max_parallel: Maximum number of parallel downloads (default: 32)
        
    Returns:
        Dictionary mapping URLs to local file paths (None if failed)
    """
    return image_cache_multi.get_cached_image_paths(urls, max_parallel)
