import os
import hashlib
import subprocess
import tempfile
import logging
from typing import List, Dict, Optional, Set
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModel
import open_clip
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageCacheMulti:
    """
    Parallel vision token caching service that downloads images and caches vision-encoded tokens as .pt files.
    Supports different vision encoders based on the model type (SmolVLM or Gemma).
    """
    
    def __init__(self, cache_dir: str = None, model_name: str = "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit"):
        """
        Initialize the parallel vision token cache.
        
        Args:
            cache_dir: Directory to store cached vision tokens. Defaults to '../tokens-cache' relative to this file.
            model_name: Vision model name to determine the correct encoder.
        """
        if cache_dir is None:
            # Default to tokens-cache directory in project root
            base_dir = os.path.dirname(os.path.dirname(__file__))
            cache_dir = os.path.join(base_dir, "tokens-cache")
        
        self.cache_dir = cache_dir
        self.model_name = model_name
        self._cached_urls = set()  # In-memory cache of URLs that are already cached
        self._vision_model = None
        self._processor = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            Cache filename with .pt extension for vision tokens
        """
        url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
        return f"{url_hash}.pt"
    
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
    
    def _initialize_vision_encoder(self):
        """
        Initialize the vision encoder based on the model type.
        """
        if self._vision_model is not None:
            return  # Already initialized
        
        logger.info(f"Initializing vision encoder for model: {self.model_name}")
        
        try:
            if "SmolVLM" in self.model_name:
                # For SmolVLM, we need to use the full multimodal model
                # because the vision and text components are tightly integrated
                from transformers import AutoModelForVision2Seq
                
                self._processor = AutoProcessor.from_pretrained(self.model_name)
                # Load the full multimodal model
                self._vision_model = AutoModelForVision2Seq.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto"
                ).eval()
                logger.info("SmolVLM multimodal model initialized")
                
            elif "gemma-3n" in self.model_name.lower():
                # For Gemma 3n models, use the full multimodal model with MobileNet-V5-300M vision encoder
                from transformers import AutoModelForVision2Seq
                
                self._processor = AutoProcessor.from_pretrained(self.model_name)
                # Load the full Gemma 3n multimodal model
                self._vision_model = AutoModelForVision2Seq.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto"
                ).eval()
                logger.info(f"Gemma 3n multimodal model initialized with MobileNet-V5-300M vision encoder: {self.model_name}")
                
            elif "gemma" in self.model_name.lower():
                # For other Gemma models, use open_clip approach
                self._vision_model, _, self._processor = open_clip.create_model_and_transforms(
                    "ViT-B-32", pretrained="openai"
                )
                self._vision_model = self._vision_model.visual.eval().to(self._device, torch.bfloat16)
                logger.info("Gemma vision encoder (OpenCLIP) initialized")
                
            else:
                # Default to OpenCLIP for unknown models
                self._vision_model, _, self._processor = open_clip.create_model_and_transforms(
                    "ViT-B-32", pretrained="openai"
                )
                self._vision_model = self._vision_model.visual.eval().to(self._device, torch.bfloat16)
                logger.info("Default vision encoder (OpenCLIP) initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize vision encoder: {e}")
            raise
    
    def _encode_image_to_tokens(self, image_path: str) -> torch.Tensor:
        """
        Encode an image to vision tokens.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Vision tokens as a tensor
        """
        self._initialize_vision_encoder()
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            if "SmolVLM" in self.model_name or "gemma-3n" in self.model_name.lower():
                # For SmolVLM and Gemma 3n, we use the full multimodal model with a dummy text prompt
                # to extract vision embeddings from the integrated pipeline
                
                # Create a minimal text prompt to trigger vision processing
                if "SmolVLM" in self.model_name:
                    dummy_text = "<image>\nDescribe this image."
                else:  # Gemma 3n
                    dummy_text = "<image>\nWhat is in this image?"
                
                # Process both image and text together
                inputs = self._processor(
                    text=dummy_text,
                    images=image,
                    return_tensors="pt"
                )
                
                # Move inputs to device
                inputs = {k: v.to(self._vision_model.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
                
                with torch.inference_mode():
                    # Get the model's internal representations
                    # We'll extract vision features from the model's forward pass
                    outputs = self._vision_model.generate(
                        **inputs,
                        max_new_tokens=1,  # Minimal generation to get vision embeddings
                        output_hidden_states=True,
                        return_dict_in_generate=True
                    )
                    
                    # Extract vision embeddings from the hidden states
                    # The vision information is embedded in the early layers
                    if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                        # Take the first layer's hidden states and pool them
                        first_layer_states = outputs.hidden_states[0][0]  # [batch, seq, hidden]
                        # Pool the sequence dimension to get a fixed-size representation
                        vision_tokens = first_layer_states.mean(dim=1)  # [batch, hidden]
                    else:
                        # Fallback: use a simple approach with model embeddings
                        # Get input embeddings which include vision information
                        input_embeds = self._vision_model.get_input_embeddings()
                        if hasattr(inputs, 'input_ids'):
                            vision_tokens = input_embeds(inputs['input_ids']).mean(dim=1)
                        else:
                            # Create a dummy embedding with appropriate size
                            hidden_size = 768 if "SmolVLM" in self.model_name else 2048  # Gemma 3n has larger hidden size
                            vision_tokens = torch.zeros(1, hidden_size, dtype=torch.bfloat16, device=self._vision_model.device)
                    
            else:
                # Use OpenCLIP processor
                image_tensor = self._processor(image).unsqueeze(0).to(self._device, torch.bfloat16)
                
                with torch.inference_mode():
                    vision_tokens = self._vision_model(image_tensor)
            
            return vision_tokens.cpu()  # Move back to CPU for storage
            
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            raise
    
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
        Process a downloaded image: encode to vision tokens and save as .pt file.
        
        Args:
            temp_path: Path to temporary downloaded file
            final_path: Final cache path for vision tokens (.pt file)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create a temporary processed image for encoding
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_processed:
                processed_path = temp_processed.name
            
            try:
                # Open and resize the image (same preprocessing as before)
                with Image.open(temp_path) as img:
                    # Convert to RGB if necessary (handles RGBA, P, etc.)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Resize while maintaining aspect ratio
                    img.thumbnail((256, 256), Image.Resampling.LANCZOS)
                    
                    # Create a new 256x256 black background image
                    padded_img = Image.new('RGB', (256, 256), (0, 0, 0))  # Black background
                    
                    # Calculate position to center the resized image
                    x_offset = (256 - img.width) // 2
                    y_offset = (256 - img.height) // 2
                    
                    # Paste the resized image onto the black background
                    padded_img.paste(img, (x_offset, y_offset))
                    
                    # Save processed image temporarily
                    padded_img.save(processed_path, 'JPEG', quality=85, optimize=True)
                
                # Encode the processed image to vision tokens
                vision_tokens = self._encode_image_to_tokens(processed_path)
                
                # Save vision tokens as .pt file
                torch.save(vision_tokens, final_path)
                
                return True
                
            finally:
                # Clean up temporary processed image
                if os.path.exists(processed_path):
                    os.unlink(processed_path)
            
        except Exception as e:
            logger.error(f"Failed to process image {temp_path}: {e}")
            return False
    
    def get_cached_image_paths(self, urls: List[str], max_parallel: int = 32) -> Dict[str, Optional[str]]:
        """
        Get local paths for multiple image URLs as vision token files. Downloads, processes, and caches vision tokens in parallel if not already cached.
        
        Args:
            urls: List of image URLs
            max_parallel: Maximum number of parallel downloads (default: 32)
            
        Returns:
            Dictionary mapping URLs to local .pt file paths containing vision tokens (None if failed)
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
    Convenience function to get cached vision token paths for multiple URLs.
    
    Args:
        urls: List of image URLs
        max_parallel: Maximum number of parallel downloads (default: 32)
        
    Returns:
        Dictionary mapping URLs to local .pt file paths containing vision tokens (None if failed)
    """
    return image_cache_multi.get_cached_image_paths(urls, max_parallel)
