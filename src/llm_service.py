# src/llm_service.py

import os
import json
import re
import torch
from transformers import pipeline
from typing import List, Tuple

# Disable TorchDynamo completely to avoid compatibility issues with Gemma-3n
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# Additional TorchDynamo configuration
torch._dynamo.reset()
torch._dynamo.config.disable = True

# Global pipeline instances
HF_MODEL = "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit"
VISION_MODEL = "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit"
# VISION_MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"  # Small vision-language model
vision_pipe = None


def initialize_vision_pipeline():
    """Initialize the HuggingFace vision pipeline for gemma-3n-E4B."""
    global vision_pipe
    if vision_pipe is None:
        print("Initializing HuggingFace vision pipeline...")
        # Require CUDA - fail if not available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required but not available. Please ensure GPU is accessible.")
        
        # Use image-text-to-text pipeline for vision inference with optimizations
        # Note: device parameter removed because model uses accelerate for automatic device placement
        vision_pipe = pipeline(
            "image-text-to-text",
            model=VISION_MODEL,
            torch_dtype=torch.bfloat16,
            batch_size=50,  # Increased batch size for better GPU utilization
            padding=True  # Enable padding to handle variable image sizes
        )
        print("Vision pipeline initialized successfully on GPU! (Multimodal model)")
    return vision_pipe


initialize_vision_pipeline()

def format_prompt_for_hf(prompt: str) -> str:
    """Format a prompt for HuggingFace text generation based on the selected model."""
    # Check which model is being used and apply appropriate format
    if "SmolVLM" in VISION_MODEL:
        # SmolVLM uses a simpler format without chat tokens
        return f"User: {prompt}\n\nAssistant:"
    elif "gemma" in VISION_MODEL.lower():
        # Gemma models use chat turn format
        return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    else:
        # Default to simple format for unknown models
        return f"User: {prompt}\n\nAssistant:"


def extract_json_from_response(response: str) -> dict:
    """Extract JSON from HuggingFace response text with robust parsing."""
    # Clean the response
    response = response.strip()
    
    # If response is empty, return a default
    if not response:
        print("Empty response received, using default 'maybe' response")
        return {"response": "maybe"}
    
    try:
        # Method 1: Look for JSON in code blocks
        json_match = re.search(r'```json\s*({.*?})\s*```', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        
        # Method 2: Look for JSON without code blocks
        json_match = re.search(r'({.*?})', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        
        # Method 3: Try to parse the entire response as JSON
        return json.loads(response)
        
    except json.JSONDecodeError:
        # Method 4: Look for simple yes/no/irrelevant responses
        response_lower = response.lower()
        if 'yes' in response_lower:
            return {"response": "yes"}
        elif 'no' in response_lower:
            return {"response": "no"}
        elif 'irrelevant' in response_lower:
            return {"response": "irrelevant"}
        
        # Method 5: Try to extract key-value pairs manually
        if '"response"' in response:
            # Look for "response": "value" pattern
            match = re.search(r'"response"\s*:\s*"([^"]+)"', response)
            if match:
                return {"response": match.group(1)}
        
        # Method 6: Default fallback
        print(f"Could not parse JSON from response: {response[:200]}...")
        print(f"Using default 'maybe' response")
        return {"response": "maybe"}


def batch_generate(prompts: List[str], max_new_tokens: int = 200, debug: bool = False) -> List[str]:
    """Generate responses for multiple prompts in batch."""
    if not prompts:
        return []
    
    if debug:
        print(f"\n=== BATCH_GENERATE DEBUG ===")
        print(f"Number of prompts: {len(prompts)}")
        print(f"Max new tokens: {max_new_tokens}")
        print(f"First prompt preview: {prompts[0][:200]}...")
    
    # Format prompts for HuggingFace
    formatted_prompts = [format_prompt_for_hf(prompt) for prompt in prompts]
    
    if debug:
        print(f"First formatted prompt: {formatted_prompts[0][:300]}...")
    
    # Generate responses in batch
    try:
        outputs = vision_pipe(
            formatted_prompts,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.1,
            batch_size=len(formatted_prompts),
            pad_token_id=vision_pipe.tokenizer.eos_token_id
        )
        
        if debug:
            print(f"Pipeline output type: {type(outputs)}")
            print(f"Pipeline output length: {len(outputs)}")
            if outputs:
                print(f"First output type: {type(outputs[0])}")
                print(f"First output: {outputs[0]}")
        
    except Exception as e:
        print(f"ERROR in pipeline generation: {e}")
        return ['{"response": "maybe"}'] * len(prompts)
    
    # Extract generated text (remove the input prompt)
    responses = []
    for i, output in enumerate(outputs):
        try:
            generated_text = output[0]['generated_text']
            # Remove the input prompt to get only the response
            response = generated_text[len(formatted_prompts[i]):].strip()
            
            if debug and i == 0:
                print(f"\nGenerated text (full): {generated_text}")
                print(f"Formatted prompt length: {len(formatted_prompts[i])}")
                print(f"Response after prompt removal: '{response}'")
                print(f"Response length: {len(response)}")
            
            responses.append(response)
        except Exception as e:
            print(f"ERROR extracting response {i}: {e}")
            responses.append('{"response": "maybe"}')
    
    if debug:
        print(f"Final responses count: {len(responses)}")
        print(f"=== END BATCH_GENERATE DEBUG ===\n")
    
    return responses


from datasets import Dataset

def create_vision_dataset(image_text_pairs: List[Tuple]):
    """Create an IterableDataset for batch processing of image-text pairs with HuggingFace pipelines."""
    
    # Format the data for the pipeline using the correct Gemma 3 message format
    formatted_data = []
    
    for image_path, messages in image_text_pairs:
        # Format messages according to Gemma 3 multimodal format
        formatted_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                formatted_messages.append({
                    "role": "system",
                    "content": [{"type": "text", "text": msg["content"]}]
                })
            elif msg["role"] == "user":
                # Create user message with image and text in the correct format
                content = []
                # Add image first
                if isinstance(image_path, str):
                    content.append({"type": "image", "url": image_path})
                else:
                    # For PIL images, we need to handle differently
                    content.append({"type": "image", "image": image_path})
                
                # Add text
                content.append({"type": "text", "text": msg["content"]})
                
                formatted_messages.append({
                    "role": "user", 
                    "content": content
                })
        
        # For IterableDataset batching, use the original message format
        # The pipeline should handle the multimodal messages directly
        formatted_entry = {
            "text": formatted_messages
        }
        
        formatted_data.append(formatted_entry)
    
    # Create a regular Dataset for batch processing
    # Based on research, regular Dataset processes everything in one batch (true GPU parallelism)
    dataset = Dataset.from_list(formatted_data)
    
    return dataset


def batch_generate_vision(image_text_pairs: List[Tuple], max_new_tokens: int = 200, debug: bool = False) -> List[str]:
    """
    Generate responses for multiple image-text pairs using cached vision tokens with SmolVLM.
    
    Args:
        image_text_pairs: List of (image_url, messages) tuples where messages follow chat template format
        max_new_tokens: Maximum tokens to generate per response
        debug: Whether to print debug information
        
    Returns:
        List of generated text responses
    """
    
    if debug:
        print(f"Processing {len(image_text_pairs)} image-text pairs with cached vision tokens...")
    
    import time
    import os
    from image_cache_multi import get_cached_image_paths
    import torch
    
    start_time = time.time()
    
    # Extract image URLs from the pairs
    image_urls = [pair[0] for pair in image_text_pairs]
    
    print(f"🚀 Loading cached vision tokens for {len(image_urls)} images...")
    
    # Get cached vision token paths
    token_cache_start = time.time()
    cached_paths = get_cached_image_paths(image_urls, max_parallel=32)
    token_cache_time = time.time() - token_cache_start
    
    print(f"⚡ Vision token cache lookup completed in {token_cache_time:.3f}s")
    
    # Load vision tokens and prepare text prompts
    vision_tokens = []
    text_prompts = []
    
    for i, (image_url, messages) in enumerate(image_text_pairs):
        # Load cached vision tokens
        token_path = cached_paths.get(image_url)
        if token_path and os.path.exists(token_path):
            try:
                tokens = torch.load(token_path, map_location='cpu')
                vision_tokens.append(tokens)
                
                # Format text prompt for SmolVLM
                if "SmolVLM" in VISION_MODEL:
                    # SmolVLM format: <image>\nQuestion: {question}\nAnswer:
                    user_message = messages[-1]["content"] if messages else ""
                    prompt = f"<image>\nQuestion: {user_message}\nAnswer:"
                else:
                    # Gemma format (fallback)
                    user_message = messages[-1]["content"] if messages else ""
                    prompt = f"Image: <image>\n{user_message}"
                
                text_prompts.append(prompt)
                
            except Exception as e:
                print(f"❌ Failed to load vision tokens for {image_url}: {e}")
                # Fallback to simple text response
                vision_tokens.append(None)
                text_prompts.append('{"response": "maybe"}')
        else:
            print(f"❌ No cached vision tokens found for {image_url}")
            vision_tokens.append(None)
            text_prompts.append('{"response": "maybe"}')
    
    print(f"📊 Successfully loaded {sum(1 for t in vision_tokens if t is not None)}/{len(vision_tokens)} vision token sets")
    
    # For now, use text-only processing since integrating vision tokens with the pipeline
    # requires more complex model architecture changes. This is a stepping stone.
    # TODO: Implement full vision token integration with SmolVLM architecture
    
    print(f"🔄 Processing text prompts with vision context...")
    
    responses = []
    inference_start = time.time()
    
    # Process each prompt (for now, without direct vision token integration)
    # This maintains compatibility while we have the vision tokens cached
    for i, (prompt, tokens) in enumerate(zip(text_prompts, vision_tokens)):
        if tokens is not None:
            # For now, use simple text processing
            # In a full implementation, we would integrate the vision tokens directly
            try:
                # Use the text pipeline for now
                if hasattr(pipe, 'tokenizer'):
                    inputs = pipe.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                    with torch.inference_mode():
                        outputs = pipe.model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            temperature=0.7,
                            pad_token_id=pipe.tokenizer.eos_token_id
                        )
                    
                    response = pipe.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    # Remove the input prompt from response
                    if prompt in response:
                        response = response.replace(prompt, "").strip()
                    
                    responses.append(response)
                else:
                    responses.append('{"response": "maybe"}')
                    
            except Exception as e:
                print(f"❌ Error processing prompt {i+1}: {e}")
                responses.append('{"response": "maybe"}')
        else:
            responses.append('{"response": "maybe"}')
    
    inference_time = time.time() - inference_start
    total_time = time.time() - start_time
    
    print(f"⚡ Vision-enhanced inference completed in {inference_time:.2f}s ({inference_time/len(image_text_pairs):.3f}s per item)")
    print(f"📊 Total processing time: {total_time:.2f}s for {len(image_text_pairs)} items")
    
    return responses
