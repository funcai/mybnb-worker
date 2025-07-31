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
HF_MODEL = "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit"
VISION_MODEL = "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit"  # Using same model for both text and vision
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
            batch_size=50  # Increased batch size for better GPU utilization
        )
        print("Vision pipeline initialized successfully on GPU! (Multimodal model)")
    return vision_pipe


initialize_vision_pipeline()

def format_prompt_for_hf(prompt: str) -> str:
    """Format a prompt for HuggingFace text generation."""
    return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"


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
    Generate responses for multiple image-text pairs using gemma-3n-E4B multimodal model with true batching.
    
    Args:
        image_text_pairs: List of (image_path, messages) tuples where messages follow chat template format
        max_new_tokens: Maximum tokens to generate per response
        debug: Whether to print debug information
        
    Returns:
        List of generated text responses
    """
    
    if debug:
        print(f"Processing {len(image_text_pairs)} image-text pairs with multimodal model...")
    
    # Create Dataset for true batch processing
    dataset = create_vision_dataset(image_text_pairs)
    
    if debug:
        print(f"Created Dataset with {len(dataset)} items")
    
    # Process using Dataset-based batching for GPU parallelism
    responses = []
    
    import time
    start_time = time.time()
    
    print(f"🚀 Starting TRUE batch inference for {len(image_text_pairs)} items using Dataset...")
    print(f"📊 Using Dataset-based batching for GPU parallelism")
    
    try:
        inference_start = time.time()
        
        # Use the pipeline with IterableDataset for true batching
        # This enables proper DataLoader usage and GPU-parallel processing
        batch_size = 100  # Start with smaller batch size for testing
        batch_outputs = []
        
        print(f"🔍 Processing with batch_size={batch_size}")
        
        # Use KeyDataset to extract the 'text' key for pipeline processing
        from transformers.pipelines.pt_utils import KeyDataset
        
        # Process the dataset with batching using KeyDataset
        for output in vision_pipe(KeyDataset(dataset, "text"), batch_size=batch_size, max_new_tokens=max_new_tokens):
            batch_outputs.append(output)
            if debug:
                print(f"Batch output type: {type(output)}, Content: {output}")
        
        inference_time = time.time() - inference_start
        
        print(f"⚡ TRUE batch inference completed in {inference_time:.2f}s ({inference_time/len(image_text_pairs):.3f}s per item)")
        print(f"📊 Output type: {type(batch_outputs)}, Length: {len(batch_outputs)}")
        
        # Process the batch outputs
        for i, output in enumerate(batch_outputs):
            try:
                # Extract response from the correct output format
                if isinstance(output, dict) and "generated_text" in output:
                    generated_text = output["generated_text"]
                    response = str(generated_text).strip()
                elif isinstance(output, list) and len(output) > 0:
                    # Handle list of outputs
                    first_output = output[0]
                    if isinstance(first_output, dict) and "generated_text" in first_output:
                        response = str(first_output["generated_text"]).strip()
                    else:
                        response = str(first_output).strip()
                else:
                    response = str(output).strip()
                
                responses.append(response)
                
                if debug:
                    print(f"Response {i+1}: {response}")
                    
            except Exception as e:
                print(f"ERROR extracting response {i+1}: {e}")
                responses.append('{"response": "maybe"}')
                
    except Exception as e:
        print(f"❌ ERROR in batch processing: {e}")
        print("🔄 Falling back to sequential processing...")
        
        # Fallback to sequential processing if batch processing fails
        sequential_start = time.time()
        
        # Format messages for sequential processing
        formatted_messages = []
        for image_path, messages in image_text_pairs:
            formatted_message = []
            
            for msg in messages:
                if msg["role"] == "system":
                    formatted_message.append({
                        "role": "system",
                        "content": [{"type": "text", "text": msg["content"]}]
                    })
                elif msg["role"] == "user":
                    # Create user message with image and text
                    content = []
                    # Add image first
                    if isinstance(image_path, str):
                        content.append({"type": "image", "url": image_path})
                    else:
                        # For PIL images, we need to handle differently
                        content.append({"type": "image", "image": image_path})
                    
                    # Add text
                    content.append({"type": "text", "text": msg["content"]})
                    
                    formatted_message.append({
                        "role": "user", 
                        "content": content
                    })
            
            formatted_messages.append(formatted_message)
        
        for i, message in enumerate(formatted_messages):
            if debug:
                print(f"Processing item {i+1}/{len(formatted_messages)} (fallback)")
            
            try:
                output = vision_pipe(message, max_new_tokens=max_new_tokens)
                
                if isinstance(output, list) and len(output) > 0:
                    generated_text = output[0].get("generated_text", [])
                    if isinstance(generated_text, list) and len(generated_text) > 0:
                        last_message = generated_text[-1]
                        if isinstance(last_message, dict) and "content" in last_message:
                            response = last_message["content"].strip()
                        else:
                            response = str(last_message).strip()
                    else:
                        response = str(generated_text).strip()
                else:
                    response = str(output).strip()
                
                responses.append(response)
                
            except Exception as e:
                print(f"ERROR in fallback processing {i+1}: {e}")
                responses.append('{"response": "maybe"}')
        
        sequential_time = time.time() - sequential_start
        print(f"🔄 Sequential processing completed in {sequential_time:.2f}s ({sequential_time/len(image_text_pairs):.3f}s per item)")
    
    total_time = time.time() - start_time
    print(f"📊 Total processing time: {total_time:.2f}s for {len(image_text_pairs)} items")
    
    return responses
