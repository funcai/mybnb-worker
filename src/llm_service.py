# src/llm_service.py

import os
import json
import re
import torch
from transformers import pipeline
from typing import List

# Disable TorchDynamo completely to avoid compatibility issues with Gemma-3n
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# Additional TorchDynamo configuration
torch._dynamo.reset()
torch._dynamo.config.disable = True

# Global pipeline instance
HF_MODEL = "google/gemma-2-2b-it"
pipe = None


def initialize_pipeline():
    """Initialize the HuggingFace pipeline once."""
    global pipe
    if pipe is None:
        print("Initializing HuggingFace pipeline...")
        pipe = pipeline(
            "text-generation",
            model=HF_MODEL,
            device="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        # Fix padding side and pad token for batch generation
        if hasattr(pipe.tokenizer, 'padding_side'):
            pipe.tokenizer.padding_side = 'left'
        if pipe.tokenizer.pad_token is None:
            pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
        print("Pipeline initialized successfully!")
    return pipe


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
        outputs = pipe(
            formatted_prompts,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.1,
            batch_size=len(formatted_prompts),
            pad_token_id=pipe.tokenizer.eos_token_id
        )
        
        if debug:
            print(f"Pipeline output type: {type(outputs)}")
            print(f"Pipeline output length: {len(outputs)}")
            if outputs:
                print(f"First output type: {type(outputs[0])}")
                print(f"First output: {outputs[0]}")
        
    except Exception as e:
        print(f"ERROR in pipeline generation: {e}")
        return ["Error in generation"] * len(prompts)
    
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
            responses.append("Error extracting response")
    
    if debug:
        print(f"Final responses count: {len(responses)}")
        print(f"=== END BATCH_GENERATE DEBUG ===\n")
    
    return responses
