# src/my_server.py

# Disable TorchDynamo completely to avoid compatibility issues with Gemma-3n
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch

# Local imports from our project files
from agent_filter import load_apartment_df, filter_apartments
from llm_service import initialize_vision_pipeline
from query_processor import get_questions_and_filters_from_query
from apartment_scorer import score_apartments_batch, Apartment, ScoreDetail

# Additional TorchDynamo configuration
torch._dynamo.reset()
torch._dynamo.config.disable = True

app = FastAPI(title="My helper service")

BASE_DIR = os.path.dirname(__file__)
DATA_APARTMENTS = os.path.join(BASE_DIR, "apartments.json")

# --- Pydantic Models for API validation ---

class ProcessRequest(BaseModel):
    query: str

class ProcessResponse(BaseModel):
    apartments: List[Apartment]

# JSON schema for scoring validation (kept for reference)
scoring_schema = {
    "type": "object",
    "additionalProperties": False,
    "required": ["response"],
    "properties": {
        "response": {
            "type": "string",
            "enum": ["yes", "no", "irrelevant"],
            "description": "Whether the apartment matches the question. Answer 'yes' if it matches, 'no' if it doesn't match, or 'irrelevant' if the question doesn't apply to this apartment."
        }
    }
}

@app.on_event("startup")
def startup_event():
    """Initialize the pipeline on startup."""
    print("Starting up the application...")
    initialize_vision_pipeline()
    print("Application startup complete!")

# --- Data Loading ---
APARTMENT_DF = load_apartment_df(DATA_APARTMENTS)


# --- Core Logic ---
def process_request_logic(query: str) -> List[Apartment]:
    """Processes a user query to find and score apartments using batch processing."""
    print(f"Processing query: {query}")

    # Step 1: Process query to extract filters and qualitative questions
    filters, qualitative_questions = get_questions_and_filters_from_query(query)

    # Step 2: Apply extracted filters to get candidate apartments
    print(f"Extracted filters: {filters}")
    candidate_df = filter_apartments(
        df=APARTMENT_DF,
        rooms=filters.get('rooms'),
        rent_monthly=filters.get('rent_monthly'),
        area_m2=filters.get('area_m2'),
        beds=filters.get('beds'),
        deposit=filters.get('deposit'),
        currency=filters.get('currency'),
        availableFrom=filters.get('availableFrom'),
        city_name=filters.get('city_name')
    )
    print(f"Found {len(candidate_df)} candidates after initial filtering.")

    # Step 3: Score candidate apartments based on qualitative questions using batch processing
    scored_apartments = score_apartments_batch(candidate_df, qualitative_questions)

    # Sort apartments by score, descending and keep only the top 50 results
    print(f"\nFinal results:")
    print(f"  Total apartments before sorting: {len(scored_apartments)}")
    
    scored_apartments.sort(key=lambda x: x.overall_score, reverse=True)
    scored_apartments = scored_apartments[:50]
    
    print(f"  Returning top {len(scored_apartments)} apartments")
    if scored_apartments:
        print(f"  Score range: {scored_apartments[0].overall_score:.2f} to {scored_apartments[-1].overall_score:.2f}")
    
    return scored_apartments

@app.post("/generate", response_model=ProcessResponse)
def generate(req: ProcessRequest) -> ProcessResponse:
    """Main API endpoint for processing apartment search queries."""
    try:
        apartments = process_request_logic(req.query)
        return ProcessResponse(apartments=apartments)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in generate endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# --- Main Execution --- 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)