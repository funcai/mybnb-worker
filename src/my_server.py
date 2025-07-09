# src/my_server.py

# from fastapi import FastAPI
# import requests, os, pydantic

# OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
# app = FastAPI(title="My helper service")
import pandas as pd
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import ollama
from typing import List, Optional, Union

# Local imports from our project files
from agent_filter import load_apartment_df, filter_apartments
from prompts import create_questions_prompt, description_match_prompt

# --- Pydantic Models for API validation and structured LLM output ---

class ProcessRequest(BaseModel):
    query: str

class ScoreDetail(BaseModel):
    question: str
    score: float
    explanation: str
    keyword: str

class Apartment(BaseModel):
    id: str
    overall_score: float
    score_details: list[ScoreDetail]

class ProcessResponse(BaseModel):
    apartments: list[Apartment]

# Pydantic models for structured output from Ollama
class Question(BaseModel):
    question: str
    can_use_filter: bool
    filter_name: Optional[str] = None
    value: Optional[Union[str, int, float, bool]] = None
    keyword: str

class QueryAnalysis(BaseModel):
    questions: List[Question]

class ScoringResponse(BaseModel):
    response: str = Field(description="'yes', 'no', or 'irrelevant'")

scoring_schema = {
    "type": "object",
    "additionalProperties": False,
    "required": ["response"],
    "properties": {
        "response": {
            "type": "string",
            "enum": ["yes", "no", "irrelevant"],
            "description": "Classifier-style verdict"
        }
    }
}

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Apartment Search Server",
    description="An API to find and score apartments based on natural language queries.",
    version="1.0.0",
)

# --- Data Loading ---
APARTMENT_DF = load_apartment_df("data/apartments.json")

# --- Core Logic ---
def process_request_logic(query: str) -> list[Apartment]:
    """Processes a user query to find and score apartments."""
    print(f"Processing query: {query}")

    # Step 1: Use LLM to analyze the query and extract filters/questions
    prompt = create_questions_prompt(query=query)
    try:
        response = ollama.chat(
            model='gemma3n:e4b',
            messages=[{'role': 'user', 'content': prompt}],
            format='json',
            options={'json_schema': QueryAnalysis.model_json_schema()},
        )
        llm_response_str = response['message']['content']
        print(f"LLM response for query analysis: {llm_response_str}")
        query_analysis = QueryAnalysis.model_validate_json(llm_response_str)
        questions = [q.model_dump() for q in query_analysis.questions]
    except Exception as e:
        print(f"FATAL: Could not get or validate structured response for query analysis: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze query with LLM")

    # Step 2: Apply extracted filters to get candidate apartments
    filters = {q['filter_name']: q['value'] for q in questions if q.get('can_use_filter') and q.get('value') is not None}
    print(f"Extracted filters: {filters}")
    candidate_df = filter_apartments(
        df=APARTMENT_DF,
        rooms=filters.get('rooms'),
        rent_monthly=filters.get('rent_monthly'),
        area_m2=filters.get('area_m2'),
        beds=filters.get('beds'),
        deposit=filters.get('deposit'),
        currency=filters.get('currency'),
        furnished=filters.get('furnished'),
        availableFrom=filters.get('availableFrom'),
        city_name=filters.get('city_name')
    )
    print(f"Found {len(candidate_df)} candidates after initial filtering.")

    # Step 3: Score candidate apartments based on qualitative questions
    qualitative_questions = [q for q in questions if not q.get('can_use_filter')]
    scored_apartments = []

    for _, row in candidate_df.iterrows():
        description = row["description"]
        score_details = []
        total_score = 0

        if not qualitative_questions:
            overall_score = 1.0
        else:
            for question_obj in qualitative_questions:
                question_text = question_obj.get("question", "")
                keyword = question_obj.get("keyword", "")
                if not question_text:
                    continue

                question_score = 0
                llm_explanation = "Error during scoring."
                match_prompt = description_match_prompt(description=description, question=question_text)
                try:
                    response = ollama.chat(
                        model='gemma3n:e4b',
                        messages=[{'role': 'user', 'content': match_prompt}],
                        format=scoring_schema,
                        options={"temperature": 0}
                        # options={'json_schema': ScoringResponse.model_json_schema()},
                    )
                    llm_response_str = response['message']['content']
                    print(f"Scoring apt {row['listingId']} for question '{question_text}': {llm_response_str}")
                    scoring_response = ScoringResponse.model_validate_json(llm_response_str)
                    response_text = scoring_response.response.lower()
                    llm_explanation = response_text
                except Exception as e:
                    print(f"  - WARN: Could not get structured response from LLM: {e}. Defaulting to 'maybe'.")
                    response_text = "maybe"
                    llm_explanation = str(e)

                if "yes" in response_text:
                    question_score = 1.0
                
                score_details.append(ScoreDetail(question=question_text, score=question_score, explanation=llm_explanation, keyword=keyword))
                total_score += question_score
            
            overall_score = total_score / len(qualitative_questions) if qualitative_questions else 1.0
        
        scored_apartments.append(Apartment(id=row["listingId"], overall_score=overall_score, score_details=score_details))

    # Sort apartments by score, descending
    scored_apartments.sort(key=lambda x: x.overall_score, reverse=True)
    return scored_apartments

# class Req(pydantic.BaseModel):
#     prompt: str = "Hello RunPod"

@app.post("/generate", response_model=ProcessResponse)
def generate(req: ProcessRequest):
    # payload = {"model": "gemma3n:e4b", "prompt": body.prompt, "stream": False}
    # r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=120)
    # r.raise_for_status()
    # return r.json()  # FastAPI turns this into JSON for us
    
    try:
        apartments = process_request_logic(req.query)
        return ProcessResponse(apartments=apartments)
    except HTTPException as e:
        # Re-raise HTTPException to let FastAPI handle it
        raise e
    except Exception as e:
        # Catch any other unexpected errors
        print(f"FATAL: An unexpected error occurred in process_request: {e}")
        # Optionally log the full traceback here
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {e}")
