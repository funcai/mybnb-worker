# src/apartment_scorer.py

from typing import List, Dict, Tuple
import pandas as pd
from pydantic import BaseModel, Field

from llm_service import batch_generate, extract_json_from_response
from prompts import description_match_prompt


# Pydantic models for scoring
class ScoreDetail(BaseModel):
    question: str
    score: float
    explanation: str
    keyword: str


class Apartment(BaseModel):
    id: str
    url: str
    provider: str
    address: dict
    facts: dict
    overall_score: float
    score_details: List[ScoreDetail]


class ScoringResponse(BaseModel):
    response: str = Field(description="'yes', 'no', or 'irrelevant'")


def prepare_scoring_prompts(candidate_df: pd.DataFrame, qualitative_questions: List[Dict]) -> Tuple[List[str], List[Dict]]:
    """
    Prepare batch prompts for apartment scoring.
    
    Args:
        candidate_df: DataFrame of candidate apartments
        qualitative_questions: List of qualitative questions to score against
        
    Returns:
        Tuple of (batch_prompts, prompt_metadata)
    """
    batch_prompts = []
    prompt_metadata = []  # To track which prompt corresponds to which apartment/question
    
    for apt_idx, (_, row) in enumerate(candidate_df.iterrows()):
        description = row["description"]
        for q_idx, question_obj in enumerate(qualitative_questions):
            question_text = question_obj.get("question", "")
            if question_text:
                match_prompt = description_match_prompt(description=description, question=question_text)
                batch_prompts.append(match_prompt)
                prompt_metadata.append({
                    'apt_idx': apt_idx,
                    'question_idx': q_idx,
                    'question_text': question_text,
                    'keyword': question_obj.get("keyword", ""),
                    'row': row
                })
    
    print(f"Prepared {len(batch_prompts)} scoring prompts for {len(candidate_df)} apartments")
    return batch_prompts, prompt_metadata


def process_scoring_responses(batch_responses: List[str], prompt_metadata: List[Dict], qualitative_questions: List[Dict]) -> Dict[int, Dict]:
    """
    Process batch scoring responses and organize by apartment.
    
    Args:
        batch_responses: List of LLM responses from batch generation
        prompt_metadata: Metadata for each prompt (apartment/question mapping)
        qualitative_questions: List of qualitative questions
        
    Returns:
        Dictionary mapping apartment index to score data
    """
    apartment_scores = {}
    
    for i, (response, metadata) in enumerate(zip(batch_responses, prompt_metadata)):
        apt_idx = metadata['apt_idx']
        question_idx = metadata['question_idx']
        question_text = metadata['question_text']
        keyword = metadata['keyword']
        
        # Parse scoring response (extract_json_from_response handles all error cases)
        scoring_dict = extract_json_from_response(response)
        scoring_response = ScoringResponse.model_validate(scoring_dict)
        response_text = scoring_response.response.lower()
        llm_explanation = response_text
        
        question_score = 1.0 if "yes" in response_text else 0.0
        
        # Initialize apartment scores if not exists
        if apt_idx not in apartment_scores:
            apartment_scores[apt_idx] = {
                'total_score': 0,
                'score_details': [],
                'row': metadata['row']
            }
        
        # Add score detail
        apartment_scores[apt_idx]['score_details'].append(
            ScoreDetail(
                question=question_text,
                score=question_score,
                explanation=llm_explanation,
                keyword=keyword
            )
        )
        apartment_scores[apt_idx]['total_score'] += question_score
    
    return apartment_scores


def create_apartment_objects(apartment_scores: Dict[int, Dict], qualitative_questions: List[Dict]) -> List[Apartment]:
    """
    Create Apartment objects from scoring results.
    
    Args:
        apartment_scores: Dictionary mapping apartment index to score data
        qualitative_questions: List of qualitative questions (for calculating overall score)
        
    Returns:
        List of Apartment objects with scores
    """
    scored_apartments = []
    
    for apt_idx, score_data in apartment_scores.items():
        row = score_data['row']
        overall_score = score_data['total_score'] / len(qualitative_questions) if qualitative_questions else 1.0
        
        print(f"  Apartment {apt_idx}: overall_score={overall_score:.2f} (total={score_data['total_score']:.1f}/{len(qualitative_questions)})")
        
        scored_apartments.append(
            Apartment(
                id=row["_id"]["$oid"],
                url=row.get("url", ""),
                provider=row.get("provider", ""),
                address=row.get("address", {}),
                facts=row.get("facts", {}),
                overall_score=overall_score,
                score_details=score_data['score_details'],
            )
        )
    
    return scored_apartments


def create_default_apartments(candidate_df: pd.DataFrame) -> List[Apartment]:
    """
    Create apartment objects with default score (1.0) when no qualitative questions exist.
    
    Args:
        candidate_df: DataFrame of candidate apartments
        
    Returns:
        List of Apartment objects with default scores
    """
    scored_apartments = []
    
    for _, row in candidate_df.iterrows():
        scored_apartments.append(
            Apartment(
                id=row["_id"]["$oid"],
                url=row.get("url", ""),
                provider=row.get("provider", ""),
                address=row.get("address", {}),
                facts=row.get("facts", {}),
                overall_score=1.0,
                score_details=[],
            )
        )
    
    print(f"Created {len(scored_apartments)} apartments with default scores")
    return scored_apartments


def score_apartments_batch(candidate_df: pd.DataFrame, qualitative_questions: List[Dict]) -> List[Apartment]:
    """
    Score apartments using batch processing for qualitative questions.
    
    Args:
        candidate_df: DataFrame of candidate apartments to score
        qualitative_questions: List of qualitative questions to score against
        
    Returns:
        List of scored Apartment objects
    """
    if not qualitative_questions:
        # No qualitative questions, all apartments get score 1.0
        return create_default_apartments(candidate_df)
    
    # Prepare batch scoring prompts
    batch_prompts, prompt_metadata = prepare_scoring_prompts(candidate_df, qualitative_questions)
    
    if not batch_prompts:
        return create_default_apartments(candidate_df)
    
    print(f"Processing {len(batch_prompts)} scoring prompts in batch...")
    
    # Process all scoring prompts in batch
    batch_responses = batch_generate(batch_prompts, max_new_tokens=100, debug=False)
    
    # Process batch responses and organize by apartment
    apartment_scores = process_scoring_responses(batch_responses, prompt_metadata, qualitative_questions)
    
    # Create final apartment objects
    scored_apartments = create_apartment_objects(apartment_scores, qualitative_questions)
    
    return scored_apartments
