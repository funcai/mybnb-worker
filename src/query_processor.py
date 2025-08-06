# src/query_processor.py

from typing import List, Dict, Optional, Union
from fastapi import HTTPException
from pydantic import BaseModel, Field

from llm_service import batch_generate, extract_json_from_response
from prompts import create_questions_prompt


# Pydantic models for structured output
class Question(BaseModel):
    question: str
    can_use_filter: bool
    filter_name: Optional[str] = None
    value: Optional[Union[str, int, float, bool]] = None
    keyword: str
    scoring_type: Optional[str] = "default"  # "default", "one_true", or "one_false"


class QueryAnalysis(BaseModel):
    questions: List[Question]


def let_llm_generate_query_parts(query: str) -> QueryAnalysis:
    """
    Analyze a user query using LLM to extract filters and questions.
    
    Args:
        query: The user's search query
        
    Returns:
        QueryAnalysis object containing extracted questions and filters
        
    Raises:
        HTTPException: If query analysis fails
    """
    print(f"Analyzing query: {query}")
    
    # Create prompt for query analysis
    prompt = create_questions_prompt(query=query)
    
    try:
        # Use LLM to analyze the query
        responses = batch_generate([prompt], max_new_tokens=500)
        llm_response_str = responses[0]
        print(f"LLM response for query analysis: {llm_response_str}")
        
        # Extract and parse JSON from response
        query_analysis_dict = extract_json_from_response(llm_response_str)
        query_analysis = QueryAnalysis.model_validate(query_analysis_dict)
        
        print(f"Successfully analyzed query into {len(query_analysis.questions)} questions")
        return query_analysis
        
    except Exception as e:
        print(f"FATAL: Could not get or validate structured response for query analysis: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze query with LLM")


def extract_filters_from_questions(questions: List[Question]) -> Dict[str, Union[str, int, float, bool]]:
    """
    Extract filter parameters from analyzed questions.
    
    Args:
        questions: List of Question objects from query analysis
        
    Returns:
        Dictionary of filter_name -> value mappings
    """
    # Convert to dict format for easier processing
    questions_dict = [q.model_dump() for q in questions]
    
    # Extract filters where can_use_filter is True and filter_name is not None
    filters = {
        q['filter_name']: q['value'] 
        for q in questions_dict 
        if q.get('can_use_filter') and q.get('filter_name') is not None and q.get('value') is not None
    }
    
    print(f"Extracted filters: {filters}")
    return filters


def extract_qualitative_questions(questions: List[Question]) -> List[Dict]:
    """
    Extract qualitative questions that need LLM scoring.
    
    Args:
        questions: List of Question objects from query analysis
        
    Returns:
        List of question dictionaries that are qualitative (not filterable)
    """
    # Convert to dict format for easier processing
    questions_dict = [q.model_dump() for q in questions]
    
    # Questions are qualitative if can_use_filter is false OR filter_name is null
    qualitative_questions = [
        q for q in questions_dict 
        if not q.get('can_use_filter') or q.get('filter_name') is None
    ]
    
    print(f"Found {len(qualitative_questions)} qualitative questions")
    return qualitative_questions


def get_questions_and_filters_from_query(query: str) -> tuple[Dict[str, Union[str, int, float, bool]], List[Dict]]:
    """
    Complete query processing pipeline.
    
    Args:
        query: The user's search query
        
    Returns:
        Tuple of (filters_dict, qualitative_questions_list)
    """
    # Step 1: Analyze query with LLM
    query_analysis = let_llm_generate_query_parts(query)
    
    # Step 2: Extract filters and qualitative questions
    filters = extract_filters_from_questions(query_analysis.questions)
    qualitative_questions = extract_qualitative_questions(query_analysis.questions)
    
    return filters, qualitative_questions
