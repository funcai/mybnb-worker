# src/apartment_scorer.py

from typing import List, Dict, Tuple
import pandas as pd
from pydantic import BaseModel, Field

from llm_service import batch_generate, batch_generate_vision, extract_json_from_response
from prompts import description_match_prompt
from image_cache_multi import get_cached_image_paths


# Pydantic models for scoring
class ScoreDetail(BaseModel):
    question: str
    score: float
    description_matches: bool
    vision_matches: bool
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
    llmResponse: bool | None = Field(description="true, false, or null")


class VisionScoringResponse(BaseModel):
    llmResponse: bool | None = Field(description="true, false, or null")


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
        response_text = scoring_response.llmResponse
        
        question_score = 1.0 if response_text else 0.0
        
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
                description_matches=bool(response_text) if response_text is not None else False,
                vision_matches=False,  # Will be updated later in vision scoring
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
    Score apartments using batch processing for qualitative questions, including vision scoring.
    
    Args:
        candidate_df: DataFrame of candidate apartments to score
        qualitative_questions: List of qualitative questions to score against
        
    Returns:
        List of scored Apartment objects
    """
    if qualitative_questions is None or len(qualitative_questions) == 0:
        return create_default_apartments(candidate_df)
    
    print(f"Scoring {len(candidate_df)} apartments against {len(qualitative_questions)} qualitative questions...")
    
    # Prepare batch prompts for description scoring
    batch_prompts, prompt_metadata = prepare_scoring_prompts(candidate_df, qualitative_questions)
    
    if not batch_prompts:
        return create_default_apartments(candidate_df)
    
    print(f"Processing {len(batch_prompts)} description scoring prompts in batch...")
    
    # Process all description scoring prompts in batch
    batch_responses = batch_generate(batch_prompts, max_new_tokens=100, debug=False)
    
    # Process batch responses and organize by apartment
    apartment_scores = process_scoring_responses(batch_responses, prompt_metadata, qualitative_questions)
    
    # Prepare and process vision scoring
    print("Preparing vision scoring prompts...")
    vision_image_text_pairs, vision_prompt_metadata = prepare_vision_scoring_prompts(candidate_df, qualitative_questions)
    
    if vision_image_text_pairs:
        print(f"Processing {len(vision_image_text_pairs)} vision scoring prompts in batch...")
        
        # Process all vision scoring prompts in batch
        vision_batch_responses = batch_generate_vision(vision_image_text_pairs, max_new_tokens=50, debug=False)
        
        # Process vision responses
        apartment_vision_scores = process_vision_scoring_responses(vision_batch_responses, vision_prompt_metadata)
        
        # Calculate final vision scores
        final_vision_scores = calculate_vision_scores(apartment_vision_scores)
        
        # Integrate vision scores with description scores
        apartment_scores = integrate_vision_scores(apartment_scores, final_vision_scores, qualitative_questions)
    else:
        print("No images found for vision scoring.")
    
    # Create final apartment objects
    scored_apartments = create_apartment_objects(apartment_scores, qualitative_questions)
    
    return scored_apartments


def prepare_vision_scoring_prompts(candidate_df: pd.DataFrame, qualitative_questions: List[Dict]) -> Tuple[List[Tuple], List[Dict]]:
    """
    Prepare batch vision prompts for apartment image scoring.
    
    Args:
        candidate_df: DataFrame of candidate apartments
        qualitative_questions: List of qualitative questions to score against
        
    Returns:
        Tuple of (image_text_pairs, prompt_metadata)
    """
    # First, collect all unique image URLs
    all_image_urls = set()
    image_metadata = []  # Track which images belong to which apartments/questions
    
    for apt_idx, (_, row) in enumerate(candidate_df.iterrows()):
        # Get apartment images
        images = row.get("images", [])
        if not images:
            continue
            
        for q_idx, question_obj in enumerate(qualitative_questions):
            question_text = question_obj.get("question", "")
            keyword = question_obj.get("keyword", "")
            scoring_type = question_obj.get("scoring_type", "default")
            
            # Track each image for this question
            for img_idx, image_url in enumerate(images):
                all_image_urls.add(image_url)
                image_metadata.append({
                    'apt_idx': apt_idx,
                    'question_idx': q_idx,
                    'question_text': question_text,
                    'keyword': keyword,
                    'scoring_type': scoring_type,
                    'image_idx': img_idx,
                    'image_url': image_url,
                    'row': row
                })
    
    # Download all images in parallel using multi image cache
    print(f"Downloading {len(all_image_urls)} unique images in parallel...")
    url_to_path = get_cached_image_paths(list(all_image_urls), max_parallel=32)
    
    # Now create vision prompts for successfully cached images
    image_text_pairs = []
    prompt_metadata = []
    
    for metadata in image_metadata:
        image_url = metadata['image_url']
        local_image_path = url_to_path.get(image_url)
        
        if not local_image_path:
            continue  # Skip images that failed to download
            
        question_text = metadata['question_text']
        
        # Create vision prompt
        system_prompt = f"""You are analyzing apartment images to answer the user's question: "{question_text}"

Look at this apartment image and determine if it shows or contains what the user is asking about.

Respond with ONLY a JSON object in this exact format:
{{
    "llmResponse": true | false | null
}}

Guidelines:
- true: The image clearly shows what the user is asking about
- false: The image clearly does NOT show what the user is asking about  
- null: The image doesn't provide enough information to determine either way"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Does this apartment image show: {question_text}?"}
        ]
        
        image_text_pairs.append((local_image_path, messages))
        prompt_metadata.append(metadata)
    
    return image_text_pairs, prompt_metadata


def process_vision_scoring_responses(batch_responses: List[str], prompt_metadata: List[Dict]) -> Dict[int, Dict]:
    """
    Process batch vision scoring responses and organize by apartment.
    
    Args:
        batch_responses: List of LLM responses from batch vision generation
        prompt_metadata: Metadata for each prompt (apartment/question/image mapping)
        
    Returns:
        Dictionary mapping apartment index to vision score data
    """
    apartment_vision_scores = {}
    
    for i, (response, metadata) in enumerate(zip(batch_responses, prompt_metadata)):
        apt_idx = metadata['apt_idx']
        question_idx = metadata['question_idx']
        question_text = metadata['question_text']
        keyword = metadata['keyword']
        scoring_type = metadata.get('scoring_type', 'default')
        
        # Parse vision scoring response
        scoring_dict = extract_json_from_response(response)
        try:
            vision_response = VisionScoringResponse.model_validate(scoring_dict)
            raw_response = vision_response.llmResponse
            
            # Map various response formats to our expected values
            if raw_response is True:
                response_text = "true"
            elif raw_response is False:
                response_text = "false"
            else:
                response_text = "null"
                
        except Exception as e:
            # If JSON parsing fails, try to extract meaning from raw response
            response_lower = response.lower()
            if "true" in response_lower or "yes" in response_lower or "shows" in response_lower:
                response_text = "true"
            elif "false" in response_lower or "no" in response_lower or "not" in response_lower:
                response_text = "false"
            else:
                response_text = "null"
            print(f"JSON parsing failed for vision response: {e}. Extracted: '{response_text}' from: {response[:100]}...")
        
        # Initialize apartment vision scores if not exists
        if apt_idx not in apartment_vision_scores:
            apartment_vision_scores[apt_idx] = {}
        
        # Initialize question scores if not exists
        if question_idx not in apartment_vision_scores[apt_idx]:
            apartment_vision_scores[apt_idx][question_idx] = {
                'question_text': question_text,
                'keyword': keyword,
                'scoring_type': scoring_type,
                'matches': False,
                'doesnt_match': False,
                'responses': []
            }
        
        # Track responses
        apartment_vision_scores[apt_idx][question_idx]['responses'].append(response_text)
        
        # Update match flags
        if response_text == "true":
            apartment_vision_scores[apt_idx][question_idx]['matches'] = True
        elif response_text == "false":
            apartment_vision_scores[apt_idx][question_idx]['doesnt_match'] = True
    
    return apartment_vision_scores


def calculate_vision_scores(apartment_vision_scores: Dict[int, Dict]) -> Dict[int, Dict]:
    """
    Calculate final vision scores based on the scoring_type aggregation strategy.
    
    Scoring types:
    - "default": Average of all scores (if a response is true do +1, if one is false, -1, if null, ignore. Then divide by the total amount of non-null responses)
    - "one_true": One correct score is enough (+1 if any match, 0 otherwise)
    - "one_false": One incorrect score means failure (-1 if any doesn't match. If none don't match set to +1 if any match, 0 otherwise)
    
    Args:
        apartment_vision_scores: Raw vision scoring data with the following structure:
            {
                apartment_idx: {
                    question_idx: {
                        'question_text': str,     # The question being asked
                        'keyword': str,           # Keyword associated with the question
                        'scoring_type': str,      # "default", "one_true", or "one_false"
                        'matches': bool,          # True if any image matched the question
                        'doesnt_match': bool,     # True if any image explicitly didn't match
                        'responses': List[str]    # List of individual image responses: ["true", "false", "null", ...]
                    }
                }
            }
        
    Returns:
        Dictionary mapping apartment index to final vision scores per question:
            {
                apartment_idx: {
                    question_idx: {
                        'question': str,
                        'score': float,
                        'explanation': str,
                        'matches': bool,
                        'keyword': str,
                        'scoring_type': str
                    }
                }
            }
    """
    final_vision_scores = {}
    
    for apt_idx, questions in apartment_vision_scores.items():
        final_vision_scores[apt_idx] = {}
        
        for question_idx, question_data in questions.items():
            question_text = question_data['question_text']
            keyword = question_data['keyword']
            scoring_type = question_data.get('scoring_type', 'default')
            matches = question_data['matches']
            doesnt_match = question_data['doesnt_match']
            responses = question_data.get('responses', [])
            
            # Apply scoring rules based on scoring_type
            if scoring_type == "one_true":
                # One correct score is enough
                if matches:
                    score = 1.0
                    explanation = "At least one image shows matching content (one_true strategy)"
                else:
                    score = 0.0
                    explanation = "No images show matching content (one_true strategy)"
                    
            elif scoring_type == "one_false":
                # One incorrect score means failure
                if doesnt_match:
                    score = -1.0
                    explanation = "At least one image shows non-matching content (one_false strategy)"
                elif matches:
                    score = 1.0
                    explanation = "Images show matching content with no contradictions (one_false strategy)"
                else:
                    score = 0.0
                    explanation = "Images are irrelevant or inconclusive (one_false strategy)"
                    
            else:  # "default" or any other value
                # Standard aggregation: calculate average based on individual responses
                if not responses:
                    score = 0.0
                    explanation = "No image responses available (default strategy)"
                else:
                    # Calculate average score from individual responses
                    true_count = responses.count("true")
                    false_count = responses.count("false")
                    null_count = responses.count("null")
                    total_responses = len(responses)
                    non_null_responses = true_count + false_count  # Only count non-null responses
                    
                    if non_null_responses == 0:
                        # All responses are null - treat as neutral/inconclusive
                        score = 0.0
                        explanation = f"All {total_responses} responses were inconclusive (default strategy)"
                    else:
                        # Calculate weighted average: true=+1, false=-1, null=ignored
                        # Divide by non-null responses only as per docstring
                        score = (true_count - false_count) / non_null_responses
                        explanation = f"Average score from {non_null_responses} conclusive images: {true_count} positive, {false_count} negative, {null_count} inconclusive (default strategy)"
            print(f"Explanation {explanation} and score {score}")
            final_vision_scores[apt_idx][question_idx] = {
                'question': question_text,
                'score': score,
                'explanation': explanation,
                'matches': matches,
                'keyword': keyword,
                'scoring_type': scoring_type
            }
    
    return final_vision_scores


def integrate_vision_scores(apartment_scores: Dict[int, Dict], final_vision_scores: Dict[int, Dict], qualitative_questions: List[Dict]) -> Dict[int, Dict]:
    """
    Integrate vision scores with description scores.
    
    Args:
        apartment_scores: Description-based scores
        final_vision_scores: Vision-based scores
        qualitative_questions: List of qualitative questions
        
    Returns:
        Updated apartment scores with integrated vision scores
    """
    for apt_idx, vision_scores in final_vision_scores.items():
        if apt_idx not in apartment_scores:
            # Initialize if apartment wasn't scored by description (shouldn't happen)
            apartment_scores[apt_idx] = {
                'total_score': 0,
                'score_details': [],
                'row': None  # Will need to be filled
            }
        
        # Update scores for each question
        for question_idx, vision_score_data in vision_scores.items():
            # Find the corresponding description score detail
            for score_detail in apartment_scores[apt_idx]['score_details']:
                if score_detail.question == vision_score_data['question']:
                    # Apply vision score adjustment
                    vision_adjustment = vision_score_data['score']
                    original_score = score_detail.score
                    
                    # Update the score with vision adjustment
                    new_score = original_score + vision_adjustment
                    
                    # Update the score detail
                    score_detail.score = new_score
                    
                    # Update explanation to include vision info
                    score_detail.description_matches = original_score > 0
                    score_detail.vision_matches = vision_score_data['matches']
                    
                    # Update total score
                    apartment_scores[apt_idx]['total_score'] += vision_adjustment
                    
                    break
    
    return apartment_scores
