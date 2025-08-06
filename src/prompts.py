def create_questions_prompt(query: str):
    return f"""
    # General Instructions
    You are a search filter and question generator for a user query designed to help the user find apartments.

    It is your task to make a list of either search filters or general questions that can be answered by looking at the apartment images / description.
    For search filters you have to set "can_use_filter" to true and set "filter_name" to any of the following:
    ['city_name', 'area_m2', 'rooms', 'beds', 'rent_monthly', 'deposit']

    For general questions you have to set "can_use_filter" to false and set "filter_name" to null.
    The question will then be used later on to check the apartment images/description.
    Further, for general questions, indicate the scoring_type used to aggregate several scores from different images:
    "one_true": One correct score is enough (e.g. if the question is "Does the apartment have a tea kettle?" because as long as one apartment shows a tea kettle, it is a match.)
    "one_false": One incorrect score means the apartment is not a match (e.g. if the question is "Are there no steps in the apartment?" because if one image does show a step, it means the apartment is not a good fit for the user.)
    "default": The average of all scores should be used (e.g. if the question is "Is there a lovely view outside?" because the more windows show a lovely view, the better.)

    # Example
    ## Query:
    "apartment with 3 rooms and max 3000 euros monthly rent with a lovely view and a walk-in shower"
    ## Answer:
    ```json
    {{
        "questions": [
            {{
                "question": "Are there at least 3 rooms?",
                "can_use_filter": true,
                "filter_name": "rooms",
                "value": 3,
                "keyword": "rooms",
                "scoring_type": "default",
            }},
            {{
                "question": "Is the rent below 3000 euros monthly?",
                "can_use_filter": true,
                "filter_name": "rent_monthly",
                "value": 3000,
                "keyword": "rent_monthly",
                "scoring_type": "default",
            }},
            {{
                "question": "Is there a lovely view?",
                "can_use_filter": false,
                "filter_name": null,
                "value": null,
                "keyword": "Lovely view",
                "scoring_type": "default",
            }},
            {{
                "question": "Does the apartment have a walk-in shower?",
                "can_use_filter": false,
                "filter_name": null,
                "value": null,
                "keyword": "walk-in shower",
                "scoring_type": "one_true",
            }}
        ]
    }}
    ```

    # User query
    {query}

    Return list in json format.
    """

def description_match_prompt(description: str, question: str):
    return f"""
    # Task: Analyze apartment description against question
    
    You must respond with ONLY a JSON object in this exact format:
    {{"llmResponse": true}} or {{"llmResponse": false}} or {{"llmResponse": null}}
    
    # Rules:
    - Use true if the description clearly mentions that it matches the question
    - Use false if the description clearly mentions that it doesn't match the question  
    - Use null if the description contains no definite information about the question
    
    # Example:
    Description: "A beautiful apartment with a stunning view of the park."
    Question: "Does the apartment have a nice view?"
    Response: {{"llmResponse": true}}
    
    # Your task:
    Description: {description}
    Question: {question}
    
    Response (JSON only):
    """