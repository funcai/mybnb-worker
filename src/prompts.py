def create_questions_prompt(query: str):
    return f"""
    # General Instructions
    The user wants to find apartments with the following requirements:

    First make a list of what the user wants to know based on their request.
    There are existing filters that can be applied to search:
    ['city_name', 'area_m2', 'rooms', 'beds', 'rent_monthly', 'deposit', 'currency', 'availableFrom']
    For each question, indicate if one of the filters can be used (true) or not (false).
    If a filterable attribute is not mentioned in the query, its value should be `null`.

    # Example
    ## Query:
    "apartment with 3 rooms and 3000 euros monthly rent with a lovely view"
    ## Answer:
    ```json
    {{
        "questions": [
            {{
                "question": "What is the number of rooms?",
                "can_use_filter": true,
                "filter_name": "rooms",
                "value": 3,
                "keyword": "rooms"
            }},
            {{
                "question": "What is the monthly rent?",
                "can_use_filter": true,
                "filter_name": "rent_monthly",
                "value": 3000,
                "keyword": "rent_monthly"
            }},
            {{
                "question": "Is there a lovely view?",
                "can_use_filter": false,
                "filter_name": null,
                "value": null,
                "keyword": "view"
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
    {{"response": "yes"}} or {{"response": "no"}} or {{"response": "irrelevant"}}
    
    # Rules:
    - Use "yes" if the description clearly mentions that it matches the question
    - Use "no" if the description clearly mentions that it doesn't match the question  
    - Use "irrelevant" if the description contains no definite information about the question
    
    # Example:
    Description: "A beautiful apartment with a stunning view of the park."
    Question: "Does the apartment have a nice view?"
    Response: {{"response": "yes"}}
    
    # Your task:
    Description: {description}
    Question: {question}
    
    Response (JSON only):
    """