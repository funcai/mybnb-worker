# src/handler.py

import requests, runpod, os, json

API_URL = "http://127.0.0.1:8000/generate"  # our FastAPI service

def handler(job):
    user_in = job["input"]  # {"prompt": "..."}
    # Short-circuit: if we send a warm-up event
    if user_in.get("command") == "boot":
        # Do nothing else – respond with a simple OK so the
        # platform knows the container is up
        return {"status": "ok"}

    # forward to local python micro-service
    # Ensure payload field matches FastAPI schema (expects 'query')
    if 'prompt' in user_in and 'query' not in user_in:
        user_in = {'query': user_in['prompt']}

    resp = requests.post(API_URL, json=user_in, timeout=120)
    resp.raise_for_status()
    return resp.json()  # whatever the service returns

runpod.serverless.start({"handler": handler})