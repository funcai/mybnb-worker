# src/handler.py

import requests, runpod, os, json

API_URL = "http://127.0.0.1:8000/generate"  # our FastAPI service

def handler(job):
    user_in = job["input"]  # {"prompt": "..."}
    # forward to local python micro-service
    resp = requests.post(API_URL, json=user_in, timeout=120)
    resp.raise_for_status()
    return resp.json()  # whatever the service returns

runpod.serverless.start({"handler": handler})