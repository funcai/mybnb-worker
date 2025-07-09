# src/my_server.py

from fastapi import FastAPI
import requests, os, pydantic

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
app = FastAPI(title="My helper service")

class Req(pydantic.BaseModel):
    prompt: str = "Hello RunPod"

@app.post("/generate")
def generate(body: Req):
    payload = {"model": "gemma3n:e4b", "prompt": body.prompt, "stream": False}
    r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()  # FastAPI turns this into JSON for us