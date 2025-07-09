Below is a GPU-ready rewrite of the worker we sketched earlier.
Everything else (FastAPI helper, handler, directory layout) stays the same—only the base image and the Ollama install differ.

1 Directory layout expected by RunPod 
docs.runpod.io
bash
Copy
Edit
ollama-runpod-worker/
├── builder/
│ └── requirements.txt # Python deps for _your_ code
├── src/
│ ├── handler.py # RunPod handler (entry point for Serverless)
│ ├── my_server.py # Your custom Python micro‑service
│ └── models.yaml # (optional) Ollama model manifest
├── start.sh # Starts all run‑time processes
└── Dockerfile # Builds the image
Tip for fast cold‑starts – add a models.yaml with the models you need (e.g. phi3:latest) and call ollama pull in start.sh; that way the model is baked into the image and the worker is ready in seconds.

2 Python pieces (place in src/)
my_server.py – a FastAPI thin wrapper around Ollama
python
Copy
Edit

# src/my_server.py

from fastapi import FastAPI
import requests, os, pydantic

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
app = FastAPI(title="My helper service")

class Req(pydantic.BaseModel):
prompt: str = "Hello RunPod"

@app.post("/generate")
def generate(body: Req):
payload = {"model": "phi3", "prompt": body.prompt, "stream": False}
r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=120)
r.raise_for_status()
return r.json() # FastAPI turns this into JSON for us
handler.py – the RunPod entry‑point 
docs.runpod.io
python
Copy
Edit

# src/handler.py

import requests, runpod, os, json

API_URL = "http://127.0.0.1:8000/generate" # our FastAPI service

def handler(job):
user_in = job["input"] # {"prompt": "..."} # forward to local python micro‑service
resp = requests.post(API_URL, json=user_in, timeout=120)
resp.raise_for_status()
return resp.json() # whatever the service returns

runpod.serverless.start({ "handler": handler, "concurrency*modifier": lambda *: 1 })
3 builder/requirements.txt
makefile
Copy
Edit
runpod>=1.7
fastapi==0.110
uvicorn[standard]==0.29
requests
pydantic
(Keep it lean—smaller images start faster! )

4 start.sh – one entry‑point to rule them all 
docs.runpod.io
bash
Copy
Edit
#!/usr/bin/env bash
set -euo pipefail

# 1. start Ollama (CPU build – replace with GPU build if your endpoint has GPUs)

ollama serve > /tmp/ollama.log 2>&1 &
echo "Ollama started PID=$!"

# (optional) pre‑pull model(s) so the first request is instant

[ -f /workspace/src/models.yaml ] && ollama pull --yes $(yq '.[] | .model' /workspace/src/models.yaml) &

# 2. start your micro‑service

uvicorn src.my_server:app --host 0.0.0.0 --port 8000 --workers 1 > /tmp/my_server.log 2>&1 &
echo "Python service started PID=$!"

# 3. hand control to RunPod Serverless – replaces this shell

exec python -u /workspace/src/handler.py
The exec … is important: it makes the handler process PID 1, matching what the RunPod runtime expects. Any background services keep running because they were launched earlier in the script.

Make it executable:

bash
Copy
Edit
chmod +x start.sh

1 Dockerfile — NVIDIA/CUDA build
dockerfile
Copy
Edit

# CUDA-enabled RunPod base (Ubuntu 22.04 + CUDA 12.1 + PyTorch 2.3)

FROM runpod/pytorch:2.3.0-py3.10-cuda12.1.0-devel-ubuntu22.04 # GPU libs pre-installed :contentReference[oaicite:0]{index=0}

ARG WORKSPACE_DIR=/workspace
WORKDIR ${WORKSPACE_DIR}

# ---------- Ollama (CUDA build) ----------

# The install script detects the GPU toolchain automatically when it

# runs inside a CUDA image; no flags needed.

RUN curl -fsSL https://ollama.ai/install.sh | sh

# ---------- Python deps (same as CPU version) ----------

COPY builder/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- copy code ----------

COPY src ./src
COPY start.sh .
RUN chmod +x start.sh

# ---------- ports ----------

EXPOSE 11434 8000

CMD ["./start.sh"]
Why this works

The runpod/pytorch:_-cuda_ family already bundles the NVIDIA drivers, CUDA 12.x runtime and cuDNN—so your container only needs the Ollama binary built with GPU support.
docs.runpod.io

When the image starts on RunPod, the nvidia-container runtime is injected automatically; no --gpus flag or toolkit install is required.

2 start.sh (unchanged but GPU-aware)
bash
Copy
Edit
#!/usr/bin/env bash
set -euo pipefail

# Optional: tune GPU use; uncomment to constrain devices

# export CUDA_VISIBLE_DEVICES=0

# 1. start Ollama – it will see CUDA libs and use the GPU

ollama serve > /tmp/ollama.log 2>&1 &
echo "Ollama (GPU) PID=$!"

# 2. pull model weights now so the worker is hot right away

ollama pull --yes phi3 &

# 3. start FastAPI helper

uvicorn src.my_server:app --host 0.0.0.0 --port 8000 --workers 1 > /tmp/my_server.log 2>&1 &
echo "Python API PID=$!"

# 4. hand off to RunPod handler (PID 1)

exec python -u /workspace/src/handler.py
