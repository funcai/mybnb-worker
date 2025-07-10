#!/usr/bin/env bash
set -euo pipefail

# Optional: tune GPU use; uncomment to constrain devices
# export CUDA_VISIBLE_DEVICES=0

# 1. start Ollama - it will see CUDA libs and use the GPU
ollama serve > /tmp/ollama.log 2>&1 &
OLLAMA_PID=$!
echo "Ollama (GPU) PID=$OLLAMA_PID"

# wait for Ollama REST API to be reachable (max 30s)
for i in {1..30}; do
  if curl -s http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    echo "Ollama is ready"
    break
  fi
  sleep 1
done


# 3. add src to PYTHONPATH so local imports work without prefix
# Safely set PYTHONPATH even if it was previously unset
export PYTHONPATH="/workspace/src${PYTHONPATH:+:${PYTHONPATH}}"

# 3. start FastAPI helper
uvicorn src.my_server:app --host 0.0.0.0 --port 8000 --workers 1 > /tmp/my_server.log 2>&1 &
echo "Python API PID=$!"

# 4. hand off to RunPod handler (PID 1)
exec python -u /workspace/src/handler.py