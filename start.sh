#!/usr/bin/env bash
set -euo pipefail

# Optional: tune GPU use; uncomment to constrain devices
# export CUDA_VISIBLE_DEVICES=0

# 3. add src to PYTHONPATH so local imports work without prefix
# Safely set PYTHONPATH even if it was previously unset
export PYTHONPATH="/workspace/src${PYTHONPATH:+:${PYTHONPATH}}"

# 3. start FastAPI helper
uvicorn src.my_server:app --host 0.0.0.0 --port 8000 --workers 1 > /tmp/my_server.log 2>&1 &
echo "Python API PID=$!"

# 4. hand off to RunPod handler (PID 1)
exec python -u /workspace/src/handler.py