# Official PyTorch CUDA base (Ubuntu 22.04 + CUDA 12.1 + PyTorch 2.3)
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

ARG WORKSPACE_DIR=/workspace
WORKDIR ${WORKSPACE_DIR}

# ---------- Ollama (CUDA build) ----------
# The install script detects the GPU toolchain automatically when it
# runs inside a CUDA image; no flags needed.
RUN apt-get update && apt-get install -y curl fuse-overlayfs \
    && curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o /tmp/ollama.tgz \
    && tar -C /usr/local -xzf /tmp/ollama.tgz \
    && rm /tmp/ollama.tgz

# ---------- Pre-pull Ollama model ----------
# Start the Ollama server temporarily, pull the model, then stop the server so the
# model files are cached inside the image. Using a background server avoids the
# 'ollama server not responding' error during build.
RUN ollama serve > /tmp/ollama.log 2>&1 & \
    bash -c 'for i in {1..30}; do curl -s http://localhost:11434/api/tags && break || sleep 1; done' && \
    ollama pull gemma3n:e4b && \
    pkill ollama

# ---------- Python deps (same as CPU version) ----------
COPY builder/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- copy code ----------
COPY src ./src
COPY start.sh .
# ---------- local test files ----------
COPY test_input.json .
COPY test_handler.sh .
RUN chmod +x test_handler.sh
RUN chmod +x start.sh

# ---------- ports ----------
EXPOSE 11434 8000

CMD ["./start.sh"]