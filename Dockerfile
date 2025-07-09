# Official PyTorch CUDA base (Ubuntu 22.04 + CUDA 12.1 + PyTorch 2.3)
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

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