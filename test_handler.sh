#!/usr/bin/env bash
# Local test script for Runpod serverless handler inside the container
# Runs the handler with the test_input.json file.

# start real Ollama in background
ollama serve &
OLLAMA_PID=$!

# wait a bit for server to start
sleep 5

# (model pull intentionally skipped per user request)
# OLLAMA_PID already set above
echo "Ollama PID=$OLLAMA_PID"

# start FastAPI helper in background
uvicorn src.my_server:app --host 0.0.0.0 --port 8000 --workers 1 &
API_PID=$!
echo "API PID=$API_PID"

# ensure background servers are terminated on exit
trap "kill $OLLAMA_PID $API_PID" EXIT

sleep 15

python src/handler.py  # handler auto-detects test_input.json
