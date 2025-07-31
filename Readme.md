# Test the worker

```
docker build -t mybnb-worker .

docker run --rm -p 8000:8000 -p 11434:11434 mybnb-worker ./start.sh
```

# Manual testing

- use python 3.10
- install dependencies from builder/requirements.txt
- Run python src/my_server.py
- Wait for it to be running
- `curl -X POST "http://localhost:8000/generate" -H "Content-Type: application/json" -d '{"query": "2 bedroom apartment with tea kettle"}'`
