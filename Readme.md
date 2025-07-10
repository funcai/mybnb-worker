# Test the worker

```
docker build -t mybnb-worker .

docker run --rm -p 8000:8000 -p 11434:11434 mybnb-worker ./start.sh
```
