# myBnB Worker

An intelligent apartment search service that uses AI to match user queries with apartments based on both textual descriptions and image analysis. This worker service processes natural language queries and returns ranked apartment listings that best match the user's requirements.

The frontend code can be found at [https://github.com/funcai/mybnb-web](https://github.com/funcai/mybnb-web).

## How It Works

1. **Query Processing**: User queries are parsed to extract:

   - Quantitative filters (rooms, rent, area, beds, etc.)
   - Qualitative questions (amenities, features, preferences)

2. **Initial Filtering**: Apartments are filtered based on quantitative criteria to create a candidate pool

3. **AI Scoring**: Each candidate is scored using:

   - **Text Analysis**: LLM evaluates apartment descriptions against qualitative questions
   - **Vision Analysis**: Vision model analyzes apartment images for visual confirmation

4. **Ranking**: Apartments are ranked by combined scores and the top 50 results are returned

### Image + Text Classifier

We train a classifier to find out if a image satisfies a criterion. In this part we use Gemma 3n as embedding model (from the last hidden layer) and train a classifier on top that uses attention-based pooling of the sequence dimension.

The code for classifier training can be found at [finetune/train_classifier.ipynb](finetune/train_classifier.ipynb).

## API Endpoints

### POST `/generate`

Main endpoint for apartment search queries.

**Request Body:**

```json
{
  "query": "2 bedroom apartment with tea kettle near downtown"
}
```

**Response:**

```json
{
  "apartments": [
    {
      "id": "apt_123",
      "url": "https://example.com/apartment/123",
      "provider": "example_provider",
      "address": {...},
      "facts": {...},
      "overall_score": 2.5,
      "score_details": [
        {
          "question": "Does it have a tea kettle?",
          "score": 1.0,
          "explanation": "yes; Vision: Image shows matching content",
          "keyword": "tea kettle"
        }
      ]
    }
  ]
}
```

### GET `/healthz`

Health check endpoint that returns `{"status": "ok"}`.

## Getting Started

### Docker Deployment (Recommended)

```bash
# Build the Docker image
docker build -t mybnb-worker .

# Run the service
docker run --rm -p 8000:8000 -p 11434:11434 mybnb-worker ./start.sh
```

The service will be available at `http://localhost:8000`.

### Local Development

**Requirements:**

- Python 3.10
- Dependencies from `builder/requirements.txt`

**Setup:**

```bash
# Install dependencies
pip install -r builder/requirements.txt

# Run the server
python src/my_server.py
```

### Testing the Service

Once the service is running, test it with a sample query:

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"query": "2 bedroom apartment with tea kettle"}'
```
