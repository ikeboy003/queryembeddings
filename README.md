# Semantic Query Router / Embedding Cache Service

A Flask microservice that acts as a semantic query router and embedding cache. It accepts user queries, generates embeddings, checks ChromaDB for similar cached queries, and returns the cached query if found, otherwise stores the new query.

## Features

- **Embedding Generation**: Uses Ollama for local embedding generation
- **Semantic Caching**: Checks ChromaDB for similar queries before returning cached results
- **REST API**: Flask-based endpoints for easy integration
- **Auto-reload**: Development mode with automatic reload on code changes

## Setup

1. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure Ollama is running locally (default: http://localhost:11434)

4. Configure environment variables (optional, defaults provided):
```bash
# .env file
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=gemma3:270m
CHROMA_COLLECTION_NAME=query_embeddings
SIMILARITY_THRESHOLD=0.85
```

## Usage

### Start the API Server

**Option 1: Direct Flask run (development with auto-reload)**
```bash
python3 app.py
```

**Option 2: Development script**
```bash
python3 run_dev.py
```

**Option 3: Gunicorn with auto-reload (production-like)**
```bash
./run.sh
# or
gunicorn --bind 0.0.0.0:8000 --workers 2 --reload app:app
```

### API Endpoints

**POST /query**
Process a user query through the semantic cache.

Request:
```json
{
  "query": "What are Jokic's stats tonight?"
}
```

Response:
```json
{
  "query": "What are Jokic's stats tonight?"
}
```

If a similar query was cached (similarity >= 0.85), you'll get the cached query string. Otherwise, you'll get the original query back and it will be stored for future use.

**GET /health**
Health check endpoint.
```json
{"status": "healthy"}
```

**GET /**
Service info endpoint.
```json
{
  "service": "Semantic Query Router",
  "status": "running",
  "version": "1.0.0"
}
```

## Project Structure

```
queryembeddings/
├── app.py                    # Flask application entrypoint
├── core/
│   ├── container.py         # Dependency injection container
│   └── settings.py          # Configuration from environment
├── providers/
│   └── ollama_provider.py   # Ollama embedding generation
├── storage/
│   └── chroma_store.py      # ChromaDB storage implementation
├── services/
│   ├── semantic_service.py  # Core orchestrator
│   └── similarity.py        # Cosine similarity utilities
├── tests/
│   ├── test_embedding.py    # Embedding tests
│   └── test_storage.py      # Storage tests
├── chroma_db/               # ChromaDB data directory
├── requirements.txt
└── run.sh                   # Gunicorn run script
```

## Testing

Run tests with pytest:
```bash
python3 -m pytest tests/
```

## Workflow

1. User sends query → Flask receives POST /query
2. Generate embedding using Ollama
3. Check ChromaDB for similar embeddings (above threshold)
4. If found: Return cached query string
5. If not found: Return original query and store it in ChromaDB

## Running as Daemon

The app automatically reloads when code changes are detected. To run in background:

```bash
# With gunicorn
nohup gunicorn --bind 0.0.0.0:8000 --workers 2 --reload app:app > app.log 2>&1 &

# Or with Flask dev server
nohup python3 app.py > app.log 2>&1 &
```
