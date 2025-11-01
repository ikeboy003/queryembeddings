"""Flask application entrypoint."""
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from core.container import container

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize services on startup
try:
    _ = container.storage
    _ = container.embedding_provider
    logger.info("Service initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize service: {e}")
    raise


def _validate_query(data: dict | None) -> str:
    """Validate and extract query from request data."""
    if not data or "query" not in data:
        raise ValueError("Missing 'query' field")
    
    query_text = data["query"]
    if not isinstance(query_text, str) or not query_text.strip():
        raise ValueError("Query must be a non-empty string")
    
    return query_text


@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({
        "service": "Semantic Query Router",
        "status": "healthy",
        "version": "1.0.0"
    })


@app.route("/query", methods=["POST"])
def query():
    """
    Process a user query through the semantic cache.
    
    Request JSON: {"query": "string"}
    Response JSON: {"query": "string"}
    """
    try:
        query_text = _validate_query(request.get_json())
        result_query = container.semantic_service.process_query(query_text)
        return jsonify({"query": result_query})
    
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True, use_reloader=True)
