"""Flask application entrypoint."""
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from core.container import container

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)


@app.route("/")
def root():
    """Health check endpoint."""
    return jsonify({
        "service": "Semantic Query Router",
        "status": "running",
        "version": "1.0.0"
    })


@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})


@app.route("/query", methods=["POST"])
def query():
    """
    Process a user query through the semantic cache.
    
    - Generates embedding for the query
    - Checks for similar cached queries
    - Returns cached query string if found above threshold
    - Otherwise returns the original query and stores it
    
    Request JSON: {"query": "string"}
    Response JSON: {"query": "string"}
    """
    try:
        data = request.get_json()
        if not data or "query" not in data:
            return jsonify({"error": "Missing 'query' field"}), 400
        
        query_text = data["query"]
        if not isinstance(query_text, str) or not query_text.strip():
            return jsonify({"error": "Query must be a non-empty string"}), 400
        
        # Get semantic service from container
        semantic_service = container.semantic_service
        
        # Process the query - returns query string
        result_query = semantic_service.process_query(query_text)
        
        return jsonify({"query": result_query})
    
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


if __name__ == "__main__":
    # Development mode with auto-reload
    app.run(host="0.0.0.0", port=8000, debug=True, use_reloader=True)
