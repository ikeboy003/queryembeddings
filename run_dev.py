"""Run script for development with auto-reload."""
import os
import sys

if __name__ == "__main__":
    # Enable auto-reload in Flask development server
    os.environ['FLASK_ENV'] = 'development'
    os.environ['FLASK_DEBUG'] = '1'
    
    from app import app
    app.run(host="0.0.0.0", port=8000, debug=True, use_reloader=True)
