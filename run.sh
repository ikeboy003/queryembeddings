#!/bin/bash
# Run Flask app with auto-reload using gunicorn
# This runs as a daemon with auto-reload on file changes

cd "$(dirname "$0")"
gunicorn --bind 0.0.0.0:8000 --workers 2 --reload --log-level info app:app
