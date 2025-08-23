#!/usr/bin/env python3
"""
AlphaPente Web Interface Launcher

Run this script to start the web interface for playing Pente against AI.
"""

import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from web.app import create_app

if __name__ == '__main__':
    app = create_app()
    
    print("🔴 AlphaPente Web Interface")
    print("=" * 40)
    print("Starting Flask development server...")
    print("Open your browser to: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the server")
    print("=" * 40)
    
    app.run(debug=True, host='127.0.0.1', port=5000)