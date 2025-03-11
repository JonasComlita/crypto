#!/bin/bash
# For Linux/Mac

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    echo "Installing macOS dependencies..."
    pip install -r requirements-mac.txt
    
    # Install additional macOS system dependencies
    brew install python-tk
    brew install openssl
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    echo "Installing Linux dependencies..."
    
    # Install system dependencies
    sudo apt-get update
    sudo apt-get install -y python3-dev python3-tk
    
    # Install Python dependencies
    pip install -r requirements-linux.txt
fi 