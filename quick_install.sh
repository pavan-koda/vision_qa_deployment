#!/bin/bash
# Super fast install - no checks, just install everything

echo "Quick Installing Vision PDF QA Dependencies..."
echo

# Activate or create venv
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Install everything quickly
pip install --upgrade pip

echo "Installing packages (this takes 2-3 minutes)..."
pip install \
    Flask==3.0.0 \
    Werkzeug==3.0.1 \
    PyMuPDF==1.23.8 \
    Pillow==10.1.0 \
    chromadb==0.4.22 \
    transformers==4.36.2 \
    torch==2.1.2 \
    sentence-transformers==2.2.2 \
    faiss-cpu==1.7.4 \
    numpy==1.26.2 \
    requests==2.31.0 \
    ollama==0.1.6 \
    tqdm==4.66.1

echo
echo "✅ Installation complete!"
echo
echo "Run: python app_vision.py"
