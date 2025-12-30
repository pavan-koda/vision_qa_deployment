#!/bin/bash
# Simple dependency installer for Vision PDF QA System

echo "========================================="
echo "Installing Vision PDF QA Dependencies"
echo "========================================="
echo

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "✓ Activating virtual environment..."
    source venv/bin/activate
else
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✓ Virtual environment created"
fi

echo
echo "Upgrading pip..."
pip install --upgrade pip

echo
echo "Installing core dependencies..."
pip install Flask==3.0.0
pip install Werkzeug==3.0.1
pip install requests==2.31.0

echo
echo "Installing PDF processing libraries..."
pip install PyMuPDF==1.23.8
pip install pypdf==3.17.4
pip install Pillow==10.1.0

echo
echo "Installing AI/ML libraries..."
pip install transformers==4.36.2
pip install torch==2.1.2
pip install torchvision==0.16.2
pip install sentence-transformers==2.2.2

echo
echo "Installing vector databases..."
pip install chromadb==0.4.22
pip install hnswlib==0.8.0
pip install pydantic==2.5.3
pip install faiss-cpu==1.7.4

echo
echo "Installing Ollama client..."
pip install ollama==0.1.6

echo
echo "Installing utilities..."
pip install numpy==1.26.2
pip install tqdm==4.66.1
pip install python-dotenv==1.0.0
pip install python-json-logger==2.0.7
pip install colorlog==6.8.0

echo
echo "========================================="
echo "Verifying installation..."
echo "========================================="

python << EOF
import sys
try:
    import fitz
    print("✓ PyMuPDF (fitz) installed")
except ImportError:
    print("✗ PyMuPDF (fitz) FAILED")
    sys.exit(1)

try:
    import chromadb
    print("✓ ChromaDB installed")
except ImportError:
    print("✗ ChromaDB FAILED")
    sys.exit(1)

try:
    import flask
    print("✓ Flask installed")
except ImportError:
    print("✗ Flask FAILED")
    sys.exit(1)

try:
    import torch
    print("✓ PyTorch installed")
except ImportError:
    print("✗ PyTorch FAILED")
    sys.exit(1)

try:
    import transformers
    print("✓ Transformers installed")
except ImportError:
    print("✗ Transformers FAILED")
    sys.exit(1)

try:
    import ollama
    print("✓ Ollama client installed")
except ImportError:
    print("✗ Ollama client FAILED")
    sys.exit(1)

print()
print("✅ All critical dependencies installed successfully!")
EOF

if [ $? -eq 0 ]; then
    echo
    echo "========================================="
    echo "✅ Installation Complete!"
    echo "========================================="
    echo
    echo "You can now run:"
    echo "  python app_vision.py"
    echo
    echo "Or use the startup script:"
    echo "  ./start_vision_app.sh"
    echo
else
    echo
    echo "❌ Installation failed. Please check errors above."
    exit 1
fi
