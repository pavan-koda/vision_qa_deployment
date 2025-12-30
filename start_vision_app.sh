#!/bin/bash
# Vision-Based PDF QA System Startup Script for Linux/macOS
# Handles Ollama setup and application launch

set -e

echo "========================================================================"
echo "   VISION-BASED PDF QA SYSTEM"
echo "   Powered by Llama 3.2-Vision + ColPali"
echo "========================================================================"
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}ERROR: Python 3 is not installed${NC}"
    echo "Please install Python 3.9+ from https://www.python.org"
    exit 1
fi

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo -e "${RED}ERROR: Ollama is not installed${NC}"
    echo
    echo "Please install Ollama:"
    echo "  macOS: brew install ollama"
    echo "  Linux: curl -fsSL https://ollama.ai/install.sh | sh"
    echo "  Or visit: https://ollama.ai/download"
    exit 1
fi

echo "[1/6] Checking Ollama installation..."
ollama --version
echo

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "[2/6] Creating virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}Virtual environment created successfully${NC}"
else
    echo "[2/6] Virtual environment already exists"
fi
echo

# Activate virtual environment
echo "[3/6] Activating virtual environment..."
source venv/bin/activate
echo

# Check if dependencies are installed
echo "[4/6] Checking dependencies..."

# Quick check for PyMuPDF only (fastest check)
if ! python -c "import fitz" 2>/dev/null; then
    echo "PyMuPDF not found. Installing all dependencies..."
    pip install --upgrade pip --quiet

    # Install dependencies one by one to avoid hanging
    echo "Installing core packages..."
    pip install Flask Werkzeug requests --quiet

    echo "Installing PDF processing..."
    pip install PyMuPDF Pillow --quiet

    echo "Installing AI libraries (this may take a few minutes)..."
    pip install chromadb transformers sentence-transformers --quiet

    echo "Installing remaining packages..."
    pip install torch faiss-cpu numpy tqdm ollama --quiet

    echo -e "${GREEN}Dependencies installed successfully${NC}"
else
    echo "PyMuPDF found. Verifying other dependencies..."
    # Install any missing dependencies without checking (faster)
    pip install -r requirements_vision.txt --quiet 2>/dev/null || true
    echo "Dependencies verified"
fi
echo

# Start Ollama if not running
echo "[5/6] Starting Ollama server..."
if ! curl -s http://localhost:11434/api/tags &>/dev/null; then
    echo "Ollama not running, starting Ollama..."

    # Try systemd first (Linux)
    if command -v systemctl &> /dev/null; then
        sudo systemctl start ollama || ollama serve &
    else
        # macOS or systems without systemd
        ollama serve &
    fi

    sleep 5
    echo -e "${GREEN}Ollama server started${NC}"
else
    echo "Ollama server already running"
fi
echo

# Check if Llama 3.2-Vision model is installed
echo "[6/6] Checking Llama 3.2-Vision model..."
if ! ollama list | grep -q "llama3.2-vision"; then
    echo
    echo "========================================================================"
    echo -e "${YELLOW}WARNING: Llama 3.2-Vision model not found${NC}"
    echo "========================================================================"
    echo
    echo "The vision model needs to be downloaded (~7GB)"
    echo "This is a one-time download and will take several minutes."
    echo
    read -p "Download llama3.2-vision:11b now? (y/n): " download

    if [[ "$download" == "y" || "$download" == "Y" ]]; then
        echo
        echo "Downloading Llama 3.2-Vision 11B model..."
        echo "This may take 5-15 minutes depending on your internet speed."
        echo
        ollama pull llama3.2-vision:11b
        echo
        echo -e "${GREEN}Model downloaded successfully!${NC}"
        echo
    else
        echo
        echo "Skipping model download."
        echo "Note: The application will not work without the vision model."
        echo "To download later, run: ollama pull llama3.2-vision:11b"
        echo
    fi
else
    echo "Llama 3.2-Vision model found"
fi
echo

echo "========================================================================"
echo "   STARTING APPLICATION"
echo "========================================================================"
echo
echo "Server will start at: http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo
echo "========================================================================"
echo

# Start the application
python app_vision.py

# Cleanup on exit
trap "echo 'Shutting down...'; deactivate; exit" INT TERM
