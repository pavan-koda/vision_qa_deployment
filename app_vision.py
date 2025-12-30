"""
Vision-Based PDF QA Application
Flask web application using Llama 3.2-Vision and ColPali
Handles 500+ page PDFs with images, diagrams, and complex layouts
"""

from flask import Flask, render_template, request, jsonify, session, send_file
from werkzeug.utils import secure_filename
import os
import uuid
import logging
from pathlib import Path
import time
from datetime import datetime

from vision_pdf_processor import VisionPDFProcessor
from vision_qa_engine import VisionQAEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Create necessary directories
for directory in ['uploads', 'data', 'logs', 'processed_pdfs', 'chroma_db']:
    os.makedirs(directory, exist_ok=True)

# Initialize Vision QA Engine
try:
    qa_engine = VisionQAEngine(
        ollama_url=os.getenv('OLLAMA_URL', 'http://localhost:11434'),
        model_name=os.getenv('VISION_MODEL', 'llama3.2-vision:11b'),
        chroma_persist_dir='chroma_db',
        use_colpali=True
    )
    logger.info("Vision QA Engine initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Vision QA Engine: {str(e)}")
    qa_engine = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def log_performance(session_id, question, answer, response_time, page_info):
    """Log performance metrics with latest entries on top - simplified format."""
    log_file = Path('logs') / 'vision_performance.txt'
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Simplified log entry - only essential metrics, no full response
    log_entry = f"""
{'='*80}
Timestamp: {timestamp}
Session ID: {session_id}
Question: {question}
Response Time: {response_time:.3f} seconds
Pages Used: {page_info}
Answer Length: {len(answer)} characters
{'='*80}

"""

    # Read existing content if file exists
    existing_content = ""
    if log_file.exists():
        with open(log_file, 'r', encoding='utf-8') as f:
            existing_content = f.read()

    # Write new entry at the top, followed by existing content
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(log_entry)
        if existing_content:
            f.write(existing_content)


@app.route('/')
def index():
    return render_template('vision_upload.html')


@app.route('/qa')
def qa_page():
    if 'session_id' not in session:
        return render_template('vision_upload.html')

    metadata = session.get('metadata', {})
    return render_template('vision_qa.html', metadata=metadata)


@app.route('/upload', methods=['POST'])
def upload_pdf():
    """Upload and process PDF with vision capabilities."""
    try:
        if not qa_engine:
            return jsonify({'error': 'Vision QA Engine not initialized. Please check Ollama is running.'}), 500

        # Check if file is present
        if 'pdf_file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['pdf_file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload a PDF file.'}), 400

        # Generate unique session ID
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id

        # Save the PDF file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
        file.save(filepath)

        logger.info(f"PDF uploaded: {filepath}")
        logger.info(f"File size: {os.path.getsize(filepath) / (1024*1024):.2f} MB")

        # Process PDF with vision
        output_dir = os.path.join('processed_pdfs', session_id)
        os.makedirs(output_dir, exist_ok=True)

        # Get processing options
        dpi = int(request.form.get('dpi', 150))
        extract_images = request.form.get('extract_images', 'true').lower() == 'true'

        logger.info(f"Processing PDF with DPI={dpi}, extract_images={extract_images}")

        processor = VisionPDFProcessor(
            dpi=dpi,
            extract_images=extract_images,
            extract_text=True,
            batch_size=10
        )

        # Process PDF
        start_time = time.time()
        result = processor.process_pdf(filepath, output_dir)
        processing_time = time.time() - start_time

        if not result:
            os.remove(filepath)
            return jsonify({'error': 'Failed to process PDF. Please check the file.'}), 400

        logger.info(f"PDF processed in {processing_time:.2f} seconds")

        # Create ChromaDB collection and ColPali index
        logger.info("Creating vector index...")
        index_start = time.time()

        success = qa_engine.create_collection(
            session_id=session_id,
            page_images=result['page_images'],
            page_texts=result['page_text'],
            metadata=result['metadata']
        )

        index_time = time.time() - index_start

        if not success:
            return jsonify({'error': 'Failed to create search index'}), 500

        logger.info(f"Index created in {index_time:.2f} seconds")

        # Save metadata to session
        metadata = {
            'filename': filename,
            'total_pages': result['metadata']['total_pages'],
            'num_page_images': len(result['page_images']),
            'num_embedded_images': len(result['embedded_images']),
            'processing_time': round(processing_time, 2),
            'index_time': round(index_time, 2),
            'session_id': session_id
        }
        session['metadata'] = metadata
        session['conversation_history'] = []  # Initialize empty conversation history

        return jsonify({
            'success': True,
            'message': f'PDF processed successfully! {metadata["total_pages"]} pages indexed.',
            'metadata': metadata
        }), 200

    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
        return jsonify({'error': f'Error processing PDF: {str(e)}'}), 500


@app.route('/ask', methods=['POST'])
def ask_question():
    """Answer question using vision AI."""
    try:
        if not qa_engine:
            return jsonify({'error': 'Vision QA Engine not available'}), 500

        data = request.get_json()

        if not data or 'question' not in data:
            return jsonify({'error': 'No question provided'}), 400

        question = data['question'].strip()

        if not question:
            return jsonify({'error': 'Question cannot be empty'}), 400

        if len(question) > 1000:
            return jsonify({'error': 'Question is too long. Maximum 1000 characters.'}), 400

        session_id = session.get('session_id')

        if not session_id:
            return jsonify({'error': 'No PDF uploaded. Please upload a PDF first.'}), 400

        # Options
        use_vision = data.get('use_vision', True)
        top_k = int(data.get('top_k', 5))

        # Get conversation history from session (limit to last 5)
        conversation_history = session.get('conversation_history', [])[-5:]

        logger.info(f"Question: {question[:100]}... (use_vision={use_vision}, top_k={top_k}, history_len={len(conversation_history)})")

        # Track response time
        start_time = time.time()

        # Generate answer with conversation history
        result = qa_engine.answer_question(
            question=question,
            session_id=session_id,
            top_k=top_k,
            use_vision=use_vision,
            use_text_context=True,
            return_images=True,  # Get images from relevant pages
            conversation_history=conversation_history  # Pass conversation context
        )

        response_time = time.time() - start_time

        # Handle result (can be string or dict with images)
        if isinstance(result, dict):
            answer = result.get('answer', '')
            images = result.get('images', [])
            page_used = result.get('page', None)
        else:
            answer = result
            images = []
            page_used = None

        if not answer:
            return jsonify({'error': 'Could not generate an answer. Please try rephrasing your question.'}), 500

        # Log performance
        page_info = f"Page {page_used}" if page_used else f"Top {top_k} pages"
        log_performance(session_id, question, answer, response_time, page_info)

        # Add to conversation history (limit to last 5 exchanges)
        current_timestamp = datetime.now().strftime('%H:%M:%S')
        if 'conversation_history' not in session:
            session['conversation_history'] = []

        session['conversation_history'].append({
            'question': question,
            'answer': answer,
            'page': page_used,
            'timestamp': current_timestamp
        })

        # Keep only last 5 exchanges
        session['conversation_history'] = session['conversation_history'][-5:]

        logger.info(f"Answer generated in {response_time:.2f} seconds")

        return jsonify({
            'success': True,
            'answer': answer,
            'question': question,
            'response_time': round(response_time, 3),
            # Don't send server timestamp - frontend will use client time
            'used_vision': use_vision,
            'images': images,  # Return extracted images
            'page': page_used
        }), 200

    except Exception as e:
        logger.error(f"Error answering question: {str(e)}", exc_info=True)
        return jsonify({'error': f'Error generating answer: {str(e)}'}), 500


@app.route('/reset', methods=['POST'])
def reset_session():
    """Reset session and cleanup."""
    try:
        session_id = session.get('session_id')

        if session_id:
            # Clean up ChromaDB collection
            if qa_engine:
                qa_engine.cleanup_session(session_id)

            # Clean up files
            upload_dir = Path(app.config['UPLOAD_FOLDER'])
            for file in upload_dir.glob(f"{session_id}_*"):
                file.unlink()

            # Clean up processed PDFs
            processed_dir = Path('processed_pdfs') / session_id
            if processed_dir.exists():
                import shutil
                shutil.rmtree(processed_dir)

        session.clear()

        return jsonify({'success': True, 'message': 'Session reset successfully'}), 200

    except Exception as e:
        logger.error(f"Error resetting session: {str(e)}")
        return jsonify({'error': f'Error resetting session: {str(e)}'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    health_status = {
        'status': 'healthy',
        'qa_engine': qa_engine is not None
    }

    if qa_engine:
        # Check Ollama connection
        health_status['ollama_connected'] = qa_engine._check_ollama_connection()

    return jsonify(health_status), 200


@app.route('/download-log', methods=['GET'])
def download_log():
    """Download performance log."""
    try:
        log_file = Path('logs') / 'vision_performance.txt'

        if not log_file.exists():
            return jsonify({'error': 'No performance log found'}), 404

        return send_file(
            log_file,
            as_attachment=True,
            download_name=f"vision_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mimetype='text/plain'
        )

    except Exception as e:
        logger.error(f"Error downloading log: {str(e)}")
        return jsonify({'error': f'Error downloading log: {str(e)}'}), 500


@app.route('/data/<session_id>/embedded_images/<filename>')
def serve_embedded_image(session_id, filename):
    """Serve extracted embedded images from PDF."""
    try:
        image_path = Path('data') / session_id / 'embedded_images' / filename

        if not image_path.exists():
            return jsonify({'error': 'Embedded image not found'}), 404

        return send_file(image_path, mimetype='image/png')

    except Exception as e:
        logger.error(f"Error serving embedded image: {str(e)}")
        return jsonify({'error': f'Error serving embedded image: {str(e)}'}), 500


@app.route('/data/<session_id>/<filename>')
def serve_page_image(session_id, filename):
    """Serve page images (stored in processed_pdfs directory)."""
    try:
        # Page images are stored in processed_pdfs/{session_id}/page_XXXX.png
        image_path = Path('processed_pdfs') / session_id / filename

        if not image_path.exists():
            logger.error(f"Page image not found: {image_path}")
            return jsonify({'error': 'Page image not found'}), 404

        return send_file(image_path, mimetype='image/png')

    except Exception as e:
        logger.error(f"Error serving page image: {str(e)}")
        return jsonify({'error': f'Error serving page image: {str(e)}'}), 500


if __name__ == '__main__':
    # Check if Ollama is available
    if qa_engine and qa_engine._check_ollama_connection():
        logger.info("="*80)
        logger.info("Vision PDF QA System Starting")
        logger.info("="*80)
        logger.info(f"Ollama URL: {qa_engine.ollama_url}")
        logger.info(f"Vision Model: {qa_engine.model_name}")
        logger.info(f"ColPali Enabled: {qa_engine.use_colpali}")
        logger.info("="*80)
        logger.info("Server running at: http://localhost:5000")
        logger.info("="*80)

        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        logger.error("="*80)
        logger.error("STARTUP FAILED")
        logger.error("="*80)
        logger.error("Ollama is not running or Llama 3.2-Vision is not installed")
        logger.error("")
        logger.error("Please follow these steps:")
        logger.error("1. Install Ollama: https://ollama.ai/download")
        logger.error("2. Start Ollama: ollama serve")
        logger.error("3. Pull model: ollama pull llama3.2-vision:11b")
        logger.error("4. Run this app again")
        logger.error("="*80)
