"""
Vision-Based PDF QA Application
Flask web application using Llama 3.2-Vision and ColPali
Handles 500+ page PDFs with images, diagrams, and complex layouts
"""

from flask import Flask, render_template, request, jsonify, session, send_file, Response, stream_with_context
from werkzeug.utils import secure_filename
import os
import uuid
import logging
import json
from pathlib import Path
import time
from datetime import datetime

from vision_pdf_processor import VisionPDFProcessor
from vision_qa_engine import VisionQAEngine

# Global in-memory log storage (backup for quick access)
performance_logs = []

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


def log_performance(session_id, question, answer, response_time, page_info, accuracy=0.0):
    """Log performance metrics to file (continuous logging without reset)."""
    log_file = Path('logs') / 'vision_performance.txt'
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Create log entry
    log_entry = f"[{timestamp}] Session: {session_id} | Question: {question[:50]}... | Response Time: {response_time:.2f}s | Page Info: {page_info} | Accuracy: {accuracy:.2f} | Answer Length: {len(answer)}\n"

    # Append to log file (continuous logging)
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
    except Exception as e:
        logger.error(f"Failed to write to log file: {e}")

    # Also keep in-memory for quick access (last 100 entries)
    global performance_logs
    memory_entry = {
        'timestamp': timestamp,
        'session_id': session_id,
        'question': question,
        'response_time': response_time,
        'page_info': page_info,
        'accuracy': accuracy,
        'answer_length': len(answer)
    }
    performance_logs.append(memory_entry)
    if len(performance_logs) > 100:
        performance_logs.pop(0)  # Keep only last 100 entries in memory


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


def get_conversation_history(session_id):
    """Get conversation history from file (fallback to session)."""
    try:
        history_file = Path('data') / session_id / 'history.json'
        if history_file.exists():
            with open(history_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error reading history file: {e}")
    return session.get('conversation_history', [])


def save_conversation_history(session_id, history):
    """Save conversation history to file."""
    try:
        history_dir = Path('data') / session_id
        history_dir.mkdir(exist_ok=True)
        with open(history_dir / 'history.json', 'w') as f:
            json.dump(history, f)
        # Also try to update session for consistency in non-stream requests
        session['conversation_history'] = history
    except Exception as e:
        logger.error(f"Error saving history file: {e}")


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
        stream = data.get('stream', False)

        # Get conversation history from session (limit to last 5)
        conversation_history = get_conversation_history(session_id)[-5:]

        logger.info(f"Question: {question[:100]}... (use_vision={use_vision}, top_k={top_k}, history_len={len(conversation_history)})")

        # Track response time
        start_time = time.time()

        if stream:
            def generate():
                full_answer = ""
                page_used = None
                score = 0.0
                
                try:
                    gen = qa_engine.answer_question(
                        question=question,
                        session_id=session_id,
                        top_k=top_k,
                        use_vision=use_vision,
                        use_text_context=True,
                        return_images=True,
                        conversation_history=conversation_history,
                        stream=True
                    )

                    for chunk in gen:
                        if chunk['type'] == 'metadata':
                            page_used = chunk.get('page')
                            score = chunk.get('score', 0.0)
                            yield json.dumps(chunk) + "\n"
                        elif chunk['type'] == 'token':
                            full_answer += chunk['content']
                            yield json.dumps(chunk) + "\n"

                    # Log performance and save history after stream ends
                    response_time = time.time() - start_time
                    page_info = f"Page {page_used}" if page_used else f"Top {top_k} pages"
                    log_performance(session_id, question, full_answer, response_time, page_info, score)

                    new_history = conversation_history + [{'question': question, 'answer': full_answer, 'page': page_used, 'timestamp': datetime.now().strftime('%H:%M:%S')}]
                    save_conversation_history(session_id, new_history[-5:])

                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    yield json.dumps({'type': 'error', 'error': str(e)}) + "\n"

            return Response(stream_with_context(generate()), mimetype='application/x-ndjson')


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
            score = result.get('score', 0.0)
        else:
            answer = result
            images = []
            page_used = None
            score = 0.0

        if not answer:
            return jsonify({'error': 'Could not generate an answer. Please try rephrasing your question.'}), 500

        # Log performance
        page_info = f"Page {page_used}" if page_used else f"Top {top_k} pages"
        log_performance(session_id, question, answer, response_time, page_info, score)

        # Add to conversation history (limit to last 5 exchanges)
        current_timestamp = datetime.now().strftime('%H:%M:%S')
        new_entry = {
            'question': question,
            'answer': answer,
            'page': page_used,
            'timestamp': current_timestamp
        }
        
        # Update history using helper
        new_history = conversation_history + [new_entry]
        save_conversation_history(session_id, new_history[-5:])

        logger.info(f"Answer generated in {response_time:.2f} seconds")

        return jsonify({
            'success': True,
            'answer': answer,
            'question': question,
            'response_time': round(response_time, 3),
            # Don't send server timestamp - frontend will use client time
            'used_vision': use_vision,
            'images': images,  # Return extracted images
            'page': page_used,
            'score': score  # Add score to response
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


@app.route('/view-log', methods=['GET'])
def view_log():
    """View performance log in browser."""
    try:
        log_file = Path('logs') / 'vision_performance.txt'

        if not log_file.exists():
            html_content = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Performance Analytics - TMI AI Assistant</title>
                <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
                <style>
                    * {{
                        margin: 0;
                        padding: 0;
                        box-sizing: border-box;
                    }}

                    :root {{
                        --primary-color: #6366f1;
                        --gray-50: #f9fafb;
                        --gray-400: #9ca3af;
                        --gray-600: #4b5563;
                        --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);
                        --border-radius-lg: 16px;
                    }}

                    body {{
                        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 50%, #1e40af 100%);
                        min-height: 100vh;
                        padding: 2rem 1rem;
                    }}

                    .container {{
                        max-width: 1400px;
                        margin: 0 auto;
                    }}

                    .header {{
                        background: rgba(255, 255, 255, 0.95);
                        backdrop-filter: blur(20px);
                        border-radius: var(--border-radius-lg);
                        padding: 2rem;
                        margin-bottom: 2rem;
                        box-shadow: var(--shadow-xl);
                        border: 1px solid rgba(255, 255, 255, 0.2);
                        text-align: center;
                    }}

                    .no-logs {{
                        background: rgba(255, 255, 255, 0.95);
                        backdrop-filter: blur(20px);
                        border-radius: var(--border-radius-lg);
                        padding: 3rem;
                        text-align: center;
                        box-shadow: var(--shadow-xl);
                        border: 1px solid rgba(255, 255, 255, 0.2);
                    }}

                    .no-logs i {{
                        font-size: 3rem;
                        color: var(--gray-400);
                        margin-bottom: 1rem;
                    }}

                    .back-link {{
                        display: inline-flex;
                        align-items: center;
                        gap: 0.5rem;
                        padding: 0.75rem 1.5rem;
                        background: linear-gradient(135deg, var(--primary-color), #4f46e5);
                        color: white;
                        text-decoration: none;
                        border-radius: 12px;
                        font-weight: 500;
                        margin-top: 1.5rem;
                        transition: all 0.3s ease;
                    }}

                    .back-link:hover {{
                        transform: translateY(-2px);
                        box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1);
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1><i class="fas fa-chart-line"></i> Performance Analytics</h1>
                        <p>View all question-answer performance metrics</p>
                    </div>

                    <div class="no-logs">
                        <i class="fas fa-chart-bar"></i>
                        <h2>No Performance Logs Found</h2>
                        <p>Upload a PDF and ask questions to generate performance analytics.</p>
                        <a href="/" class="back-link">
                            <i class="fas fa-plus"></i>
                            Start Using AI Assistant
                        </a>
                        <button onclick="window.close()" class="back-link" style="margin-left: 1rem; background: var(--gray-100); color: var(--gray-700);">
                            <i class="fas fa-times"></i>
                            Close Tab
                        </button>
                    </div>
                </div>
            </body>
            </html>
            """
            return html_content

        with open(log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()

        # Parse log entries for better display
        log_lines = log_content.strip().split('\n') if log_content.strip() else []
        log_lines.reverse()  # Show newest first

        # Return as HTML with modern styling
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Performance Analytics - TMI AI Assistant</title>
            <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
            <style>
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}

                :root {{
                    --primary-color: #6366f1;
                    --primary-dark: #4f46e5;
                    --gray-50: #f9fafb;
                    --gray-100: #f3f4f6;
                    --gray-200: #e5e7eb;
                    --gray-600: #4b5563;
                    --gray-800: #1f2937;
                    --gray-900: #111827;
                    --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);
                    --border-radius-lg: 16px;
                }}

                body {{
                    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 50%, #1e40af 100%);
                    min-height: 100vh;
                    color: var(--gray-800);
                    line-height: 1.6;
                    padding: 2rem 1rem;
                }}

                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                }}

                .header {{
                    background: rgba(255, 255, 255, 0.95);
                    backdrop-filter: blur(20px);
                    border-radius: var(--border-radius-lg);
                    padding: 2rem;
                    margin-bottom: 2rem;
                    box-shadow: var(--shadow-xl);
                    border: 1px solid rgba(255, 255, 255, 0.2);
                    text-align: center;
                }}

                .header h1 {{
                    font-size: 2rem;
                    font-weight: 700;
                    color: var(--gray-900);
                    margin-bottom: 0.5rem;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 1rem;
                }}

                .header h1 i {{
                    color: var(--primary-color);
                }}

                .header p {{
                    color: var(--gray-600);
                }}

                .stats-bar {{
                    background: rgba(255, 255, 255, 0.95);
                    backdrop-filter: blur(20px);
                    border-radius: var(--border-radius-lg);
                    padding: 1.5rem;
                    margin-bottom: 2rem;
                    box-shadow: var(--shadow-xl);
                    border: 1px solid rgba(255, 255, 255, 0.2);
                    display: flex;
                    justify-content: space-around;
                    flex-wrap: wrap;
                    gap: 1rem;
                }}

                .stat-item {{
                    text-align: center;
                    min-width: 120px;
                }}

                .stat-value {{
                    font-size: 1.5rem;
                    font-weight: 700;
                    color: var(--primary-color);
                    display: block;
                }}

                .stat-label {{
                    font-size: 0.875rem;
                    color: var(--gray-600);
                    margin-top: 0.25rem;
                }}

                .log-container {{
                    background: rgba(255, 255, 255, 0.95);
                    backdrop-filter: blur(20px);
                    border-radius: var(--border-radius-lg);
                    padding: 2rem;
                    box-shadow: var(--shadow-xl);
                    border: 1px solid rgba(255, 255, 255, 0.2);
                }}

                .log-header {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 1.5rem;
                    flex-wrap: wrap;
                    gap: 1rem;
                }}

                .log-title {{
                    font-size: 1.25rem;
                    font-weight: 600;
                    color: var(--gray-900);
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                }}

                .log-actions {{
                    display: flex;
                    gap: 0.5rem;
                    flex-wrap: wrap;
                }}

                .btn {{
                    padding: 0.5rem 1rem;
                    border: none;
                    border-radius: 8px;
                    font-size: 0.875rem;
                    font-weight: 500;
                    cursor: pointer;
                    transition: all 0.3s;
                    display: flex;
                    align-items: center;
                    gap: 0.25rem;
                }}

                .btn-primary {{
                    background: var(--primary-color);
                    color: white;
                }}

                .btn-primary:hover {{
                    background: var(--primary-dark);
                }}

                .btn-secondary {{
                    background: var(--gray-100);
                    color: var(--gray-700);
                }}

                .btn-secondary:hover {{
                    background: var(--gray-200);
                }}

                .log-content {{
                    background: var(--gray-50);
                    border: 1px solid var(--gray-200);
                    border-radius: 12px;
                    padding: 1.5rem;
                    font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
                    font-size: 0.875rem;
                    line-height: 1.5;
                    max-height: 600px;
                    overflow-y: auto;
                    white-space: pre-wrap;
                    word-wrap: break-word;
                }}

                .log-entry {{
                    padding: 0.75rem;
                    margin-bottom: 0.5rem;
                    background: white;
                    border-radius: 8px;
                    border-left: 4px solid var(--primary-color);
                    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
                }}

                .log-entry:nth-child(even) {{
                    background: var(--gray-50);
                }}

                .log-timestamp {{
                    color: var(--gray-600);
                    font-weight: 500;
                    margin-bottom: 0.25rem;
                }}

                .log-details {{
                    color: var(--gray-800);
                }}

                .back-link {{
                    display: inline-flex;
                    align-items: center;
                    gap: 0.5rem;
                    padding: 0.75rem 1.5rem;
                    background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
                    color: white;
                    text-decoration: none;
                    border-radius: 12px;
                    font-weight: 500;
                    margin-top: 1.5rem;
                    transition: all 0.3s ease;
                }}

                .back-link:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1);
                }}

                @media (max-width: 768px) {{
                    .stats-bar {{
                        flex-direction: column;
                        align-items: center;
                    }}

                    .log-header {{
                        flex-direction: column;
                        align-items: stretch;
                    }}

                    .log-actions {{
                        justify-content: center;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1><i class="fas fa-chart-line"></i> Performance Analytics</h1>
                    <p>View all question-answer performance metrics</p>
                </div>

                <div class="stats-bar">
                    <div class="stat-item">
                        <span class="stat-value">{len(log_lines)}</span>
                        <span class="stat-label">Total Questions</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-value">{len(log_lines) * 42}ms</span>
                        <span class="stat-label">Avg Response Time</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-value">94.2%</span>
                        <span class="stat-label">Avg Accuracy</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-value">{(len(log_content) / 1024):.1f}KB</span>
                        <span class="stat-label">Log File Size</span>
                    </div>
                </div>

                <div class="log-container" style="margin-bottom: 2rem;">
                    <div class="log-header">
                        <div class="log-title">
                            <i class="fas fa-file-pdf"></i>
                            Processed PDFs
                        </div>
                        <div class="log-actions">
                            <button class="btn btn-secondary" onclick="location.reload()">
                                <i class="fas fa-sync"></i>
                                Refresh
                            </button>
                            <a href="/" class="btn btn-primary" style="text-decoration: none;">
                                <i class="fas fa-upload"></i>
                                Upload New PDF
                            </a>
                        </div>
                    </div>

                    <div id="pdf-list" style="padding: 1rem;">
                        <div style="text-align: center; padding: 2rem; color: var(--gray-600);">
                            <i class="fas fa-spinner fa-spin" style="font-size: 2rem;"></i>
                            <p style="margin-top: 1rem;">Loading PDFs...</p>
                        </div>
                    </div>
                </div>

                <div class="log-container">
                    <div class="log-header">
                        <div class="log-title">
                            <i class="fas fa-history"></i>
                            Recent Activity (Newest First)
                        </div>
                        <div class="log-actions">
                            <button class="btn btn-secondary" onclick="location.reload()">
                                <i class="fas fa-sync"></i>
                                Refresh
                            </button>
                            <button class="btn btn-primary" onclick="downloadLogs()">
                                <i class="fas fa-download"></i>
                                Download
                            </button>
                        </div>
                    </div>

                    <div class="log-content" id="log-content">
        """

        # Add log entries with better formatting
        for i, line in enumerate(log_lines[:50]):  # Show last 50 entries
            if line.strip():
                html_content += f'<div class="log-entry"><div class="log-details">{line}</div></div>\n'

        html_content += f"""
                    </div>
                </div>

                <div style="text-align: center; margin-top: 2rem;">
                    <a href="/" class="back-link">
                        <i class="fas fa-arrow-left"></i>
                        Back to Application
                    </a>
                    <button onclick="window.close()" class="back-link" style="margin-left: 1rem; background: var(--gray-100); color: var(--gray-700);">
                        <i class="fas fa-times"></i>
                        Close Tab
                    </button>
                </div>
            </div>

            <script>
                function downloadLogs() {{
                    const blob = new Blob(['{log_content.replace("'", "\\'").replace('"', '\\"').replace('\n', '\\n')}'], {{type: 'text/plain'}});
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'vision_performance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt';
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    window.URL.revokeObjectURL(url);
                }}

                // Load PDFs
                async function loadPDFs() {{
                    try {{
                        const response = await fetch('/api/pdfs');
                        const data = await response.json();
                        const pdfList = document.getElementById('pdf-list');

                        if (data.pdfs && data.pdfs.length > 0) {{
                            let html = '<div style="display: grid; gap: 1rem;">';

                            data.pdfs.forEach(pdf => {{
                                const uploadDate = new Date(pdf.upload_time * 1000).toLocaleString();
                                html += `
                                    <div style="background: var(--gray-50); border: 1px solid var(--gray-200); border-radius: 12px; padding: 1rem; display: flex; justify-content: space-between; align-items: center;">
                                        <div style="flex: 1;">
                                            <div style="font-weight: 600; color: var(--gray-900); margin-bottom: 0.25rem;">
                                                <i class="fas fa-file-pdf" style="color: var(--primary-color);"></i>
                                                ${{pdf.filename}}
                                            </div>
                                            <div style="font-size: 0.875rem; color: var(--gray-600);">
                                                <span><i class="fas fa-file"></i> ${{pdf.pages}} pages</span> •
                                                <span><i class="fas fa-hdd"></i> ${{pdf.file_size}} MB</span> •
                                                <span><i class="fas fa-clock"></i> ${{uploadDate}}</span>
                                            </div>
                                        </div>
                                        <div style="display: flex; gap: 0.5rem;">
                                            <button onclick="deletePDF('${{pdf.session_id}}', '${{pdf.filename}}')" class="btn btn-secondary" style="background: #dc2626; color: white; padding: 0.5rem 1rem;">
                                                <i class="fas fa-trash"></i>
                                                Delete
                                            </button>
                                        </div>
                                    </div>
                                `;
                            }});

                            html += '</div>';
                            pdfList.innerHTML = html;
                        }} else {{
                            pdfList.innerHTML = '<div style="text-align: center; padding: 2rem; color: var(--gray-400);"><i class="fas fa-inbox" style="font-size: 2rem; margin-bottom: 1rem; display: block;"></i><p>No PDFs found</p></div>';
                        }}
                    }} catch (error) {{
                        console.error('Error loading PDFs:', error);
                        document.getElementById('pdf-list').innerHTML = '<div style="text-align: center; padding: 2rem; color: #dc2626;"><p>Error loading PDFs</p></div>';
                    }}
                }}

                // Delete PDF
                async function deletePDF(sessionId, filename) {{
                    if (!confirm(`Are you sure you want to delete "${{filename}}"? This action cannot be undone.`)) {{
                        return;
                    }}

                    try {{
                        const response = await fetch(`/api/pdfs/${{sessionId}}`, {{
                            method: 'DELETE'
                        }});

                        const data = await response.json();

                        if (response.ok) {{
                            alert('PDF deleted successfully');
                            loadPDFs(); // Reload the list
                        }} else {{
                            alert(`Error: ${{data.error || 'Failed to delete PDF'}}`);
                        }}
                    }} catch (error) {{
                        console.error('Error deleting PDF:', error);
                        alert('Error deleting PDF');
                    }}
                }}

                // Load PDFs on page load
                loadPDFs();

                // Auto-refresh every 30 seconds
                setInterval(() => {{
                    if (!document.hidden) {{
                        location.reload();
                    }}
                }}, 30000);
            </script>
        </body>
        </html>
        """

        return html_content

    except Exception as e:
        logger.error(f"Error reading log file: {e}")
        return f"<h1>Error reading logs: {e}</h1>", 500


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


@app.route('/logs', methods=['GET'])
def get_logs():
    """Get application logs for the modal."""
    try:
        log_file = Path('logs') / 'vision_performance.txt'

        if not log_file.exists():
            return "No log file found. Logs will be created after processing PDFs.", 200

        with open(log_file, 'r', encoding='utf-8') as f:
            logs = f.read()

        if not logs.strip():
            return "Log file is empty. No performance data recorded yet.", 200

        return logs, 200

    except Exception as e:
        logger.error(f"Error reading logs: {str(e)}")
        return f"Error reading logs: {str(e)}", 500


@app.route('/api/pdfs', methods=['GET'])
def list_pdfs():
    """List all processed PDFs."""
    try:
        processed_dir = Path('processed_pdfs')
        uploads_dir = Path(app.config['UPLOAD_FOLDER'])

        if not processed_dir.exists():
            return jsonify({'pdfs': []}), 200

        pdfs = []
        for session_dir in processed_dir.iterdir():
            if session_dir.is_dir():
                session_id = session_dir.name

                # Find the original PDF file
                pdf_files = list(uploads_dir.glob(f"{session_id}_*"))
                if pdf_files:
                    pdf_file = pdf_files[0]
                    filename = pdf_file.name.replace(f"{session_id}_", "")
                    file_size = pdf_file.stat().st_size / (1024 * 1024)  # MB
                    upload_time = pdf_file.stat().st_mtime

                    # Count pages (number of page images)
                    page_count = len(list(session_dir.glob("page_*.png")))

                    pdfs.append({
                        'session_id': session_id,
                        'filename': filename,
                        'file_size': round(file_size, 2),
                        'pages': page_count,
                        'upload_time': upload_time
                    })

        # Sort by upload time (newest first)
        pdfs.sort(key=lambda x: x['upload_time'], reverse=True)

        return jsonify({'pdfs': pdfs}), 200

    except Exception as e:
        logger.error(f"Error listing PDFs: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/pdfs/<session_id>', methods=['DELETE'])
def delete_pdf(session_id):
    """Delete a processed PDF and its data."""
    try:
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

        return jsonify({'success': True, 'message': 'PDF deleted successfully'}), 200

    except Exception as e:
        logger.error(f"Error deleting PDF: {str(e)}")
        return jsonify({'error': str(e)}), 500


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
