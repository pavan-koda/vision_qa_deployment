// DOM Elements
const uploadArea = document.getElementById('upload-area');
const pdfInput = document.getElementById('pdf-input');
const browseBtn = document.getElementById('browse-btn');
const fileInfo = document.getElementById('file-info');
const filenameSpan = document.getElementById('filename');
const uploadBtn = document.getElementById('upload-btn');
const cancelBtn = document.getElementById('cancel-btn');
const statusSection = document.getElementById('status-section');
const statusMessage = document.getElementById('status-message');
const qaSection = document.getElementById('qa-section');
const chatHistory = document.getElementById('chat-history');
const questionInput = document.getElementById('question-input');
const askBtn = document.getElementById('ask-btn');
const resetBtn = document.getElementById('reset-btn');
const loadingOverlay = document.getElementById('loading-overlay');
const loadingText = document.getElementById('loading-text');

let selectedFile = null;

// Event Listeners
browseBtn.addEventListener('click', () => pdfInput.click());
pdfInput.addEventListener('change', handleFileSelect);
uploadBtn.addEventListener('click', uploadPDF);
cancelBtn.addEventListener('click', cancelUpload);
askBtn.addEventListener('click', askQuestion);
resetBtn.addEventListener('click', resetSession);
questionInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        askQuestion();
    }
});

// Drag and Drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type === 'application/pdf') {
        pdfInput.files = files;
        handleFileSelect();
    } else {
        showStatus('Please drop a valid PDF file', 'error');
    }
});

uploadArea.addEventListener('click', (e) => {
    if (e.target === uploadArea || e.target.closest('.upload-icon') || e.target.closest('p')) {
        pdfInput.click();
    }
});

// Functions
function handleFileSelect() {
    const file = pdfInput.files[0];
    if (file) {
        if (file.type !== 'application/pdf') {
            showStatus('Please select a valid PDF file', 'error');
            return;
        }

        if (file.size > 16 * 1024 * 1024) {
            showStatus('File is too large. Maximum size is 16MB', 'error');
            return;
        }

        selectedFile = file;
        filenameSpan.textContent = file.name;
        fileInfo.style.display = 'block';
        uploadArea.style.display = 'none';
    }
}

function cancelUpload() {
    selectedFile = null;
    pdfInput.value = '';
    fileInfo.style.display = 'none';
    uploadArea.style.display = 'block';
    hideStatus();
}

async function uploadPDF() {
    if (!selectedFile) {
        showStatus('Please select a PDF file first', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('pdf_file', selectedFile);

    showLoading('Processing PDF...');
    uploadBtn.disabled = true;

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok && data.success) {
            showStatus(data.message, 'success');
            qaSection.style.display = 'block';
            fileInfo.style.display = 'none';
            chatHistory.innerHTML = '';
            questionInput.value = '';
            questionInput.focus();
        } else {
            showStatus(data.error || 'Failed to process PDF', 'error');
            uploadBtn.disabled = false;
        }
    } catch (error) {
        console.error('Error:', error);
        showStatus('An error occurred while uploading the PDF', 'error');
        uploadBtn.disabled = false;
    } finally {
        hideLoading();
    }
}

async function askQuestion() {
    const question = questionInput.value.trim();

    if (!question) {
        showStatus('Please enter a question', 'error');
        return;
    }

    if (question.length > 500) {
        showStatus('Question is too long. Maximum 500 characters.', 'error');
        return;
    }

    // Add question to chat
    addMessageToChat(question, 'question');
    questionInput.value = '';
    askBtn.disabled = true;

    showLoading('Generating answer...');

    try {
        const response = await fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question })
        });

        const data = await response.json();

        if (response.ok && data.success) {
            addMessageToChat(data.answer, 'answer');
            hideStatus();
        } else {
            showStatus(data.error || 'Failed to generate answer', 'error');
            // Remove the question from chat if answer failed
            const lastMessage = chatHistory.lastElementChild;
            if (lastMessage) {
                lastMessage.remove();
            }
        }
    } catch (error) {
        console.error('Error:', error);
        showStatus('An error occurred while generating the answer', 'error');
        // Remove the question from chat if answer failed
        const lastMessage = chatHistory.lastElementChild;
        if (lastMessage) {
            lastMessage.remove();
        }
    } finally {
        hideLoading();
        askBtn.disabled = false;
        questionInput.focus();
    }
}

function addMessageToChat(message, type) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'chat-message';

    const label = document.createElement('div');
    label.className = `message-label ${type}`;
    label.innerHTML = type === 'question' ? '❓ Your Question:' : '🤖 Answer:';

    const content = document.createElement('div');
    content.className = `message-content ${type}`;
    content.textContent = message;

    messageDiv.appendChild(label);
    messageDiv.appendChild(content);
    chatHistory.appendChild(messageDiv);

    // Scroll to bottom
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

async function resetSession() {
    if (!confirm('Are you sure you want to upload a new PDF? This will clear the current session.')) {
        return;
    }

    showLoading('Resetting session...');

    try {
        const response = await fetch('/reset', {
            method: 'POST'
        });

        const data = await response.json();

        if (response.ok && data.success) {
            // Reset UI
            selectedFile = null;
            pdfInput.value = '';
            fileInfo.style.display = 'none';
            uploadArea.style.display = 'block';
            qaSection.style.display = 'none';
            chatHistory.innerHTML = '';
            questionInput.value = '';
            uploadBtn.disabled = false;
            hideStatus();
            showStatus('Session reset. Please upload a new PDF.', 'success');
        } else {
            showStatus(data.error || 'Failed to reset session', 'error');
        }
    } catch (error) {
        console.error('Error:', error);
        showStatus('An error occurred while resetting the session', 'error');
    } finally {
        hideLoading();
    }
}

function showStatus(message, type) {
    statusMessage.textContent = message;
    statusMessage.className = `status-message ${type}`;
    statusSection.style.display = 'block';

    // Auto-hide success messages after 5 seconds
    if (type === 'success') {
        setTimeout(() => {
            hideStatus();
        }, 5000);
    }
}

function hideStatus() {
    statusSection.style.display = 'none';
}

function showLoading(text) {
    loadingText.textContent = text;
    loadingOverlay.style.display = 'flex';
}

function hideLoading() {
    loadingOverlay.style.display = 'none';
}

// Check server health on load
async function checkHealth() {
    try {
        const response = await fetch('/health');
        const data = await response.json();

        if (!data.models_loaded) {
            showStatus('Warning: AI models are still loading. Please wait...', 'error');
        }
    } catch (error) {
        console.error('Health check failed:', error);
    }
}

// Run health check on page load
checkHealth();
