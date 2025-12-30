# Deployment Checklist for 12 Systems

Use this checklist when deploying to each of your 12 systems.

---

## Pre-Deployment (Do Once)

- [ ] Test the system on current machine
- [ ] Verify all fixes are working:
  - [ ] Text-only mode gives complete answers
  - [ ] Images load properly (no 404 errors)
  - [ ] Modal close button works
  - [ ] Conversation history works
  - [ ] Timestamps show correctly
- [ ] Package the `vision_qa_deployment/` folder
- [ ] Optional: Create a ZIP/tar for easy transfer

```bash
cd /path/to/PDF-QA-System
tar -czf vision_qa_deployment.tar.gz vision_qa_deployment/
```

---

## For Each System (1-12)

### System Information
- **System Name/Number**: _____________
- **IP Address**: _____________
- **Username**: _____________
- **Deployment Date**: _____________

### Steps

#### 1. Prerequisites Check
- [ ] Python 3.9+ installed
  ```bash
  python3 --version
  # Should show 3.9 or higher
  ```

- [ ] Ollama installed
  ```bash
  ollama --version
  # If not: curl -fsSL https://ollama.ai/install.sh | sh
  ```

- [ ] Sufficient disk space (20GB minimum)
  ```bash
  df -h
  ```

#### 2. Transfer Files
- [ ] Copy deployment folder to system
  ```bash
  # Option 1: SCP
  scp -r vision_qa_deployment/ user@system-ip:/path/to/destination/

  # Option 2: Transfer tar.gz then extract
  scp vision_qa_deployment.tar.gz user@system-ip:/path/
  ssh user@system-ip "cd /path && tar -xzf vision_qa_deployment.tar.gz"
  ```

#### 3. Setup
- [ ] SSH into the system
  ```bash
  ssh user@system-ip
  ```

- [ ] Navigate to deployment folder
  ```bash
  cd /path/to/vision_qa_deployment
  ```

- [ ] Make scripts executable
  ```bash
  chmod +x start_vision_app.sh quick_install.sh install_deps.sh
  ```

#### 4. Installation
- [ ] Run startup script
  ```bash
  ./start_vision_app.sh
  ```

- [ ] Wait for model download (if needed)
  - First time: ~5-10 minutes (downloads 7GB model)
  - Subsequent runs: ~15 seconds

- [ ] Verify app starts successfully
  - Should see: "Running on http://127.0.0.1:5000"

#### 5. Testing
- [ ] Open browser to http://localhost:5000 (or system-ip:5000)
- [ ] Upload a small test PDF (1-5 pages)
- [ ] Ask a text-based question
  - Verify: "⚡ Text-only mode" shown
  - Verify: Response time < 20 seconds
  - Verify: Answer is complete and detailed
- [ ] Ask a question about a diagram
  - Verify: "👁️ Vision AI used" shown
  - Verify: Images display correctly
  - Verify: Can open and close image modal
- [ ] Test conversation history
  - Ask initial question
  - Follow up with "tell me more about that"
  - Verify: Context is understood
- [ ] Check timestamps
  - Verify: All messages show time in format HH:MM:SS

#### 6. Configuration (Optional)

- [ ] Change port if needed (default: 5000)
  ```bash
  nano app_vision.py
  # Change line 406: port=5000 to port=YOUR_PORT
  ```

- [ ] Enable external access (if needed)
  ```bash
  # Already set to host='0.0.0.0' by default
  # Access from network: http://system-ip:5000
  ```

- [ ] Configure auto-start on boot (optional)
  ```bash
  # Create systemd service
  sudo nano /etc/systemd/system/vision-qa.service
  ```

  Add:
  ```ini
  [Unit]
  Description=Vision PDF QA System
  After=network.target ollama.service

  [Service]
  Type=simple
  User=YOUR_USERNAME
  WorkingDirectory=/path/to/vision_qa_deployment
  ExecStart=/path/to/vision_qa_deployment/venv/bin/python app_vision.py
  Restart=on-failure

  [Install]
  WantedBy=multi-user.target
  ```

  Enable:
  ```bash
  sudo systemctl daemon-reload
  sudo systemctl enable vision-qa
  sudo systemctl start vision-qa
  ```

#### 7. Documentation
- [ ] Note system-specific details:
  - Port used: _____________
  - Installation path: _____________
  - Any custom configurations: _____________
  - Performance notes: _____________

#### 8. Verification
- [ ] App runs without errors
- [ ] All features working correctly
- [ ] Logs directory created and writable
- [ ] Performance acceptable for use case

---

## Troubleshooting Reference

### Common Issues & Solutions

| Issue | Quick Fix |
|-------|-----------|
| Port 5000 in use | Change port in app_vision.py line 406 |
| Ollama not found | `curl -fsSL https://ollama.ai/install.sh \| sh` |
| Dependencies fail | Run `./quick_install.sh` instead |
| Slow responses | Check CPU usage, reduce concurrent users |
| Out of memory | Reduce num_ctx in vision_qa_engine.py |
| Images 404 | Verify processed_pdfs/ directory exists |

---

## Post-Deployment

### System 1
- Status: ⬜ Not Started / 🔄 In Progress / ✅ Complete
- Date: _____________
- Notes: _____________

### System 2
- Status: ⬜ Not Started / 🔄 In Progress / ✅ Complete
- Date: _____________
- Notes: _____________

### System 3
- Status: ⬜ Not Started / 🔄 In Progress / ✅ Complete
- Date: _____________
- Notes: _____________

### System 4
- Status: ⬜ Not Started / 🔄 In Progress / ✅ Complete
- Date: _____________
- Notes: _____________

### System 5
- Status: ⬜ Not Started / 🔄 In Progress / ✅ Complete
- Date: _____________
- Notes: _____________

### System 6
- Status: ⬜ Not Started / 🔄 In Progress / ✅ Complete
- Date: _____________
- Notes: _____________

### System 7
- Status: ⬜ Not Started / 🔄 In Progress / ✅ Complete
- Date: _____________
- Notes: _____________

### System 8
- Status: ⬜ Not Started / 🔄 In Progress / ✅ Complete
- Date: _____________
- Notes: _____________

### System 9
- Status: ⬜ Not Started / 🔄 In Progress / ✅ Complete
- Date: _____________
- Notes: _____________

### System 10
- Status: ⬜ Not Started / 🔄 In Progress / ✅ Complete
- Date: _____________
- Notes: _____________

### System 11
- Status: ⬜ Not Started / 🔄 In Progress / ✅ Complete
- Date: _____________
- Notes: _____________

### System 12
- Status: ⬜ Not Started / 🔄 In Progress / ✅ Complete
- Date: _____________
- Notes: _____________

---

## Summary

- Total Systems: 12
- Completed: ___ / 12
- Failed: ___ / 12
- Average Install Time: _____ minutes
- Average Test Time: _____ minutes

---

**Ready for mass deployment!** 🚀
