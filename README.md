# AI-IDS (MVP)

This repository contains an AI-powered Intrusion Detection System (AI-IDS) MVP.

## Step 1 — Project setup (this step)

1. Create a Python virtual environment (PowerShell):

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. Or build & run with Docker:

   ```powershell
   docker build -t ai-ids:latest .
   docker run --rm ai-ids:latest
   ```

3. Repository management

   This project does not require or use Git by default. If you prefer to use a version control system locally, you may initialize a repository on your machine and manage commits there — it's optional and up to your workflow.

## Next steps
- Data acquisition & preprocessing (CIC-IDS2017 first).
- Model implementations and training pipelines.

