@echo off
REM Start the AI Gateway server

echo Starting AI Gateway...

REM Start the gateway with uvicorn
python -m uvicorn gateway.api:app --host 0.0.0.0 --port 8000 --reload

