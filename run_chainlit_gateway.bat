@echo off
REM Start Chainlit with Gateway integration

echo Starting Chainlit with Gateway...

REM Start chainlit
chainlit run app_gateway.py --port 8001

