"""AI Gateway REST API using FastAPI"""

import logging
from datetime import datetime
from typing import List

from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .models import (
    SessionCreateRequest,
    SessionCreateResponse,
    ChatRequest,
    ChatResponse,
    SessionInfo,
    HealthResponse,
    ErrorResponse,
)
from .service import GatewayService

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Gateway API",
    description="Unified gateway for multiple AI chat frontends",
    version="1.0.0",
)

# Add CORS middleware to allow web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the gateway service
gateway_service = GatewayService()


@app.on_event("startup")
async def startup_event():
    """Pre-download RAGs on startup to avoid slow first requests"""
    logger.info("=" * 60)
    logger.info("Starting AI Gateway - Pre-downloading vectorstores...")
    logger.info("=" * 60)

    try:
        # Pre-download all agent RAGs
        await gateway_service.preload_all_rags()
        logger.info("✓ All vectorstores preloaded successfully")
    except Exception as e:
        logger.error(f"✗ Error preloading vectorstores: {str(e)}")
        logger.warning("Gateway will continue but first requests may be slow")

    logger.info("=" * 60)
    logger.info("AI Gateway ready to accept requests")
    logger.info("=" * 60)


@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        agents=gateway_service.get_available_agents(),
        models=gateway_service.get_available_models(),
    )


@app.post("/sessions", response_model=SessionCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_session(request: SessionCreateRequest):
    """Create a new chat session"""
    try:
        logger.info(f"Creating session for agent {request.agent_type} with model {request.model}")
        response = gateway_service.create_session(request)
        return response
    except Exception as e:
        logger.error(f"Error creating session: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create session: {str(e)}"
        )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message in a session"""
    try:
        response = gateway_service.chat(request)
        return response
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error processing chat: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process chat: {str(e)}"
        )


@app.get("/sessions", response_model=List[SessionInfo])
async def list_sessions():
    """List all active sessions"""
    try:
        return gateway_service.list_sessions()
    except Exception as e:
        logger.error(f"Error listing sessions: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list sessions: {str(e)}"
        )


@app.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    """Get session information"""
    session_info = gateway_service.get_session_info(session_id)
    if not session_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}"
        )
    return session_info


@app.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: str):
    """Delete a session"""
    if not gateway_service.delete_session(session_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}"
        )


@app.post("/admin/redownload-rags", status_code=status.HTTP_202_ACCEPTED)
async def redownload_rags(background_tasks: BackgroundTasks, agent_type: str = None):
    """
    Force redownload of RAG vectorstores from S3

    Parameters:
    - agent_type (optional): Specific agent to redownload ('factory', 'psrio', 'knowledge_hub')
                            If not provided, redownloads all agents

    Returns:
    - Message indicating the redownload has been queued
    """
    try:
        # Queue the redownload in background to avoid blocking
        background_tasks.add_task(gateway_service.force_redownload_rags, agent_type)

        message = f"Redownload queued for {'all agents' if not agent_type else agent_type}"
        logger.info(message)

        return {
            "status": "accepted",
            "message": message,
            "agent_type": agent_type or "all"
        }
    except Exception as e:
        logger.error(f"Error queuing RAG redownload: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to queue redownload: {str(e)}"
        )


@app.get("/admin/rag-status")
async def get_rag_status():
    """Get the current status of RAG vectorstores"""
    try:
        return gateway_service.get_rag_status()
    except Exception as e:
        logger.error(f"Error getting RAG status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get RAG status: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).model_dump()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

