"""Python Client SDK for AI Gateway"""

import logging
from typing import Optional, List
import requests

from .models import (
    AgentType,
    SessionCreateRequest,
    SessionCreateResponse,
    ChatRequest,
    ChatResponse,
    SessionInfo,
    HealthResponse,
)

logger = logging.getLogger(__name__)


class GatewayClient:
    """Client for interacting with the AI Gateway API"""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 200):
        """
        Initialize the gateway client

        Args:
            base_url: Base URL of the gateway API
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def health_check(self) -> HealthResponse:
        """Check gateway health"""
        response = self.session.get(f"{self.base_url}/", timeout=self.timeout)
        response.raise_for_status()
        return HealthResponse(**response.json())

    def create_session(
        self,
        agent_type: AgentType,
        model: str,
        study_path : str,
        language: str = "en",
        user_id: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> SessionCreateResponse:
        """
        Create a new chat session

        Args:
            agent_type: Type of agent 
            model: Model name to use
            language: Chat language (default: en)
            user_id: Optional user identifier
            metadata: Optional metadata dictionary

        Returns:
            SessionCreateResponse with session details
        """
      
        request = SessionCreateRequest(
            agent_type=agent_type,
            model=model,
            language=language,
            user_id=user_id,
            study_path= study_path,
            metadata=metadata
        )
        response = self.session.post(
            f"{self.base_url}/sessions",
            json=request.model_dump(),
            timeout=self.timeout
        )
        response.raise_for_status()
        return SessionCreateResponse(**response.json())

    def chat(self, session_id: str, message: str, stream: bool = False, files: Optional[List[dict]] = None) -> ChatResponse:
        """
        Send a message in a session

        Args:
            session_id: Session identifier
            message: User message
            stream: Enable streaming (not yet implemented)
            files: Optional list of file attachments, each with:
                   - name: str (filename)
                   - content: str (base64 encoded or text content)
                   - mime_type: str (MIME type)
                   - size: int (size in bytes)

        Returns:
            ChatResponse with AI response
        """
        request_data = {
            "session_id": session_id,
            "message": message,
            "stream": stream
        }

        if files:
            request_data["files"] = files

        request = ChatRequest(**request_data)

        response = self.session.post(
            f"{self.base_url}/chat",
            json=request.model_dump(),
            timeout=self.timeout
        )
        response.raise_for_status()
        return ChatResponse(**response.json())

    def get_session(self, session_id: str) -> SessionInfo:
        """
        Get session information

        Args:
            session_id: Session identifier

        Returns:
            SessionInfo with session details
        """
        response = self.session.get(
            f"{self.base_url}/sessions/{session_id}",
            timeout=self.timeout
        )
        response.raise_for_status()
        return SessionInfo(**response.json())

    def list_sessions(self) -> List[SessionInfo]:
        """
        List all active sessions

        Returns:
            List of SessionInfo objects
        """
        response = self.session.get(
            f"{self.base_url}/sessions",
            timeout=self.timeout
        )
        response.raise_for_status()
        return [SessionInfo(**s) for s in response.json()]

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session

        Args:
            session_id: Session identifier

        Returns:
            True if deleted successfully
        """
        response = self.session.delete(
            f"{self.base_url}/sessions/{session_id}",
            timeout=self.timeout
        )
        response.raise_for_status()
        return True

    def close(self):
        """Close the client session"""
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

