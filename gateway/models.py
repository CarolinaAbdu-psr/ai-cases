"""Gateway data models for requests and responses"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class AgentType(str, Enum):
    """Available agent types"""
    FACTORY = "factory"
    PSRIO = "psrio"
    KNOWLEDGE_HUB = "knowledge_hub"
    CASE_INPUT = "case_input"


class ChatRole(str, Enum):
    """Chat message roles"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    """A single chat message"""
    role: ChatRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None


class SessionCreateRequest(BaseModel):
    """Request to create a new chat session"""
    agent_type: AgentType
    model: str
    language: str = "en"
    user_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SessionCreateResponse(BaseModel):
    """Response after creating a session"""
    session_id: str
    agent_type: AgentType
    model: str
    language: str
    rag_date: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.now)


class FileAttachment(BaseModel):
    """File attachment metadata"""
    name: str
    content: str  # Base64 encoded content or text content
    mime_type: str
    size: int  # Size in bytes


class ChatRequest(BaseModel):
    """Request to send a message in a session"""
    session_id: str
    message: str
    stream: bool = False
    files: Optional[List[FileAttachment]] = None


class ChatResponse(BaseModel):
    """Response from the chat"""
    session_id: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None
    files_processed: Optional[List[str]] = None  # List of processed file names


class SessionInfo(BaseModel):
    """Information about a session"""
    session_id: str
    agent_type: AgentType
    model: str
    language: str
    created_at: datetime
    last_activity: datetime
    message_count: int = 0
    rag_date: Optional[datetime] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    timestamp: datetime = Field(default_factory=datetime.now)
    agents: List[str]
    models: List[str]


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

