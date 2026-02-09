"""AI Gateway Service - Core business logic"""

import datetime as dt
import logging
import os
import uuid
from typing import Dict, Tuple, Optional, Any

import helper.models
import helper.rag_input_cases

from .models import (
    AgentType,
    SessionCreateRequest,
    SessionCreateResponse,
    ChatRequest,
    ChatResponse,
    SessionInfo,
)

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages chat sessions and their state"""

    def __init__(self):
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._rag_cache: Dict[str, dt.datetime] = {}
        self._rag_versions: Dict[str, str] = {}  # Track downloaded RAG ZIP filenames

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a session by ID"""
        session = self._sessions.get(session_id)
        if session:
            session["last_activity"] = dt.datetime.now()
        return session

    def create_session(self, request: SessionCreateRequest) -> SessionCreateResponse:
        """Create a new chat session"""
        session_id = str(uuid.uuid4())

        # Initialize RAG if needed
        rag_date = self._ensure_rag_initialized(request.agent_type.value)

        # Initialize chain
        rag = self._get_rag_module(request.agent_type.value)
        chain, memory = rag.initialize(
            model=request.model,
            chat_language=request.language,
            study_path = request.study_path,
            agent_type=request.agent_type.value
        )

        # Store session
        self._sessions[session_id] = {
            "session_id": session_id,
            "agent_type": request.agent_type,
            "model": request.model,
            "language": request.language,
            "chain": chain,
            "memory": memory,
            "thread_id": f"thread_{session_id}",
            "created_at": dt.datetime.now(),
            "last_activity": dt.datetime.now(),
            "message_count": 0,
            "rag_date": rag_date,
            "user_id": request.user_id,
            "metadata": request.metadata or {}
        }

        logger.info(f"Created session {session_id} for agent {request.agent_type} with model {request.model}")

        return SessionCreateResponse(
            session_id=session_id,
            agent_type=request.agent_type,
            model=request.model,
            language=request.language,
            rag_date=rag_date,
        )

    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Deleted session {session_id}")
            return True
        return False

    def get_session_info(self, session_id: str) -> Optional[SessionInfo]:
        """Get session information"""
        session = self.get_session(session_id)
        if not session:
            return None

        return SessionInfo(
            session_id=session["session_id"],
            agent_type=session["agent_type"],
            model=session["model"],
            language=session["language"],
            created_at=session["created_at"],
            last_activity=session["last_activity"],
            message_count=session["message_count"],
            rag_date=session.get("rag_date")
        )

    def list_sessions(self) -> list[SessionInfo]:
        """List all active sessions"""
        return [self.get_session_info(sid) for sid in self._sessions.keys()]

    def _get_rag_module(self, agent_type: str):
        """Get the appropriate RAG module for an agent type"""
        if agent_type == "case_input":
            return helper.rag_input_cases
        

    def _ensure_rag_initialized(self, agent_type: str, force: bool = False) -> dt.datetime:
        """Ensure RAG is initialized and downloaded for the agent type

        Args:
            agent_type: The agent type
            force: If True, forces redownload even if cached
        """
        cache_key = agent_type

        # Check if we've already initialized this agent's RAG recently (unless forcing)
        if not force and cache_key in self._rag_cache:
            cached_date = self._rag_cache[cache_key]
            # If cached within last hour, use it
            if (dt.datetime.now() - cached_date).seconds < 3600:
                logger.info(f"Using cached RAG for {agent_type}")
                return cached_date

        # Initialize RAG
        try:
            if agent_type == "case_input":
                rag_date = self._download_rag(agent_type, force)
            else:  # knowledge_hub
                rag_date = self._download_rag(agent_type, force)

            self._rag_cache[cache_key] = rag_date
            return rag_date

        except Exception as e:
            logger.error(f"Error initializing RAG for {agent_type}: {str(e)}")
            return dt.datetime.now()

    def _check_latest_rag(self, source_type: str) -> dt.datetime:
        """Check latest RAG date"""
        rag = self._get_rag_module(source_type.split('_')[0])
        return rag.get_latest_rag_date(source_type)

    def _download_rag(self, agent_type: str, force: bool = False) -> dt.datetime:
        """Download RAG for an agent type

        Args:
            agent_type: The agent type (factory, knowledge_hub, psrio, etc.)
            force: If True, forces redownload even if current version is up to date

        Returns:
            datetime: The date of the RAG version
        """
        rag = self._get_rag_module(agent_type.split('_')[0])
        persist_directory = rag.get_vectorstore_directory(agent_type)
        rag_date_file = os.path.join(persist_directory, "rag_date.txt")
        rag_version_file = os.path.join(persist_directory, "rag_version.txt")

        # Map agent types to source types for RAG naming
        source_type_mapping = {
            "case_input": "case_input"
        }
        source_type = source_type_mapping.get(agent_type, agent_type)

        # Check existing RAG version (ZIP filename)
        existing_version = None
        if os.path.exists(rag_version_file) and not force:
            try:
                existing_version = open(rag_version_file, "r").read().strip()
                logger.info(f"Found existing RAG version for {agent_type}: {existing_version}")
            except Exception as e:
                logger.warning(f"Could not read version file: {e}")

        try:
            # Get the latest RAG date and construct expected filename
            latest_rag_date = self._check_latest_rag(source_type)
            expected_rag_filename = f"rag_{source_type}_{latest_rag_date.strftime('%Y-%m-%d_%H-%M-%S')}.zip"

            # Check if we already have this version
            if not force and existing_version == expected_rag_filename:
                logger.info(f"✓ RAG {agent_type} is up to date (version: {existing_version})")

                # Read date from file
                if os.path.exists(rag_date_file):
                    try:
                        rag_date = dt.datetime.strptime(open(rag_date_file, "r").read().strip(), "%Y-%m-%d")
                        return rag_date
                    except:
                        pass
                return latest_rag_date

            # Download new version
            if existing_version:
                logger.info(f"📥 Downloading new RAG version for {agent_type}")
                logger.info(f"   Old: {existing_version}")
                logger.info(f"   New: {expected_rag_filename}")
            else:
                logger.info(f"📥 Downloading RAG for {agent_type} (first time)")

            rag_date = rag.download_latest_rag(persist_directory, source_type)

            # Save the version (ZIP filename) and date
            with open(rag_version_file, "w") as f:
                f.write(expected_rag_filename)
            with open(rag_date_file, "w") as f:
                f.write(rag_date.strftime("%Y-%m-%d"))

            # Update cache
            self._rag_versions[agent_type] = expected_rag_filename

            logger.info(f"✓ Downloaded RAG {agent_type} successfully (version: {expected_rag_filename})")
            return rag_date

        except Exception as e:
            logger.error(f"✗ Failed to download RAG for {agent_type}: {str(e)}")

            # Try to use existing vectorstore
            if os.path.exists(persist_directory):
                if os.path.exists(rag_date_file):
                    try:
                        rag_date = dt.datetime.strptime(open(rag_date_file, "r").read().strip(), "%Y-%m-%d")
                        logger.warning(f"Using existing RAG for {agent_type}")
                        return rag_date
                    except:
                        pass

                # No date file, use current date
                logger.warning(f"Using existing RAG for {agent_type} (no version info)")
                rag_date = dt.datetime.now()
                with open(rag_date_file, "w") as f:
                    f.write(rag_date.strftime("%Y-%m-%d"))
                return rag_date
            else:
                raise ValueError(f"No vectorstore available for {agent_type} and download failed")


class GatewayService:
    """Main gateway service for handling chat interactions"""

    def __init__(self):
        self.session_manager = SessionManager()
        logger.info("Gateway service initialized")

    def create_session(self, request: SessionCreateRequest) -> SessionCreateResponse:
        """Create a new chat session"""
        return self.session_manager.create_session(request)

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Process a chat message"""
        session = self.session_manager.get_session(request.session_id)
        if not session:
            raise ValueError(f"Session not found: {request.session_id}")

        chain = session["chain"]
        thread_id = session["thread_id"]
        agent_type = session["agent_type"].value
        language = session["language"]
        study_path = session["study_path"]

        try:
            config = {"configurable": {"thread_id": thread_id}}

            # Process file attachments if present
            file_context = ""
            files_processed = []
            if request.files:
                file_context = "\n\n**📎 Attached Files:**\n"
                for file_attachment in request.files:
                    files_processed.append(file_attachment.name)
                    file_context += f"\n### 📄 File: `{file_attachment.name}`\n"

                    # Determine file type and language for syntax highlighting
                    file_ext = file_attachment.name.split('.')[-1].lower() if '.' in file_attachment.name else ''

                    # Check if this is a text-based file
                    text_mime_types = [
                        'text/', 'application/json', 'application/xml',
                        'application/x-yaml', 'application/javascript',
                        'application/x-python', 'application/x-sh'
                    ]
                    text_extensions = [
                        'txt', 'py', 'js', 'ts', 'java', 'c', 'cpp', 'h', 'hpp',
                        'cs', 'go', 'rs', 'rb', 'php', 'html', 'css', 'scss',
                        'json', 'xml', 'yaml', 'yml', 'md', 'rst', 'sh', 'bash',
                        'sql', 'r', 'matlab', 'm', 'ini', 'cfg', 'conf', 'toml'
                    ]

                    is_text_file = (
                        any(file_attachment.mime_type.startswith(mt) for mt in text_mime_types) or
                        file_ext in text_extensions
                    )

                    if is_text_file:
                        try:
                            import base64
                            # Try to decode from base64 first
                            try:
                                content = base64.b64decode(file_attachment.content).decode('utf-8')
                            except:
                                # If decode fails, assume it's already plain text
                                content = file_attachment.content

                            # Truncate very large files
                            max_chars = 8000
                            if len(content) > max_chars:
                                content = content[:max_chars] + f"\n\n... (truncated, showing first {max_chars} characters of {len(content)})"

                            # Determine code language for syntax highlighting
                            lang_map = {
                                'py': 'python', 'js': 'javascript', 'ts': 'typescript',
                                'java': 'java', 'c': 'c', 'cpp': 'cpp', 'cs': 'csharp',
                                'go': 'go', 'rs': 'rust', 'rb': 'ruby', 'php': 'php',
                                'html': 'html', 'css': 'css', 'json': 'json',
                                'xml': 'xml', 'yaml': 'yaml', 'yml': 'yaml',
                                'sh': 'bash', 'bash': 'bash', 'sql': 'sql',
                                'md': 'markdown', 'r': 'r', 'm': 'matlab'
                            }
                            code_lang = lang_map.get(file_ext, '')

                            file_context += f"\n**File Content:**\n```{code_lang}\n{content}\n```\n"
                            logger.info(f"Processed text file: {file_attachment.name} ({len(content)} chars)")

                        except Exception as e:
                            logger.error(f"Error decoding file {file_attachment.name}: {str(e)}")
                            file_context += f"\n**Note:** Could not decode file content\n"
                    else:
                        # Binary file - provide info but don't include content
                        file_context += f"**Type:** {file_attachment.mime_type}\n"
                        file_context += f"**Size:** {file_attachment.size:,} bytes\n"
                        file_context += f"\n**Note:** This is a binary file. I can see the filename and metadata, but cannot directly read binary content. If you need me to analyze this file, please:\n"
                        file_context += f"- Convert it to text format if possible\n"
                        file_context += f"- Extract relevant text content\n"
                        file_context += f"- Describe what you'd like me to help with\n"
                        logger.info(f"Skipped binary file: {file_attachment.name}")

                logger.info(f"Processing {len(request.files)} file(s) for session {request.session_id}")

            # Combine message with file context
            full_message = request.message
            if file_context:
                full_message = f"{request.message}\n{file_context}"

            result = chain.invoke({
                "input": full_message,
                "chat_language": language,
                "agent_type": agent_type,
                "messages": []
            }, config=config)

            if "messages" in result and result["messages"]:
                response_text = result["messages"][-1].content
            else:
                response_text = "No response generated"

            # Update session stats
            session["message_count"] += 1
            session["last_activity"] = dt.datetime.now()

            return ChatResponse(
                session_id=request.session_id,
                message=response_text,
                metadata={
                    "agent_type": agent_type,
                    "model": session["model"],
                },
                files_processed=files_processed if files_processed else None
            )

        except Exception as e:
            logger.error(f"Error processing chat for session {request.session_id}: {str(e)}", exc_info=True)
            raise

    def get_session_info(self, session_id: str) -> Optional[SessionInfo]:
        """Get session information"""
        return self.session_manager.get_session_info(session_id)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        return self.session_manager.delete_session(session_id)

    def list_sessions(self) -> list[SessionInfo]:
        """List all active sessions"""
        return self.session_manager.list_sessions()

    def get_available_agents(self) -> list[str]:
        """Get available agent types"""
        return [agent.value for agent in list(AgentType)]

    def get_available_models(self) -> list[str]:
        """Get available models"""
        return helper.models.get_available_models()

    async def preload_all_rags(self):
        """Preload all RAG vectorstores on startup"""
        logger.info("Preloading RAG vectorstores for all agents...")

        agents_to_preload = [
            ("case_input", ["case_input"])
        ]

        for agent_name, agent_types in agents_to_preload:
            logger.info(f"→ Preloading {agent_name} RAGs...")
            try:
                for agent_type in agent_types:
                    self.session_manager._download_rag(agent_type, force=False)
                logger.info(f"✓ {agent_name} RAGs loaded")
            except Exception as e:
                logger.error(f"✗ Error preloading {agent_name}: {str(e)}")

    def force_redownload_rags(self, agent_type: str = None):
        """Force redownload of RAG vectorstores

        Args:
            agent_type: Specific agent to redownload, or None for all agents
        """
        if agent_type:
            # Redownload specific agent
            logger.info(f"Force redownloading RAGs for {agent_type}...")
            try:
                self.session_manager._ensure_rag_initialized(agent_type, force=True)
                logger.info(f"✓ Redownloaded {agent_type} successfully")
            except Exception as e:
                logger.error(f"✗ Error redownloading {agent_type}: {str(e)}")
                raise
        else:
            # Redownload all agents
            logger.info("Force redownloading RAGs for all agents...")
            agents = ["case_input"]

            for agent in agents:
                try:
                    logger.info(f"→ Redownloading {agent}...")
                    self.session_manager._ensure_rag_initialized(agent, force=True)
                    logger.info(f"✓ {agent} redownloaded successfully")
                except Exception as e:
                    logger.error(f"✗ Error redownloading {agent}: {str(e)}")

    def get_rag_status(self) -> dict:
        """Get the current status of all RAG vectorstores"""
        status = {}

        agents_to_check = {
            "case_input": ["case_input"]
        }

        for agent_name, agent_types in agents_to_check.items():
            agent_status = {}

            for agent_type in agent_types:
                try:
                    rag = self.session_manager._get_rag_module(agent_type.split('_')[0])
                    persist_directory = rag.get_vectorstore_directory(agent_type)
                    rag_version_file = os.path.join(persist_directory, "rag_version.txt")
                    rag_date_file = os.path.join(persist_directory, "rag_date.txt")

                    exists = os.path.exists(persist_directory)
                    version = None
                    date = None

                    if os.path.exists(rag_version_file):
                        try:
                            version = open(rag_version_file, "r").read().strip()
                        except:
                            pass

                    if os.path.exists(rag_date_file):
                        try:
                            date = open(rag_date_file, "r").read().strip()
                        except:
                            pass

                    agent_status[agent_type] = {
                        "exists": exists,
                        "version": version,
                        "date": date,
                        "path": persist_directory
                    }
                except Exception as e:
                    agent_status[agent_type] = {
                        "exists": False,
                        "error": str(e)
                    }

            status[agent_name] = agent_status

        return status


