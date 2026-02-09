import logging
import os
import base64

import chainlit as cl
from dotenv import load_dotenv

import helperdash
import helper.models
from gateway.client import GatewayClient
from gateway.models import AgentType

load_dotenv()

logger: logging.Logger = logging.getLogger(__name__)

# Translation support
current_language = 'en'
helperdash.set_language(current_language)
get_text = helperdash.get_text

# Gateway client configuration
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:8000")
gateway_client = GatewayClient(base_url=GATEWAY_URL)


STUDY_PATH = r"C:\PSR\SDDP18.0\examples\operation\1_stage\Case01"


async def process_uploaded_files(files):
    """Process uploaded files and convert to gateway format"""
    file_attachments = []

    # Common text file extensions
    text_extensions = {
        'txt', 'py', 'js', 'ts', 'jsx', 'tsx', 'java', 'c', 'cpp', 'h', 'hpp',
        'cs', 'go', 'rs', 'rb', 'php', 'html', 'css', 'scss', 'sass', 'less',
        'json', 'xml', 'yaml', 'yml', 'md', 'rst', 'tex', 'sh', 'bash', 'zsh',
        'sql', 'r', 'matlab', 'm', 'ini', 'cfg', 'conf', 'toml', 'properties',
        'log', 'csv', 'tsv', 'dockerfile', 'makefile', 'cmake', 'gradle'
    }

    for file in files:
        try:
            # Read file content
            content = file.content if hasattr(file, 'content') else None

            if content is None:
                # Try to read the file from path
                with open(file.path, 'rb') as f:
                    content = f.read()

            # Determine if this is likely a text file
            file_ext = file.name.split('.')[-1].lower() if '.' in file.name else ''
            is_text_file = (
                (file.mime and file.mime.startswith('text/')) or
                (file.mime and file.mime in ['application/json', 'application/xml', 'application/javascript', 'application/x-yaml']) or
                file_ext in text_extensions
            )

            # Process the content
            if isinstance(content, bytes):
                if is_text_file:
                    try:
                        # Decode as UTF-8 text
                        file_content = content.decode('utf-8')
                        logger.info(f"Decoded {file.name} as UTF-8 text")
                    except UnicodeDecodeError:
                        try:
                            # Try other common encodings
                            file_content = content.decode('latin-1')
                            logger.info(f"Decoded {file.name} as latin-1 text")
                        except:
                            # If all else fails, base64 encode
                            file_content = base64.b64encode(content).decode('utf-8')
                            logger.warning(f"Base64 encoded {file.name} (decode failed)")
                else:
                    # Binary file - base64 encode
                    file_content = base64.b64encode(content).decode('utf-8')
                    logger.info(f"Base64 encoded binary file: {file.name}")
            else:
                # Already a string
                file_content = content
                logger.info(f"File {file.name} already in string format")

            file_attachments.append({
                "name": file.name,
                "content": file_content,
                "mime_type": file.mime or "application/octet-stream",
                "size": len(content) if isinstance(content, bytes) else len(content.encode())
            })

            logger.info(f"✓ Processed file: {file.name} ({file.mime or 'unknown'}), size: {len(content) if isinstance(content, bytes) else len(content.encode())} bytes")

        except Exception as e:
            logger.error(f"✗ Error processing file {file.name}: {str(e)}", exc_info=True)
            await cl.Message(content=f"⚠️ Error processing file {file.name}: {str(e)}").send()

    return file_attachments


@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="Case Input",
            markdown_description="Helper specialized in input data",
            starters=[
                cl.Starter(
                    label="Get me SDDP Requirements",
                    message="What are the system requirements for installing SDDP?",
                    icon= "/public/starter/edit.svg",
                   
                ),
                cl.Starter(
                    label="Explain OptGen Strategies",
                    message="Explain how the optimization strategies in OptGen work.",
                    icon ="/public/starter/graduation-cap.svg" ,
                ),
                cl.Starter(
                    label="Battery Modeling",
                    message="What is the methodology for modeling a battery?",
                    icon = "/public/starter/idea.svg",
                ),
            ]
        ),
    ]


@cl.on_chat_start
async def start():
    """Initialize the chat session"""
    # Check API keys
    if not os.getenv("OPENAI_API_KEY"):
        await cl.Message(content=get_text("api_key_warning")).send()

    # Get chat profile and map to agent type
    chat_profile = cl.user_session.get("chat_profile")

    if chat_profile == "Case Input":
        agent_type = AgentType.CASE_INPUT
        agent_name = "Case Input"
    else:
        agent_type = AgentType.CASE_INPUT
        agent_name = "Case Input"

    # Get default model
    default_model = helper.models.DEFAULT_MODEL.name

    # Store in user session temporarily
    cl.user_session.set("agent_type", agent_type)
    cl.user_session.set("agent_name", agent_name)
    cl.user_session.set("model", default_model)
    cl.user_session.set("chat_language", current_language)
    cl.user_session.set("study_path", STUDY_PATH)

    # Create session via gateway
    try:
        loading_msg = cl.Message(content="⏳ Initializing session...")
        await loading_msg.send()

        # Build kwargs for create_session; include study_path if provided.
        create_kwargs = {
            "agent_type": agent_type,
            "model": default_model,
            "language": current_language,
            "study_path": STUDY_PATH,
            "user_id": cl.user_session.get("id"),
        }

        session_response = gateway_client.create_session(**create_kwargs)

        # Store session info
        cl.user_session.set("gateway_session_id", session_response.session_id)
        cl.user_session.set("rag_date", session_response.rag_date)

        await loading_msg.remove()

        logger.info(f"Session created: {session_response.session_id} for agent {agent_name}")

    except Exception as e:
        logger.error(f"Error creating gateway session: {str(e)}")
        await cl.Message(content=f"❌ Error initializing: {str(e)}").send()
        return

    # Configure settings
    available_models = helper.models.get_available_models()

    settings = await cl.ChatSettings(
        [
            cl.input_widget.Select(
                id="model_selector",
                label=get_text("model_selector") if hasattr(helperdash, 'get_text') else "Model",
                values=available_models,
                initial_value=default_model,
                description="Select the AI model to use"
            ),
        ]
    ).send()


@cl.on_settings_update
async def setup_agent(settings):
    """Handle settings updates"""

    agent_type = cl.user_session.get("agent_type")
    agent_name = cl.user_session.get("agent_name")
    model = settings["model_selector"]
    current_model = cl.user_session.get("model")

    # Check if model changed
    if model and model != current_model:

        # Show loading message
        loading_msg = cl.Message(content="⏳ Updating configuration...")
        await loading_msg.send()

        try:
            # Delete old session
            old_session_id = cl.user_session.get("gateway_session_id")
            if old_session_id:
                try:
                    gateway_client.delete_session(old_session_id)
                    logger.info(f"Deleted old session: {old_session_id}")
                except Exception as e:
                    logger.warning(f"Failed to delete old session: {e}")

            # Create new session with updated model
            session_response = gateway_client.create_session(
                agent_type=agent_type,
                model=model,
                language=cl.user_session.get("chat_language"),
                user_id=cl.user_session.get("id")
            )

            # Update session info
            cl.user_session.set("gateway_session_id", session_response.session_id)
            cl.user_session.set("model", model)
            cl.user_session.set("rag_date", session_response.rag_date)

            # Success feedback
            await loading_msg.remove()

            model_pretty_name = helper.models.get_model_pretty_name(model)
            success_msg = f"✅ Configuration updated!\n\n"
            success_msg += f"**Agent:** {agent_name}\n"
            success_msg += f"**Model:** {model_pretty_name}\n"
            if session_response.rag_date:
                success_msg += f"**Knowledge Base:** {session_response.rag_date.strftime('%Y-%m-%d')}"

            await cl.Message(content=success_msg).send()

            logger.info(f"Switched to model {model} for session {session_response.session_id}")

        except Exception as e:
            logger.error(f"Error updating configuration: {str(e)}")
            await loading_msg.remove()
            await cl.Message(content=f"❌ Error switching configuration: {str(e)}").send()


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages"""
    gateway_session_id = cl.user_session.get("gateway_session_id")

    if not gateway_session_id:
        await cl.Message(content="❌ Session not initialized. Please refresh the page.").send()
        return

    # Process uploaded files if any
    file_attachments = []
    if message.elements:
        files = [element for element in message.elements if isinstance(element, cl.File)]
        if files:
            # Show file processing step
            file_step = cl.Step(name="Processing Files", type="tool")
            async with file_step:
                file_step.output = f"Processing {len(files)} file(s)..."
                file_attachments = await process_uploaded_files(files)
                file_step.output = f"✓ Processed {len(files)} file(s): {', '.join([f.name for f in files])}"

    # Create a message with thinking indicator
    msg = cl.Message(content="")

    # Show a step for the AI processing
    async with cl.Step(name="Thinking", type="llm") as step:
        step.output = "🤔 Analyzing your message..."

        # Send the message first so it appears immediately
        await msg.send()

        try:
            # Update step to show we're contacting the AI
            step.output = "💭 Generating response..."

            # Send message to gateway with file attachments
            # Build kwargs for chat call; include study_path if provided.
            chat_kwargs = {
                "session_id": gateway_session_id,
                "message": message.content,
                "files": file_attachments if file_attachments else None,
            }
            try:
                response = gateway_client.chat(**chat_kwargs)
            except TypeError:
                # Fallback if the client doesn't accept study_path
                
                response = gateway_client.chat(**chat_kwargs)

            # Update step to show completion
            step.output = "✓ Response received"

            # Update the message with the response
            response_content = response.message

            # Add file processing info if files were uploaded
            if response.files_processed:
                response_content = f"📎 **Processed files:** {', '.join(response.files_processed)}\n\n{response_content}"

            msg.content = response_content
            await msg.update()

            logger.info(f"Processed message for session {gateway_session_id} with {len(file_attachments)} file(s)")

        except Exception as e:
            error_msg = f"❌ Error generating answer: {str(e)}"
            logger.error(error_msg, exc_info=True)
            step.output = "✗ Error occurred"
            msg.content = error_msg
            await msg.update()


@cl.on_chat_end
async def end():
    """Clean up when chat ends"""
    gateway_session_id = cl.user_session.get("gateway_session_id")
    if gateway_session_id:
        try:
            gateway_client.delete_session(gateway_session_id)
            logger.info(f"Deleted session on chat end: {gateway_session_id}")
        except Exception as e:
            logger.warning(f"Failed to delete session on chat end: {e}")


@cl.on_chat_resume
async def on_chat_resume(thread):
    """Handle chat resume - create new session"""
    # For now, we'll start a new session when resuming
    # In the future, you could implement session persistence
    logger.info("Chat resumed - starting new session")
    await start()

