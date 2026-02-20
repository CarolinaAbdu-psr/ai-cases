import yaml
import os
import re
import pandas as pd
from dotenv import load_dotenv
from typing import Tuple, List, Annotated, Dict, Any
from pathlib import Path 

import logging

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_openai.chat_models.base import BaseChatOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

from helper.compare_cases.compare import compare_cases


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# Load environment variables from .env file
load_dotenv()

REQUEST_TIMEOUT = 120
MAX_TOKENS = 4096

# Agent template settings
AGENTS_DIR = "agents"
COMPARE_AGENT_FILENAME = "case_compare_agent.yaml"

STUDY_PATH_A = r"C:\PSR\SDDP18.1Beta\examples\operation\1_stage\Case01"
STUDY_PATH_B = r"C:\PSR\SDDP18.1Beta\examples\operation\1_stage\Case01-edited"

# -----------------------------
# Load agent configuration
# -----------------------------

def load_compare_agent_config(filepath: str) -> bool:
    """Load prompts and templates from the agent YAML configuration."""
    global SYSTEM_PROMPT_TEMPLATE, USER_PROMPT_RELEVANT_FILENAMES, USER_PROMPT_TEXTUAL_RESPONSE
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)

        SYSTEM_PROMPT_TEMPLATE = config['system_prompt_template']
        USER_PROMPT_RELEVANT_FILENAMES = config['translation_task']['user_prompt_relevant_files']
        USER_PROMPT_TEXTUAL_RESPONSE = config['formatting_task']['user_prompt_textual_response']

        logger.info(f"Compare agent template loaded successfully from {filepath}")
        return True
    except (FileNotFoundError, yaml.YAMLError, KeyError) as e:
        logger.error(f"Error loading Cypher agent from {filepath}: {e}")
        return False


def load_agents_config() -> Dict[str, Any]:
    """Load agent configuration file."""
    filepath = os.path.join(AGENTS_DIR, COMPARE_AGENT_FILENAME)
    if load_compare_agent_config(filepath):
        return {'status': 'loaded'}
    return {'status': 'failed'}


# Load configuration once
_AGENTS_CONFIG = load_agents_config()

# -----------------------------
# Creata GraphState
# -----------------------------

class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    input: str
    descriptions : str 
    files_list: List[str]
    relevant_files: str
    tables :str
    chat_language: str
    

# -----------------------------
# Get filenames  
# -----------------------------

def create_files(study_path_a: str, study_path_b:str):
    "Create a schema and save it at the study folder"
    try: 
        logger.info("Comparing cases")
        compare_cases(study_path_a,study_path_b)
    except Exception as e: 
        logger.error(f"Error comparing cases:{e}")
        

def list_files(state:GraphState) -> str:
    """List all files in the specified output directory."""
    BASE_DIR = Path(__file__).resolve().parent
    output_dir = BASE_DIR / "comparison_results" 
    logger.info("Compainson Output dir")
    files = []
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            files.append(item_path)

    logger.info("FILES AVAILABLE", files)
    return {"files_list" : files}


# -----------------------------
# Relevant Filenames Generation 
# -----------------------------

def parse_relevant_filenames(response):
    """
    Parse a model response to extract a list of relevant filenames.
    It processes the response line by line, cleaning Markdown formatting 
    and ignoring separators or empty lines.
    """
    filenames = []
    
    content = getattr(response, 'content', str(response))

    for line in content.split('\n'):
        line = line.strip()

        # Skip empty lines or explicit separator lines (keeping the '//' logic to ignore comments)
        if not line or line.startswith('//'):
            continue

        clean_name = re.sub(r'^[\-\*]?\s*(\d+\.)?\s*', '', line)

        # Remove potential quotes around filenames (e.g., "case1.dat")
        if clean_name.lower().endswith('.csv'):
            filenames.append(clean_name)
        
        logger.info("Filenames available",filenames)

    return filenames


def generate_relevant_filenames(state: GraphState, llm: BaseChatOpenAI) -> Dict[str, Any]:
    """Generate relevant filenames from user input using the configured prompts and LLM."""

    logger.info("Step: Relevant Filenames Generation")
    system_prompt_content = SYSTEM_PROMPT_TEMPLATE.format(
        files_descriptions=""
    )

    system_message = SystemMessage(content=system_prompt_content)
    user_prompt_content = USER_PROMPT_RELEVANT_FILENAMES.format(
        user_input=state["input"],
        files_list=state["files_list"]
    )

    human_message = HumanMessage(content=user_prompt_content)
    response = llm.invoke([system_message, human_message])

    #Format multiple queries
    filenames = parse_relevant_filenames(response)
    logger.info(f"Found {len(filenames)} relavant files: {filenames}")

    return {"relevant_files": filenames}


# -----------------------------
# Generate textual final response
# -----------------------------


def load_csvs_as_context(state:GraphState, max_rows=20):
    """
    Reads multiple CSV files and converts them into a formatted Markdown string
    to be used as context for an LLM.
    
    Args:
        file_paths (list): List of paths to CSV files.
        max_rows (int): Limit rows to avoid exceeding token limits.
        
    Returns:
        str: A single string containing formatted tables for all files.
    """
    context_buffer = []
    file_paths = state["relevant_files"]
    for file_path in file_paths:
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                context_buffer.append(f"Error: File not found: {file_path}")
                continue

            # Read CSV
            df = pd.read_csv(file_path, sep=',') 
            
            # Create a clear header for the LLM
            file_name = os.path.basename(file_path)
            section = f"### Data from file: {file_name}\n"
            
            # Convert DataFrame to Markdown table
            # If the file is huge, we only take the first 'max_rows' or sample it
            if len(df) > max_rows:
                section += f"(Showing first {max_rows} rows of {len(df)} total)\n"
                table_md = df.head(max_rows).to_markdown(index=False)
            else:
                table_md = df.to_markdown(index=False)
                
            section += table_md
            context_buffer.append(section)

        except Exception as e:
            context_buffer.append(f"Error reading {file_path}: {str(e)}")

    # Join all file contents with double newlines
    print( "\n\n".join(context_buffer))
    return {"tables": "\n\n".join(context_buffer)}



def generate_textual_response(state: GraphState, llm: BaseChatOpenAI) -> Dict[str, Any]:
    """Generate a user-facing answer using the executed query result."""
    logger.info("Step: Generate textual response")
    system_message = SystemMessage(content=SYSTEM_PROMPT_TEMPLATE.format(files_descriptions=""))
    user_prompt_content = USER_PROMPT_TEXTUAL_RESPONSE.format(
        user_input=state["input"],
        relevant_files=state["relevant_files"],
        tables=state["tables"]
    )
    human_message = HumanMessage(content=user_prompt_content)
    response = llm.invoke([system_message, human_message])
    return {"messages": [AIMessage(content=response.content)]}


def initialize(model: str, chat_language: str, study_path, agent_type: str = "factory") -> Tuple[StateGraph, MemorySaver]:
    """Initialize the LLM and return the compiled LangGraph workflow and memory."""

    try:
        if model == "gpt-5-2025-08-07":
            llm = ChatOpenAI(model_name="gpt-5-2025-08-07", request_timeout=REQUEST_TIMEOUT)
        elif model == "gpt-4.1":
            llm = ChatOpenAI(model_name="gpt-4.1", temperature=0.7, max_tokens=MAX_TOKENS, request_timeout=REQUEST_TIMEOUT)
        elif model == "gpt-4.1-mini":
            llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0.7, max_tokens=MAX_TOKENS, request_timeout=REQUEST_TIMEOUT)
        elif model == "o3":
            llm = ChatOpenAI(model_name="o3", request_timeout=REQUEST_TIMEOUT)
        elif model == "claude-4-sonnet":
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(model='claude-sonnet-4-20250514', anthropic_api_key=os.getenv('ANTHROPIC_API_KEY'), temperature=0.7, max_tokens=MAX_TOKENS, timeout=REQUEST_TIMEOUT)
        elif model == "deepseek-reasoner":
            llm = BaseChatOpenAI(model='deepseek-reasoner', openai_api_key=os.getenv('DEEPSEEK_API_KEY'), openai_api_base='https://api.deepseek.com', temperature=0.7, max_tokens=MAX_TOKENS, request_timeout=REQUEST_TIMEOUT)
        elif model == "local_land":
            llm = ChatOpenAI(model_name="qwen3:14b", temperature=0.7, max_tokens=MAX_TOKENS, request_timeout=REQUEST_TIMEOUT, base_url= "http://10.246.47.184:10000/v1")
        else:
            llm = ChatOpenAI(model_name="gpt-4.1", temperature=0.7, max_tokens=MAX_TOKENS, request_timeout=REQUEST_TIMEOUT)

        create_files(STUDY_PATH_A,STUDY_PATH_B)

        app, memory = create_langgraph_workflow(llm)
        
        return app, memory

    except Exception as e:
        logger.error(f"Error initializing RAG: {str(e)}")
        

def create_langgraph_workflow(llm: BaseChatOpenAI):

    """Create LangGraph workflow that runs RAG -> get relevant files -> load content -> format answer."""
    def list_files_partial(state:GraphState):   
        return list_files(state)
    
    def generate_filenames_partial(state: GraphState):
        return generate_relevant_filenames(state, llm)
    
    def load_csvs_as_context_partial(state: GraphState):
        return load_csvs_as_context(state)
    
    def generate_textual_response_partial(state: GraphState):
        return generate_textual_response(state, llm)

    workflow = StateGraph(GraphState)

    workflow.add_node("compare_files", list_files_partial)
    workflow.add_node("get_relevant_files", generate_filenames_partial)
    workflow.add_node("load_csv", load_csvs_as_context_partial)
    workflow.add_node("generate_textual_response", generate_textual_response_partial)

    workflow.add_edge(START, "compare_files")
    workflow.add_edge("compare_files", "get_relevant_files")
    workflow.add_edge("get_relevant_files", "load_csv")
    workflow.add_edge("load_csv", "generate_textual_response")
    workflow.add_edge("generate_textual_response", END)
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app, memory

