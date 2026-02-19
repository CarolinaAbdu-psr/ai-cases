import yaml
import os
import operator
from pathlib import Path
from typing import Tuple, List, Annotated, Dict, Any
import psr.factory
import psr.outputs

import logging
import traceback

from langchain_core.messages import SystemMessage, AnyMessage, ToolMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.tools import tool
from langchain_chroma import Chroma
import chromadb.config
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
from helper import rag_common


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

REQUEST_TIMEOUT = 300
MAX_TOKENS = 4096

# Agent template settings
AGENTS_DIR = "agents"
AGENT_FILENAME = "case_output_agent.yaml"


# -----------------------------
# Load agent configuration
# -----------------------------

def load_agent_config(filepath: str) -> bool:
    """Load prompts and templates from the agent YAML configuration."""
    global SYSTEM_PROMPT_TEMPLATE
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)

        SYSTEM_PROMPT_TEMPLATE = config['system_prompt_template']

        logger.info(f"Agent template loaded successfully from {filepath}")
        return True
    except (FileNotFoundError, yaml.YAMLError, KeyError) as e:
        logger.error(f"Error loading Agent from {filepath}: {e}")
        return False


def load_agents_config() -> Dict[str, Any]:
    """Load agent configuration file."""
    filepath = os.path.join(AGENTS_DIR, AGENT_FILENAME)
    if load_agent_config(filepath):
        logger.info(f"Agent {filepath} loaded")
        return {'status': 'loaded'}
    logger.error(f"Agent {filepath} loading failed")
    return {'status': 'failed'}

# Load configuration once
load_agents_config()

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage],operator.add] # Add messages to state (humam or ai result)

class RAGAgent:

    def __init__(self, model, tools, system):
        self.system = system
        # Bind tools to the model so it knows it can call them
        self.model = model.bind_tools(tools)
        self.tools = {t.name: t for t in tools}

    def _initialize_workflow(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("llm", self.call_llm)
        workflow.add_node("retriver",self.take_action)

        workflow.add_conditional_edges(
            'llm',
            self.exists_action,
            {True:'retriver',False: END}
        )
        workflow.add_edge('retriver','llm')
        workflow.set_entry_point('llm')


        memory = MemorySaver()

        self.workflow = workflow.compile(checkpointer=memory)

        return self.workflow, memory


    def exists_action(self,state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls)>0 
    
    def call_llm(self, state: AgentState):
        logger.info("Calling LLM")
        messages = state['messages'] 
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)  # AI response
        return {'messages': [message]}

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls  # use tools 
        results = []
        for t in tool_calls:
            logger.info(f"Calling Tool: {t}")
            name = t['name']
            if not name in self.tools:
                logger.warning(f"Tool {t} doesn't exist")
                result = "Incorrect tool name. Retry and select an available tool"
            else:
                try:
                    result = self.tools[name].invoke(t['args'])

                    # If the tool returned a structured TOOL_ERROR string, attempt recovery
                    if isinstance(result, str) and result.startswith("TOOL_ERROR"):
                        fallback_notes = []
                        # Try retrive_properties to help correct property/reference names
                        if 'convert_output' in self.tools and name != 'convert_output':
                            try:
                                retr = self.tools['convert_output'].invoke(state)
                                fallback_notes.append(('convert_output', retr))
                            except Exception:
                                fallback_notes.append(('convert_output', 'failed'))

                        # Retry the original tool once
                        try:
                            retry = self.tools[name].invoke(t['args'])
                        except Exception as e:
                            retry = f"RETRY_EXCEPTION: {type(e).__name__}: {str(e)}"

                        result = {
                            'original_error': str(result),
                            'fallbacks': fallback_notes,
                            'retry_result': retry
                        }

                except Exception as e:
                    tb = traceback.format_exc()
                    logger.warning = f"TOOL_INVOCATION_EXCEPTION: {type(e).__name__}: {str(e)}\nTraceback:\n{tb}"


            results.append(ToolMessage(tool_call_id=t['id'], name=name, content=str(result)))
        logger.info("Tools Execution Complete. Back to the model")
        return {'messages': results}
    


# -----------------------------
# Retrive context
# -----------------------------

def load_vectorstore() -> Chroma:
    """Load a Chroma vectorstore persisted in `vectorstore` directory."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    persist_directory = "vectorstore"
    if os.path.exists(persist_directory):
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            client_settings=chromadb.config.Settings(anonymized_telemetry=False)
        )
    raise ValueError(f"Vectorstore not found: {persist_directory}")


@tool 
def list_enabled_outputs():
    """
    List all currently enabled outputs in the study case.

    This tool returns a dictionary containing only the outputs that are currently
    enabled (active) in the study, mapping each output filename to its description.

    This function should be used when:
        - The user asks which outputs are currently active or available.
        - The agent needs to inspect which outputs can be accessed or converted.
        - As a validation step before calling `convert_output`.

    Usage:
        - Call this function to retrieve the list of enabled outputs.
        - If file paths are required, use `list_enabled_outputs_paths` instead.

    Returns:
        dict[str, str]: A dictionary mapping output filenames to their descriptions.
    """
    try: 
        case_path = CASE_PATH
        outputs = {}
        outputs_list = psr.outputs.get_available_outputs(case_path)
        for output in outputs_list:
            outputs[output.filename]=output.description
        return outputs
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"TOOL_ERROR: list_enabled_outputs  failed: {type(e).__name__}: {str(e)}\nTraceback:\n{tb}\n.")
        return f"TOOL_ERROR: list_enabled_outputs  failed: {type(e).__name__}: {str(e)}\nTraceback:\n{tb}\n."


    

@tool 
def list_enabled_outputs_paths():
    """
    List full file paths of all currently enabled outputs in the study case.

    This tool returns a dictionary mapping each enabled output's absolute file path
    to its description.

    This function should be used when:
        - The agent needs the physical file path for reading, converting or exporting outputs.
        - Before calling `convert_output`, to validate valid input paths.

    Usage:
        - Call this function to retrieve all enabled output paths.
        - Select the desired file path and pass it to `convert_output`.

    Returns:
        dict[str, str]: A dictionary mapping full file paths to output descriptions.
    """
    try:
        case_path = CASE_PATH
        outputs_list = psr.outputs.get_available_outputs(case_path)
        paths = {}
        for output in outputs_list:
            filename = output.filename 
            ext = output.file_type
            path = str(Path(case_path) / f"{filename}.{ext}")
            paths[path] = output.description
        return paths
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"TOOL_ERROR: list_enabled_outputs_paths: {type(e).__name__}: {str(e)}\nTraceback:\n{tb}\n.")
        return f"TOOL_ERROR: list_enabled_outputs_paths: {type(e).__name__}: {str(e)}\nTraceback:\n{tb}\n."


@tool
def convert_output(file_path, format):
    """
    Convert an enabled output file to a different format.

    This tool converts an existing output file into another format such as CSV,
    bin, hdr. The input file must exist and must correspond to an enabled
    output from the study.

    This function should be used after:
        - Calling `list_enabled_outputs_paths` to obtain valid file paths.

    Usage:
        - Call list_enabled_outputs_paths().
        - Select the desired file path.
        - Call convert_output(file_path, target_format).
        - Supported target-formats:  'binpair': PSR binary format split in two files: header (hdr) and data (bin), 'singlebin' : PSR binary format in a single file (dat),
        'csv'

    Example:
        convert_output(
            "C:/study/case/outputs/gerter.dat",
            "csv"
        )

    Args:
        file_path (str): Full path of the output file to be converted.
        format (str): Target format (e.g., 'csv', 'binpair', 'singlebin').

    Returns:
        str: Confirmation message indicating successful conversion.
    """
    try: 
        p = Path(file_path)
        new_path = str(p.with_suffix(f".{format}"))
        psr.factory.convert_output(file_path,new_path)
        return "Converted with succes"
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"TOOL_ERROR: convert_output failed: {type(e).__name__}: {str(e)}\nTraceback:\n{tb}\nSuggested action: verify with the filepath exist. Call list_available_outputs_paths first.")
        return f"TOOL_ERROR: convert_output failed: {type(e).__name__}: {str(e)}\nTraceback:\n{tb}\nSuggested action: verify with the filepath exist. Call list_available_outputs_paths first."

@tool
def get_output_num():
    """
    Retrieve all available outputs and their descriptions.

    This tool returns a dictionary mapping each output ID (Num) to its textual
    description. It must be used as an intermediate step to interpret the user's
    query and identify which outputs should be enabled or disabled.

    Usage:
        - Call this function to obtain all (id → description) pairs.
        - Compare the user query semantically against the descriptions.
        - Select the output IDs whose descriptions best match the user request.
        - Pass the selected IDs to `change_output_availability`.

    Typical examples:
        - User: "Enable marginal cost outputs"
          → Match query with descriptions, find corresponding Num values,
            then call change_output_availability.

        - User: "Disable hydro generation reports"
          → Identify matching descriptions and extract their Num.

    Returns:
        dict[int, str]: A dictionary mapping output ID (Num) to output description.
    """
    try:
        case_path = CASE_PATH
        df = psr.outputs.load_index_dat(case_path)
        d = df.set_index('Num')['Description'].to_dict()
        return str(d)
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"TOOL_ERROR: get_output_num failed: {type(e).__name__}: {str(e)}\nTraceback:\n{tb}\n")
        return f"TOOL_ERROR: get_output_num failed: {type(e).__name__}: {str(e)}\nTraceback:\n{tb}\n"


def change_output_availability(outputs:dict[int,bool]):
    """
    Enable or disable study outputs based on their IDs.

    This tool updates the availability status of one or more outputs in the study.
    The input must be a dictionary mapping output IDs (Num) to boolean values,
    where:

        - True  → enable the output
        - False → disable the output

    This function must always be called after identifying the correct output IDs
    using `get_output_num`, by matching the user query against the output
    descriptions.

    Usage:
        - First, call `get_output_num()` to retrieve all (id → description) pairs.
        - Determine which outputs match the user's request.
        - Build a dictionary {Num: action}.
        - Call this function to apply the changes.

    Example:
        User: "Disable thermal generation and deficit outputs"

        Step 1: Call get_output_num()
        Step 2: Identify matching IDs, e.g. {3, 7}
        Step 3: Call:
            change_output_availability({
                3: False,
                7: False
            })

    Args:
        outputs (dict[int, bool]): Dictionary mapping output ID (Num) to action:
                                   True to enable, False to disable.

    Returns:
        dict: Updated outputs dict with output name and boolean value (True = enabled or  False = disabled) .
    """
    try: 
        df_out = psr.outputs.load(CASE_PATH)
        result = {}
        for num, action in outputs.items():
            num = float(num)
            filter = df_out['Num'] == num
            df_out.loc[filter, 'Active'] = action
            result.update(df_out.loc[filter, 'Active'].to_dict())
        psr.outputs.save(df_out,case_path=CASE_PATH)
        return result
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"TOOL_ERROR: change_output_availability failed: {type(e).__name__}: {str(e)}\nTraceback:\n{tb}\nSuggested action: confirm that the provided output IDs are valid by calling get_output_num first.")
        return f"TOOL_ERROR: change_output_availability failed: {type(e).__name__}: {str(e)}\nTraceback:\n{tb}\nSuggested action: confirm that the provided output IDs are valid by calling get_output_num first."

    
# -------------------------------------------------------
# Create workflow
#--------------------------------------------------------


def create_langgraph_workflow(llm: BaseChatOpenAI):

    tools = [list_enabled_outputs,list_enabled_outputs_paths,convert_output, get_output_num]
    
    # Create agent with system prompt (as string, not list)
    agent = RAGAgent(llm, tools, SYSTEM_PROMPT_TEMPLATE)

    app,memory = agent._initialize_workflow()
    
    return app, memory


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

        global CASE_PATH
        CASE_PATH = study_path

        app, memory = create_langgraph_workflow(llm)
        
        return app, memory

    except Exception as e:
        logger.error(f"Error initializing RAG: {str(e)}")
        raise
    
    except Exception as e:
        logger.error(f"Error initializing RAG: {e}")
        raise


# -------------------------------------------------------
# Vectorstore download functions
#--------------------------------------------------------

# Delegate to common RAG utilities
get_rag_list = rag_common.get_rag_list
get_rag_list_with_dates = rag_common.get_rag_list_with_dates
extract_rag_to_folder = rag_common.extract_rag_to_folder
get_latest_rag_date = rag_common.get_latest_rag_date
download_rag = rag_common.download_rag
download_latest_rag = rag_common.download_latest_rag




