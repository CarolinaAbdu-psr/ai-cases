import json
import argparse
import yaml
import datetime as dt
import os
import operator
from typing import Tuple, List, Annotated, Dict, Any
import psr.factory

import logging
import traceback

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage, AnyMessage, ToolMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.tools import tool
from langchain_chroma import Chroma
import chromadb.config
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
from helper.common import get_vectorstore_base_path
from helper import rag_common


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# Load environment variables from .env file
load_dotenv()

REQUEST_TIMEOUT = 120
MAX_TOKENS = 4096

# Agent template settings
AGENTS_DIR = "agents"
AGENT_FILENAME = "case_edit_agent.yaml"


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
        return {'status': 'loaded'}
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
                        if 'retrive_properties' in self.tools and name != 'retrive_properties':
                            try:
                                retr = self.tools['retrive_properties'].invoke(state)
                                fallback_notes.append(('retrive_properties', retr))
                            except Exception:
                                fallback_notes.append(('retrive_properties', 'failed'))

                        # Try get_all_objects to help with exact names
                        if 'get_all_objects' in self.tools and name not in ('get_all_objects',):
                            try:
                                allobjs = self.tools['get_all_objects'].invoke({})
                                fallback_notes.append(('get_all_objects', allobjs))
                            except Exception:
                                fallback_notes.append(('get_all_objects', 'failed'))

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
                    err = f"TOOL_INVOCATION_EXCEPTION: {type(e).__name__}: {str(e)}\nTraceback:\n{tb}"

                    # Attempt basic recovery steps for invocation exceptions
                    fallback_notes = []
                    if 'retrive_properties' in self.tools and name != 'retrive_properties':
                        try:
                            retr = self.tools['retrive_properties'].invoke(state)
                            fallback_notes.append(('retrive_properties', retr))
                        except Exception:
                            fallback_notes.append(('retrive_properties', 'failed'))

                    if 'get_all_objects' in self.tools and name not in ('get_all_objects',):
                        try:
                            allobjs = self.tools['get_all_objects'].invoke({})
                            fallback_notes.append(('get_all_objects', allobjs))
                        except Exception:
                            fallback_notes.append(('get_all_objects', 'failed'))

                    # Try a single retry of the original tool
                    try:
                        retry = self.tools[name].invoke(t['args'])
                    except Exception as e2:
                        retry = f"RETRY_EXCEPTION: {type(e2).__name__}: {str(e2)}"

                    result = {
                        'invocation_exception': err,
                        'fallbacks': fallback_notes,
                        'retry_result': retry
                    }

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


def format_properties(docs: List) -> str:
    """
    Formats a list of retrieved documents into a context string
    that includes availables properties
    """
    formatted_blocks = []
    
    available_objects = "The available objects to be used at study.find() or study.create() are the following: ACInterconnection, Area, Battery, Bus, BusShunt, Circuit, CircuitFlowConstraint, CSP, DCBus, DCLine, Demand, DemandSegment, Emission, FlowController, Fuel, FuelConsumption, FuelContract, FuelProducer, FuelReservoir, GasEmission, GasNode, GasPipeline, GenerationConstraint, GenericConstraint, HydroGenerator, HydroPlant, HydroPlantConnection, HydroStation, HydroStationConnection, Interconnection, InterpolationGenericConstraint, LCCConverter, LineReactor, Load, MTDCLink, PaymentSchedule, PowerInjection, RenewableCapacityProfile, RenewableGenerator, RenewablePlant, RenewableStation, RenewableTurbine, RenewableWindSpeedPoint, ReserveGeneration, ReservoirSet, SensitivityGroup, SeriesCapacitor, StaticVarCompensator, SumOfCircuits, SumOfInterconnections, SupplyChainDemand, SupplyChainDemandSegment, SupplyChainFixedConverter, SupplyChainFixedConverterCommodity, SupplyChainNode, SupplyChainProcess, SupplyChainProducer, SupplyChainStorage, SupplyChainTransport, SynchronousCompensator, System, TargetGeneration, ThermalCombinedCycle, ThermalGenerator, ThermalPlant, ThreeWindingsTransformer, Transformer, TransmissionLine, TwoTerminalDCLink, VSCConverter, Waterway, Zone"
    
    formatted_blocks.append(available_objects)

    for i, doc in enumerate(docs):

        # 1. Get object name and metadata (properties)
        metadata = doc.metadata
        objct_name = doc.page_content
        
        # 3. Create example
        block = f"""
        Object Name: {objct_name}

        Madatory properties to create {objct_name}: {metadata.get("mandatory")}

        Reference properties wich must be used to link objects: {metadata.get("references_objects")}

        Static properties which can be acessed with .get(PropertyName) function and created by .set(PropertyName,value) function : {metadata.get("static_properties")}

        Dynamic properties which can be acessed with .get_df(PropertyName) or .get_at(PropertyName, date) functions and created by .set_df(df) 
        or .set_at(PropertyName, date, value) function : {metadata.get("dynamic_properties")}
        """

        formatted_blocks.append(block.strip())
        
    return "\n\n" + "\n\n".join(formatted_blocks)

@tool
def retrive_properties(state:AgentState)->str:
    """
    Retrieve detailed information about available object types and their properties from the SDDP study.
    
    Use this tool FIRST to understand:
    - What object types exist (e.g., ThermalPlant, HydroPlant, Bus)
    - What mandatory properties are needed to create each object
    - What static properties can be accessed with tool get_static_property 
    - What dynamic properties can be accessed 
    - What reference properties link objects together
    
    Returns: Formatted documentation of available objects and their properties.
    Use the property names returned here when calling other tools.
    """
    try:
        vectorstore = load_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
        last_message_content = state["messages"][-1].content
        docs = retriever.invoke(last_message_content)
        properties_str = format_properties(docs)
        return properties_str
    except Exception as e:
        tb = traceback.format_exc()
        return f"TOOL_ERROR: retrive_properties failed: {type(e).__name__}: {str(e)}\nTraceback:\n{tb}\nSuggested action: verify vectorstore exists and the last message content is valid."

@tool
def get_available_objects(obj_type):
    """Get all names (list of available instances) for a given object type in the study.
    
    Args:
        obj_type: The object type name (e.g., 'ThermalPlant', 'Bus', 'HydroPlant')
        
    Returns: String with description followed by dictionary mapping object keys to names.
    
    Use this to:
    - See what instances exist before filtering by properties
    - Match user-provided names with actual study objects
    """
    try:
        description = f"Dict with key : name for all {obj_type} objects: "
        name_id_map = {}
        for obj in STUDY.find(obj_type):
            name = ''
            if obj.has_name:
                name = obj.name.strip()
            name_id_map[obj.key] = name
        
        return description + str(name_id_map)

    except Exception as e:
        tb = traceback.format_exc()
        return f"TOOL_ERROR: get_available_names failed: {type(e).__name__}: {str(e)}\nTraceback:\n{tb}\nSuggested action: verify object type name and that STUDY is loaded."

def is_dataframe(obj,key): 
    """Check if the property is a dataframe"""
    description = obj.description(key)
    if description.is_dynamic():
        return True
    if len(description.dimensions()) > 0: 
        return True
    return False

@tool
def modify_element(obj_key, property, value):
    """Edit a given property of an object (don't use to change name, code or id)
    
    Args:
        object_type: The object type name (e.g., 'ThermalPlant', 'Bus', 'HydroPlant')
        code: the code of the object (can be None since the name is not none)
        name: the name of the given object (can be none since code is not none)
        property: name of the desired option to change
        value: the new value to set to the property of the object 
        
    Returns: Ture if succed
    
    Use this to:
    - Count the objects of a given type
    """
    try:
        obj = STUDY.get_by_key(obj_key)
        if obj:
            older_value = obj.get(property)
            obj.set(property,value) 
        
        return f"Object {obj} property {property} updated from {older_value} to {value}"
    except Exception as e:
        tb = traceback.format_exc()
        return f"TOOL_ERROR: modify_element failed: {type(e).__name__}: {str(e)}\nTraceback:\n{tb}\nSuggested action: verify object type and STUDY."

@tool   
def rename_element(obj_key, new_name: str):
    """Modify the name of a object
    
    Args:
        object_type: The object type name (e.g., 'ThermalPlant', 'Bus', 'HydroPlant')
        code: the code of the object (can be None since the name is not none)
        name: the name of the given object (can be none since code is not none)
        new_name : the new name desired (str)
        
    Returns: Ture if succed
    
    Use this to:
    - Count the objects of a given type
    """
    try:
        obj = STUDY.get_by_key(obj_key)
        if obj.has_name:
            older_name = obj.name.strip()
            obj.name = new_name
        
            return f"Object {obj} name updated from {older_name} to {new_name}"
        else: 
            return f"Object {obj} has no name property"

    except Exception as e:
        tb = traceback.format_exc()
        return f"TOOL_ERROR: count_objects_by_type failed: {type(e).__name__}: {str(e)}\nTraceback:\n{tb}\nSuggested action: verify object type and STUDY."
    
@tool   
def modify_element_code(obj_key, new_code: int):
    """Modify the code of a object
    
    Args:
        object_type: The object type name (e.g., 'ThermalPlant', 'Bus', 'HydroPlant')
        code: the code of the object (can be None since the name is not none)
        name: the name of the given object (can be none since code is not none)
        new_code : the new code desired (int)
        
    Returns: Ture if succed
    
    Use this to:
    - Count the objects of a given type
    """
    try:
        obj = STUDY.get_by_key(obj_key)
        if obj.has_id:
            older_code = obj.code
            obj.code = new_code
        
            return f"Object {obj} name updated from {older_code} to {new_code}"
        else: 
            return f"Object {obj} has no code property"

    except Exception as e:
        tb = traceback.format_exc()
        return f"TOOL_ERROR: count_objects_by_type failed: {type(e).__name__}: {str(e)}\nTraceback:\n{tb}\nSuggested action: verify object type and STUDY."

@tool   
def modify_element_key(obj_key, new_key: str):
    """Modify the code of a object
    
    Args:
        object_type: The object type name (e.g., 'ThermalPlant', 'Bus', 'HydroPlant')
        code: the code of the object (can be None since the name is not none)
        name: the name of the given object (can be none since code is not none)
        new_code : the new code desired (int)
        
    Returns: Ture if succed
    
    Use this to:
    - Count the objects of a given type
    """
    try:
        obj = STUDY.get_by_key(obj_key)
        obj.key = new_key
        return f"Object {obj} name updated from {obj_key} to {new_key}"
       

    except Exception as e:
        tb = traceback.format_exc()
        return f"TOOL_ERROR: count_objects_by_type failed: {type(e).__name__}: {str(e)}\nTraceback:\n{tb}\nSuggested action: verify object type and STUDY."


@tool
def get_all_objects():
    """Get all objects with its types, codes and names
    
    Returns: A list with all objects and its names 
    
    Use this to: 
    - Find easy the names of more thant one object to use in other functions that the exact name is required
    - Give a summary of the case and it's objets"""

    try:
        return STUDY.get_key_object_map()
    except Exception as e:
        tb = traceback.format_exc()
        return f"TOOL_ERROR: get_all_objects failed: {type(e).__name__}: {str(e)}\nTraceback:\n{tb}\nSuggested action: ensure STUDY is loaded and accessible."



@tool 
def create_modification(obj_key,property:str, modifications: dict):
    try: 
        obj = STUDY.get_by_key(obj_key)
        description = obj.description(property)
        if description.is_dynamic(): 
            for date,value  in modifications.items():
                obj.set_at(property,date,value)

            return f"Add modifications to property {property}: {modifications}"

        else: 
            return f"Property {property} does not change over time, it is a static value."
    except Exception as e:
        tb = traceback.format_exc()
        return f"TOOL_ERROR: get_all_objects failed: {type(e).__name__}: {str(e)}\nTraceback:\n{tb}\nSuggested action: ensure STUDY is loaded and accessible."


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

        global STUDY  
        STUDY = psr.factory.load_study(study_path)

        app, memory = create_langgraph_workflow(llm)
        
        return app, memory

    except Exception as e:
        logger.error(f"Error initializing RAG: {str(e)}")
        raise
    
    except Exception as e:
        logger.error(f"Error initializing RAG: {e}")
        raise
    
def create_langgraph_workflow(llm: BaseChatOpenAI):

    tools = [retrive_properties, get_available_names, get_all_objects,modify_element,rename_element]
    
    # Create agent with system prompt (as string, not list)
    agent = RAGAgent(llm, tools, SYSTEM_PROMPT_TEMPLATE)

    app,memory = agent._initialize_workflow()
    
    return app, memory

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




