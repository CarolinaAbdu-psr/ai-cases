import yaml
import datetime as dt
import os
import operator
from typing import Tuple, List, Annotated, Dict, Any
import uuid
import psr.factory

import logging
import traceback
from helper import rag_common
from langchain_core.messages import SystemMessage,AnyMessage, ToolMessage
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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# Load environment variables from .env file
load_dotenv()

REQUEST_TIMEOUT = 1200
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

def get_vectorstore_directory(name: str) -> str:
    base_path = get_vectorstore_base_path()
    return os.path.join(base_path, f'{name}_vectorstore')


def load_vectorstore(directory) -> Chroma:
    """Load a Chroma vectorstore persisted in `vectorstore` directory."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    persist_directory = os.path.join("vectorstores", directory)
    if os.path.exists(persist_directory):
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            client_settings=chromadb.config.Settings(anonymized_telemetry=False)
        )
    raise ValueError(f"Vectorstore not found: {persist_directory}")



@tool 
def retrive_artifacts(plan)->str:
    """
    Retrieves PSR Factory object schemas and property definitions from the documentation.

    Call this tool BEFORE modifying or reading any object property, so you know:
      - The exact object name to use (e.g., "ThermalPlant", not "Thermal Plant")
      - The exact property name to use (e.g., "MaximumCapacity", not "Max Capacity")
      - Whether the property is static (fixed value) or dynamic (time series)
      - Reference properties (Ref*) that link objects to each other

    --- VALID TYPES ---
      - "Object"   : Schema and structure of a data object (e.g., ThermalPlant, Battery, HydroPlant)
      - "Property" : A specific attribute of an object (e.g., MaximumCapacity, RefSystem)

    --- HOW TO BUILD THE `plan` ARGUMENT ---
    Break the user request into steps. Each step is a string with format:
        "Step: <name> | Type: <type>"

    Rules:
      1. Always retrieve the Object before its Properties — you need the schema first.
      2. For each property the user wants to read or edit, add a dedicated Property step.
      3. To find how two objects relate, search for "Ref" properties (e.g., "ThermalPlant RefSystem").
         Ref properties tell you the exact field name used to link one object to another.

    --- EXAMPLES ---

    # Edit a property
    User: "I want to edit the installed capacity of my Thermal Plants"
    plan = [
        "Step: Thermal Plant | Type: Object",
        "Step: Thermal Plant Maximum Capacity | Type: Property",
    ]

    # Edit multiple properties
    User: "I want to update the min and max capacity of my Batteries"
    plan = [
        "Step: Battery | Type: Object",
        "Step: Battery Maximum Capacity | Type: Property",
        "Step: Battery Minimum Capacity | Type: Property",
    ]

    # Understand how objects relate
    User: "How is a Thermal Plant connected to a Bus?"
    plan = [
        "Step: Thermal Plant | Type: Object",
        "Step: Thermal Plant RefBus | Type: Property",
    ]

    Returns:
        A deduplicated list of artifact descriptions, each containing:
        the artifact name, type, full description (including static/dynamic context). It also breturns the available objects 
        keys of a interesting type.
    """

    artifacts_json = f"factory_json_vectorstore"
    all_docs = []
    available_objects =  "Available Objects keys of interested type : "
    inspected_types = []
    seen_content = set()
    
    try: 
        vectorstore = load_vectorstore(artifacts_json)
        logger.info(f"Vectorstore loaded {vectorstore}")

        for step in plan:
            parts = step.split("|")
            query_name = parts[0].replace("Step:", "").strip()
            query_type = parts[1].replace("Type:", "").strip()
            retriever = vectorstore.as_retriever(search_kwargs={"k": 2 ,"filter": {"type":query_type}}) 

            step_debug_info = []
            logger.info(f"Retrieving artifacts for step: {step} - Query Name = {query_name}; Query Type = {query_type}")
            docs = retriever.invoke(query_name)
            for doc in docs:

                #Debug Info
                artifact_name = doc.page_content
                artifact_type = doc.metadata.get("type", "No Type")
                artifact_description = doc.metadata.get("page_content", "No Content")
                
                # Add doc info 
                doc_info = f"Artifact Name: {artifact_name} \n Artifact Type: {artifact_type} \n Description: {artifact_description} \n"
                step_debug_info.append(doc_info)
                if doc.page_content not in seen_content:
                    all_docs.append(doc_info)
                    seen_content.add(artifact_name)

                # Add available objects
                obj_type = artifact_name.split()[0]
                if obj_type not in inspected_types:
                    objs = STUDY.find(obj_type)
                    for obj in objs: 
                        available_objects += f"{obj.key}; "
                        inspected_types.append(obj_type)
                

            logger.info(f"Properties retrived:\n" + "\n".join(all_docs) + f"{available_objects}" )
    except Exception as e: 
        logger.error(f"Error retrieving artifacts: {e}")
        
    logger.info(f"TOTAL UNIQUE DOCS RETRIEVED: {len(all_docs)}")
    return f"Properties retrived:\n" + "\n".join(all_docs) + f"{available_objects}"


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
        obj_key : The key of the element to be mofied retrived wit the tool retrive_artifacts
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
            new_value = obj.get(property)
            logger.info(f"Object {obj} property {property} updated from {older_value} to {new_value}")
            return f"Object {obj} property {property} updated from {older_value} to {new_value}"
        else: 
            logger.info(f"No object with key {obj_key}")
            return f"No object with key {obj_key}"
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"TOOL_ERROR: modify_element failed: {type(e).__name__}: {str(e)}\nTraceback:\n{tb}\nSuggested action: verify object type and STUDY.")
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

            logger.info(f"Object {obj} name updated from {older_name} to {obj.name}")
            return f"Object {obj} name updated from {older_name} to {obj.name}"
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
        if obj.has_code:
            older_code = obj.code
            obj.code = new_code
            logger.info(f"Object {obj} name updated from {older_code} to {obj.code}")
            return f"Object {obj} name updated from {older_code} to {obj.code}"
        else:
            logger.info(f"Object {obj} has no code property") 
            return f"Object {obj} has no code property"

    except Exception as e:
        tb = traceback.format_exc()
        return f"TOOL_ERROR: modify_element_code failed: {type(e).__name__}: {str(e)}\nTraceback:\n{tb}\nSuggested action: verify object type and STUDY."

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
        logger.info(f"Object {obj} name updated from {obj_key} to {obj.key}")
        return f"Object {obj} name updated from {obj_key} to {new_key}"    

    except Exception as e:
        tb = traceback.format_exc()
        return f"TOOL_ERROR: modify_element_key failed: {type(e).__name__}: {str(e)}\nTraceback:\n{tb}\nSuggested action: verify object type and STUDY."

@tool 
def create_modification(obj_key,property:str, modifications: dict):
    """
    Create or update future modifications for study objects.

    This tool defines changes to object attributes that will become effective
    in future stages or time periods of the study.

    It should be used whenever the user requests modifications such as:
        - Adding new attributes that currently do not exist.
        - Changing parameter values in the future.
        - Defining time-dependent behavior for any object.

    Typical examples:
        - "In the future, plant A will have a capacity factor of 0.6"
        - "From 2028 onwards, this unit becomes unavailable"
        - "Set fuel cost to 120 starting next year"

    Usage:
        - Identify the target object.
        - Determine the attribute to be modified.
        - Define the new value.
        - Define when the modification becomes active.
        - Call this function to register the modification.

    Important:
        - If the attribute does not currently exist, this tool will create it.
        - If the attribute already exists, this tool will overwrite its value
          for the specified time interval.

    Returns:
        bool: True if the modification was successfully created or updated.
    """
    try: 
        obj = STUDY.get_by_key(obj_key)
        values = {}
        new_values = {}
        description = obj.description(property)
        if description.is_dynamic(): 
            for date,value  in modifications.items():
                previous_value = obj.get_at(property,date)
                values[date] = previous_value
                obj.set_at(property,date,value)
                new_values[date]=obj.get_at(property,date)
            logger.info(f"TOOL RESPONSE: Add modifications to property {property}. Previous values: {values}. New values: {new_values} ")
            return f"Add modifications to property {property}. Previous values: {values}. New values: {new_values}"

        else: 
            logger.info(f"Property {property} does not change over time, it is a static value.")
            return f"Property {property} does not change over time, it is a static value."
    except Exception as e:
        tb = traceback.format_exc()
        return f"TOOL_ERROR: get_all_objects failed: {type(e).__name__}: {str(e)}\nTraceback:\n{tb}\nSuggested action: ensure STUDY is loaded and accessible."
    


def get_mandatory_refs_identifier(obj_type):
    obj = STUDY.create(obj_type)
    obj_dict = obj.as_dict()
    refs = []
    for property in obj_dict.keys():
        if obj.description(property).is_reference():
            refs.append[property]
    return refs

def register_name(obj,obj_type,name,comments):
    # Name property
    if name and obj.has_name:
        obj.name = name 
        comments += f"Name: {obj.name}\n"
    elif not name and obj.has_name:
        obj.name = f"{obj_type}_{uuid.uuid4().hex[:6]}"
        comments += f"Automatic Name: {obj.name}\n "
    return comments

def register_code(obj,obj_type,code,comments):
    if code and obj.has_code:
        if len(STUDY.find_by_code[code])==0:
            obj.code = code
            comments += f"Code: {obj.code}\n"
        else: 
            last_code = STUDY.find(obj_type)[-1].code
            obj.code = last_code +1
            obj.code = code
            comments += f"Code provided {code} was not available. Code registered: {obj.code}\n"
    elif not code and obj.has_code:
        last_code = STUDY.find(obj_type)[-1].code
        obj.code = last_code +1
        comments += f"Automatic Code: {obj.code}\n"
    return comments

def register_id(obj,id,comments):
    if id and obj.has_id:
        obj.id = id 
        comments += f"ID: {obj.id}\n"
    elif not id and obj.has_id:
        obj.id = f"{uuid.uuid4().hex[:2]}"
        comments += f"Automatic id: {obj.id}\n "
    return comments  

def register_keys(obj, key, comments):
    if key:
        obj.key = key
        comments += f"Key: {obj.key}\n"
    else:
        obj.key = f"{obj.name} [{obj.code}]\n"
        comments += f"Automatic Key: {obj.key}\n"
    return comments

@tool 
def create_element(obj_type:str, name:str= None, code:int =None, key:str=None, id=None,  properties:dict={}):
    """
    Creates a new element within the STUDY environment and registers its properties.
    
    This tool should be used whenever a new object (e.g., 'System', 'Fuel') 
    needs to be instantiated. The tool automatically handles mandatory references 
    by assigning 'Default' values if they are not explicitly provided.

    Args:
        obj_type (str): Mandatory. The type of object to create (e.g., "System").
        name (str, optional): A suggestive name for the element. 
            Constraint: Maximum 12 characters. 
            The final object key will follow the pattern: "Name [code]".
        id (str, optional): Unique identifier for the object
            Constraint: Maximum 2 characters.
        key (str, optional): Specific key for the object.
        code (int, optional): Unique code for the object. 
            Note: Avoid providing this unless necessary, as codes must be unique 
            across the study.
        properties (dict, optional): A dictionary of property names and values to set.
            - If a mandatory reference property is missing, the tool will 
              automatically find and assign the first available object of that type.
            - Supports both single references and lists of references.

    Returns:
        str: A log of the creation process, including all set properties 
            and default assignments.
    """

    obj = STUDY.create(obj_type)
    comments = f"Object of type {obj_type} created\n"

    # Basic properties
    comments = register_name(obj, obj_type,name,comments)
    comments = register_code(obj,obj_type,code,comments)
    comments = register_keys(obj, key, comments)
    comments = register_id(obj,id,comments)

    # Set properties
    for property_name in obj.descriptions().keys():
        property_description = obj.description(property_name)
        if property_name in properties.keys():

            # Set defined references 
            if property_description.is_reference():
                ref_obj_key = properties[property_name]

                # List
                if isinstance(ref_obj_key,list):
                    # case where the user gave a list of objects 
                    obj.set(property_name,ref_obj_key)
                else: 
                    if "List" in property_description.type_description():
                        obj.set(property_name,[ref_obj_key])
                        comments += f"Reference {property_name}: [{ref_obj_key}]\n"
                    else:
                        obj.set(property_name,ref_obj_key)
                        comments += f"Reference {property_name}: {ref_obj_key}\n"

            # Set other properties
            else: 
                try: 
                    value = properties[property_name]
                    obj.set(property_name,value)
                    comments += f"Property {property_name}: {value}"
                except:
                    continue

        
        # Set default references
        elif property_description.is_reference() and property_description.is_required():
            try: 
                desc_type = property_description.type_description()
                if "List" in desc_type:
                    ref_obj_type = desc_type.split()[-1][:-1] # "List of DataObject of type(s): Fuel" -> Fuel
                    ref_obj = STUDY.find(ref_obj_type)[0]
                    obj.set(property_name,[ref_obj])
                    comments += f"Default Reference {property_name}: [{ref_obj}]\n"
                else:
                    ref_obj_type = desc_type.strip() # System 
                    ref_obj = STUDY.find(ref_obj_type)[0]
                    obj.set(property_name,ref_obj)
                    comments += f"Default Reference {property_name}: {ref_obj}\n"
            except Exception as e: 
                print(f"{e}")
                continue

    try: 
        STUDY.add(obj)  
        logger.info(comments)  
        return comments
    except Exception as e:
        tb = traceback.format_exc()
        return f"TOOL_ERROR: create_element: {type(e).__name__}: {str(e)}\nTraceback:\n{tb}\nSuggestion: Abort actions."



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

    tools = [retrive_artifacts,modify_element,rename_element,modify_element_code,
            modify_element_key, create_modification,create_element]
    
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
