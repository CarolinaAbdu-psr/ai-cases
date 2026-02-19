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
AGENT_FILENAME = "case_input_agent.yaml"


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
def get_available_objects(obj_type:str):
    """Get all names (list of available instances) for a given object type in the study.
    
    Args:
        obj_type: The object type name (e.g., 'ThermalPlant', 'Bus', 'HydroPlant')
        
    Returns: String with description followed by dictionary mapping object keys to names.
    
    Use this to:
    - Find exact keys to use in get_object_summary() tool
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

@tool
def count_objects_by_type(object_type):
    """Get the count of a object type.
    
    Args:
        object_type: The object type name (e.g., 'ThermalPlant', 'Bus', 'HydroPlant')
        
    Returns: Number of objects of type "object_type"
    
    Use this to:
    - Count the objects of a given type
    """
    try:
        objs = STUDY.find(object_type)
        n = len(objs)
        return n
    except Exception as e:
        tb = traceback.format_exc()
        return f"TOOL_ERROR: count_objects_by_type failed: {type(e).__name__}: {str(e)}\nTraceback:\n{tb}\nSuggested action: verify object type and STUDY."

@tool
def get_all_objects():
    """Get all objects with its types, codes and names
    
    Returns: A dict with object key and obj
    
    Use this to: 
    - Find easy the names of more thant one object to use in other functions that the exact name is required
    - Give a summary of the case and it's objets"""

    try:
        return STUDY.get_key_object_map()
    except Exception as e:
        tb = traceback.format_exc()
        return f"TOOL_ERROR: get_all_objects failed: {type(e).__name__}: {str(e)}\nTraceback:\n{tb}\nSuggested action: ensure STUDY is loaded and accessible."

@tool
def get_object_summary(obj_key):
    """Find a specific object by its exact name/identifier.
    
    Args:
        obj_key: The object key
    Returns: The object matching the name, or empty result if not found.
    
    Use this to:
    - Locate a specific named object (e.g., 'Plant_ABC')
    - Retrieve an object before getting its properties
    - Validate if an object exists in the study
    """
    try:
        obj = STUDY.get_by_key(obj_key)
        description = f"Object Type: {obj.type}, Static Properties and References: {obj.as_dict() if obj else 'Not Found'}"
        return description
    except Exception as e:
        tb = traceback.format_exc()
        return f"TOOL_ERROR: get_object_summary failed: {type(e).__name__}: {str(e)}\nTraceback:\n{tb}\nSuggested action: verify object type and exact name."

@tool
def get_static_properties(obj_key, properties_names:list):
    """Get the values of static properties for a specific object.
    
    Args:
        obj_key: Object key (e.g., 'Thermal Plant Thermal 1 [1]')
        properties_names: List of static property names to retrieve (e.g., ['InstalledCapacity', 'Voltage'])
        
    Returns: Dictionary of {property_name: property_value} pairs.
    
    Use this to:
    - Retrieve specific properties of named objects (e.g., capacity of 'Plant_A')
    - Get a property value across all objects of a type
    - Verify property values before performing calculations
    
    Tip: Use retrive_properties first to find valid property names for your object type.
    """
    try:
        obj = STUDY.get_by_key(obj_key)
        result = {}
        if obj:
            for property in properties_names:
                result[property] = obj.get(property)

        return result
    except Exception as e:
        tb = traceback.format_exc()
        return f"TOOL_ERROR: get_static_properties failed: {type(e).__name__}: {str(e)}\nTraceback:\n{tb}\nSuggested action: verify type, property_name and object_name."
    
@tool
def get_dynamic_property(obj_key, property_name):
    """Get the value of a dynamic property (time-series data) for a specific object.
    
    Args:
        obj_key: Object key (e.g., 'Thermal Plant Thermal 1 [1]')
        property_name: The dynamic property name (e.g., 'EnergyPerBlock', 'Demand')
        
    Returns: DataFrame with dynamic property values as a string
    
    Use this to:
    - Retrieve dynamic (time-series) properties of objects
    - Get demand information (demand_segment, EnergyPerBlock)
    - Access time-series data for objects
    
    Tip: Use retrive_properties first to find valid dynamic property names for your object type.
    """
    try:
        obj = STUDY.get_by_key(obj_key)
        df = str(obj.get_df(property_name))
        return df
    except Exception as e:
        tb = traceback.format_exc()
        return f"TOOL_ERROR: get_dynamic_property failed: {type(e).__name__}: {str(e)}\nTraceback:\n{tb}\nSuggested action: verify obj_key and property_name are valid."

@tool
def find_by_property_condition(type, property_name, property_condition, condition_value):
    """Find all objects of a given type that match a property condition.
    
    Args:
        type: Object type (e.g., 'ThermalPlant', 'Bus')
        property_name: The property to filter by (e.g., 'InstalledCapacity')
        property_condition: The comparison operator - 'l' (less than), 'e' (equal), 'g' (greater than)
        condition_value: The value to compare against
        
    Returns: List of objects that match the condition.
    
    Use this to:
    - Find all plants with capacity > 100
    - Find all buses with voltage <= 500
    - Filter objects by any numeric property
    
    Examples:
    - type='ThermalPlant', property_name='InstalledCapacity', property_condition='g', condition_value=500
      → Returns all thermal plants with capacity > 500
    """
    try:
        objects = []
        all_objects = STUDY.find(type)
        for obj in all_objects:
            value = obj.get(property_name) #verificar se é estático     
            match = False
            if property_condition=="l":
                match = value < condition_value
            elif property_condition=="e":
                match = value == condition_value
            elif property_condition=="g":
                match = value > condition_value

            if match:
                objects.append(obj)
        return objects
    except Exception as e:
        tb = traceback.format_exc()
        return f"TOOL_ERROR: find_by_property_condition failed: {type(e).__name__}: {str(e)}\nTraceback:\n{tb}\nSuggested action: verify type and property_name, and that values are comparable."

@tool
def sum_by_property_condition(type, property_name, property_condition, condition_value):
    """Calculate the sum of a property across all objects matching a condition.
    
    Args:
        type: Object type (e.g., 'ThermalPlant', 'HydroGenerator')
        property_name: The property to sum (e.g., 'InstalledCapacity')
        property_condition: The filter condition - 'l' (less than), 'e' (equal), 'g' (greater than)
        condition_value: The threshold value for the condition
        
    Returns: Numeric sum of the property for all matching objects.
    
    Use this to:
    - Calculate total capacity of thermal plants with capacity > 100
    - Sum costs for expensive items (property > threshold)
    - Aggregate metrics for filtered subsets
    
    Example: Sum total capacity of all thermal plants with capacity >= 200
    """
    try:
        total = 0
        all_objects = STUDY.find(type)
        for obj in all_objects:
            value = obj.get(property_name) #verificar se é estático     
            match = False
            if property_condition=="l":
                match = value < condition_value
            elif property_condition=="e":
                match = value == condition_value
            elif property_condition=="g":
                match = value > condition_value

            if match:
                total += value 
        return total
    except Exception as e:
        tb = traceback.format_exc()
        return f"TOOL_ERROR: sum_by_property_condition failed: {type(e).__name__}: {str(e)}\nTraceback:\n{tb}\nSuggested action: verify type, property_name and that property values are numeric."

@tool
def count_by_property_condition(type, property_name, property_condition, condition_value):
    """Count how many objects of a type match a property condition.
    
    Args:
        type: Object type (e.g., 'ThermalPlant', 'HydroPlant')
        property_name: The property to evaluate (e.g., 'InstalledCapacity', 'MinimumOutput')
        property_condition: The comparison operator - 'l' (less than), 'e' (equal), 'g' (greater than)
        condition_value: The threshold value
        
    Returns: Integer count of matching objects.
    
    Use this to:
    - Count how many thermal plants have capacity > 500 MW
    - Count expensive items (cost > threshold)
    - Get statistics about filtered subsets
    
    Example: How many thermal plants have capacity >= 100?
    → type='ThermalPlant', property_name='InstalledCapacity', property_condition='g', condition_value=100
    """
    try:
        count = 0
        all_objects = STUDY.find(type)
        for obj in all_objects:
            value = obj.get(property_name) #verificar se é estático     
            match = False
            if property_condition=="l":
                match = value < condition_value
            elif property_condition=="e":
                match = value == condition_value
            elif property_condition=="g":
                match = value > condition_value

            if match:
                count+=1
        return count
    except Exception as e:
        tb = traceback.format_exc()
        return f"TOOL_ERROR: count_by_property_condition failed: {type(e).__name__}: {str(e)}\nTraceback:\n{tb}\nSuggested action: verify type and property_name."


def check_refererence(refs, reference_name):
    match = False
    for ref in refs: 
        if ref.name.strip() == reference_name:
            return True
    return match

@tool
def find_by_reference(type, reference_type, reference_name):
    """Find all objects of a type that are linked to a specific reference object.
    
    Args:
        type: Object type to search (e.g., 'ThermalPlant', 'Demand')
        reference_type: The reference property name (e.g., 'RefFuels', 'RefArea', 'RefBus')
        reference_name: The name of the reference object to match
        
    Returns: List of objects that have a link to the specified reference.
    
    Use this to:
    - Find all thermal plants using a specific fuel
    - Find all generators connected to a specific bus
    - Find all demand nodes in a specific area
    - Navigate relationships between objects
    
    Example: Find all thermal plants that use 'Natural_Gas' fuel
    → type='ThermalPlant', reference_type='RefFuels', reference_name='Natural_Gas'
    """
    try:
        objects = []
        all_objects = STUDY.find(type)
        for obj in all_objects:
            refs = obj.get(reference_type) #Ex: RefFuels  
            if not isinstance(refs, list):
                refs = [refs]
            match = check_refererence(refs,reference_name)
            if match:
                objects.append(obj)
        return objects
    except Exception as e:
        tb = traceback.format_exc()
        return f"TOOL_ERROR: find_by_reference failed: {type(e).__name__}: {str(e)}\nTraceback:\n{tb}\nSuggested action: verify reference_type and reference_name, and that STUDY is loaded."


@tool
def count_by_reference(type, reference_type, reference_name):
    """Count how many objects of a type are linked to a specific reference object.
    
    Args:
        type: Object type to count (e.g., 'ThermalPlant', 'Load')
        reference_type: The reference property name (e.g., 'RefFuels', 'RefBus', 'RefArea')
        reference_name: The name of the reference object to match
        
    Returns: Integer count of objects linked to the reference.
    
    Use this to:
    - Count thermal plants using a specific fuel
    - Count generators connected to a bus
    - Count loads in a specific area
    - Get statistics on object relationships
    
    Example: How many thermal plants use 'Coal' fuel?
    → type='ThermalPlant', reference_type='RefFuels', reference_name='Coal'
    """
    try:
        count = 0 
        all_objects = STUDY.find(type)
        for obj in all_objects:
            refs = obj.get(reference_type) #Ex: RefFuels  
            if not isinstance(refs, list):
                refs = [refs]
            match = check_refererence(refs,reference_name)
            if match:
                count+= 1
        return count
    except Exception as e:
        tb = traceback.format_exc()
        return f"TOOL_ERROR: count_by_reference failed: {type(e).__name__}: {str(e)}\nTraceback:\n{tb}\nSuggested action: verify reference parameters and STUDY."

@tool
def sum_property_by_reference(type, reference_type, reference_name, property):
    """Sum a property across all objects linked to a specific reference object.
    
    Args:
        type: Object type to aggregate (e.g., 'ThermalPlant', 'Generator')
        reference_type: The reference property name (e.g., 'RefFuels', 'RefBus', 'RefArea')
        reference_name: The name of the reference object to match
        property: The property to sum (e.g., 'InstalledCapacity', 'MinimumOutput')
        
    Returns: Numeric sum of the property for all matched objects.
    
    Use this to:
    - Sum total capacity of plants using a specific fuel
    - Sum all generation capacity connected to a bus
    - Sum costs for all items linked to a reference
    - Aggregate metrics by reference relationships
    
    Example: What is the total capacity of thermal plants using 'Natural_Gas'?
    → type='ThermalPlant', reference_type='RefFuels', reference_name='Natural_Gas', property='InstalledCapacity'
    """
    try:
        total = 0 
        all_objects = STUDY.find(type)
        for obj in all_objects:
            refs = obj.get(reference_type) #Ex: RefFuels  
            if not isinstance(refs, list):
                refs = [refs]
            match = check_refererence(refs,reference_name)
            if match:
                total += obj.get(property)
        return total 
    except Exception as e:
        tb = traceback.format_exc()
        return f"TOOL_ERROR: sum_property_by_reference failed: {type(e).__name__}: {str(e)}\nTraceback:\n{tb}\nSuggested action: verify reference and property names and that property values are numeric."


def get_conections(obj, next_level_objs): 

    connections = ""

    for item in obj.referenced_by():
        triple = f"({str(obj)[22:]},<-,{str(item)[22:]})\n"
        next_level_objs.append(item)
        connections += triple

    obj_dict = obj.as_dict()
    for property in obj_dict.keys():
        description = obj.description(property)
        if description: 
            if description.is_reference():
                if isinstance(obj.get(property),list): 
                    for item in obj.get(property):
                        triple = f"({str(obj)[22:]},->,{str(item)[22:]})\n"
                        next_level_objs.append(item)
                        connections += triple
                elif obj.get(property): 
                    item = obj.get(property)
                    triple = f"({str(obj)[22:]}, -> ,{str(item)[22:]})\n"
                    next_level_objs.append(item)
                    connections += triple

    return connections, next_level_objs

@tool
def get_neighboors(obj_key, max_level=1):
    """
    Inspect neighborhood (references) for a study object and return relation triples.

    Args:
        obj_key: Object key to inspect (e.g., 'ThermalPlant Thermal 1 [1]').
        max_level: Depth of traversal for neighbours (default is 1).

    Returns:
        A string containing one relation triple per line in the form:
        (SourceObject,->,TargetObject) for outgoing references
        (SourceObject,<-,ReferencingObject) for incoming references

    Usage:
    - Use this tool to explore relationships around a given object 
    - Find all related objects connected to a given object
    - Discover which system an element belongs to
    - Example: get_neighboors with obj_key='ThermalPlant Thermal 1 [1]', max_level=1
    """

    try:
        level = 0 
        connections = ""
        obj = STUDY.get_by_key(obj_key)
        if not obj:
            return f"TOOL_ERROR: get_neighboors: object not found for key={obj_key}."
        level_objs =[obj]
        next_level_objs = []

        while level < max_level:
            connections += f"Connections Level {level}\n"
            for obj in level_objs: 
                new_connections,next_level_objs = get_conections(obj, next_level_objs)
                connections += new_connections
            level_objs= next_level_objs
            next_level_objs = []
            level += 1

        return connections
    except Exception as e:
        tb = traceback.format_exc()
        return f"TOOL_ERROR: get_neighboors failed: {type(e).__name__}: {str(e)}\nTraceback:\n{tb}\nSuggested action: verify obj_key and max_level."

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

    tools = [retrive_properties, get_available_objects, get_all_objects,get_object_summary, get_static_properties, get_dynamic_property,
            find_by_property_condition, count_by_property_condition, sum_by_property_condition,
            find_by_reference, count_by_reference, count_objects_by_type, sum_property_by_reference,get_neighboors]
    
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




