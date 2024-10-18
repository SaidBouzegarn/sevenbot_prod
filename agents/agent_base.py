from typing import List, Optional, Dict, Any, Union, Callable, Sequence, TypedDict, Annotated, Literal, Type
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.schema.runnable import Runnable
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from PIL import Image
import io
import os
from langgraph.graph.message import add_messages
from langchain_community.tools import DuckDuckGoSearchRun
from jinja2 import Environment, FileSystemLoader
from langchain_openai import ChatOpenAI
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from langchain_mistralai import ChatMistralAI
from langchain_cohere import ChatCohere
from langchain_groq import ChatGroq
from langchain_google_vertexai import ChatVertexAI
from datetime import datetime

set_llm_cache(InMemoryCache())

# Lists of model names for each provider
OPENAI_MODELS = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"]
MISTRAL_MODELS = ["mistral-tiny", "mistral-small", "mistral-medium"]
COHERE_MODELS = ["command", "command-light", "command-nightly"]
GROQ_MODELS = ["llama2-70b-4096", "mixtral-8x7b-32768"]
VERTEXAI_MODELS = ["chat-bison", "chat-bison@001", "codechat-bison", "codechat-bison@001"]

def create_agent_message_class(agent_name: str) -> Type[AIMessage]:
    """Dynamically create a new message class for a specific agent."""
    return type(f"{agent_name}Message", (AIMessage,), {
        "__doc__": f"A message from the {agent_name} Agent.",
        "agent_name": agent_name,
        "timestamp": datetime.now().isoformat()

    })

class BaseAgent:
    def __init__(
        self,
        name: str,
        llm: str,
        llm_params: Dict[str, Any],
        assistant_llm: str,
        assistant_llm_params: Dict[str, Any],
        tools: List[BaseTool] = None,
        system_message: Optional[str] = None,
        max_iterations: int = 10,
        checkpointer: Optional[BaseCheckpointSaver] = None,
        memory_store: Optional[BaseStore] = None,
        debug: bool = False,
        **kwargs: Any
    ):
        self.name = name
        self.tools = tools or [DuckDuckGoSearchRun()]
        self.MessageClass = create_agent_message_class(name)
        self.llm = self._construct_llm(llm, llm_params)
        self.assistant_llm = self._construct_llm(assistant_llm, assistant_llm_params, tools=True)
        self.system_message = system_message
        self.max_iterations = max_iterations
        self.checkpointer = checkpointer or MemorySaver()
        self.memory_store = memory_store or InMemoryStore()
        self.debug = debug
        self.kwargs = kwargs

    def _construct_llm(self, llm_name: str, llm_params: Dict[str, Any], tools: bool = False) -> BaseLanguageModel:
        """Construct the appropriate LLM based on the input string and parameters."""
        if llm_name in OPENAI_MODELS:
            llm = ChatOpenAI(model_name=llm_name, **llm_params)
        elif llm_name in MISTRAL_MODELS:
            llm = ChatMistralAI(model=llm_name, **llm_params)
        elif llm_name in COHERE_MODELS:
            llm = ChatCohere(model=llm_name, **llm_params)
        elif llm_name in GROQ_MODELS:
            llm = ChatGroq(model=llm_name, **llm_params)
        elif llm_name in VERTEXAI_MODELS:
            llm = ChatVertexAI(model_name=llm_name, **llm_params)
        else:
            raise ValueError(f"Unsupported model: {llm_name}")
        
        if tools:
            return llm.bind_tools(self.tools)

        return llm

    def create_message(self, content: str, agent_name: str = None) -> AIMessage:
        return self.MessageClass(content=content, agent_name=agent_name if agent_name is not None else self.name, timestamp=datetime.now().isoformat())

    def _create_graph(self) -> StateGraph:
        raise NotImplementedError("Subclasses must implement this method")

    def run(self, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        state = self.state_schema(**inputs)
        default_config = {
            "thread_id": "default_thread",
            "checkpoint_ns": "default_namespace",
            "checkpoint_id": "default_checkpoint"
        }
        if config:
            default_config.update(config)
        return self.graph.invoke(state, default_config)

    def stream(self, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        state = self.state_schema(**inputs)
        default_config = {
            "thread_id": "default_thread",
            "checkpoint_ns": "default_namespace",
            "checkpoint_id": "default_checkpoint"
        }
        if config:
            default_config.update(config)
        return self.graph.stream(state, default_config)

    def arun(self, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        state = self.state_schema(**inputs)
        default_config = {
            "thread_id": "default_thread",
            "checkpoint_ns": "default_namespace",
            "checkpoint_id": "default_checkpoint"
        }
        if config:
            default_config.update(config)
        return self.graph.ainvoke(state, default_config)

    def astream(self, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        state = self.state_schema(**inputs)
        default_config = {
            "thread_id": "default_thread",
            "checkpoint_ns": "default_namespace",
            "checkpoint_id": "default_checkpoint"
        }
        if config:
            default_config.update(config)
        return self.graph.astream(state, default_config)

    @staticmethod
    def get_latest_message_list(list1: List[AIMessage], list2: List[AIMessage]) -> List[AIMessage]:
        """
        Compare two lists of messages and return the list with the most recent message.
        If both lists are empty, return an empty list.
        If one list is empty and the other is not, return the non-empty list.
        """
        if not list1 and not list2:
            return []
        if not list1:
            return list2
        if not list2:
            return list1
        
        last_msg1 = list1[-1]
        last_msg2 = list2[-1]
        
        time1 = datetime.fromisoformat(last_msg1.timestamp)
        time2 = datetime.fromisoformat(last_msg2.timestamp)
        
        return list1 if time1 > time2 else list2
