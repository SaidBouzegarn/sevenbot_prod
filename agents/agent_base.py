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

set_llm_cache(InMemoryCache())



def create_agent_message_class(agent_name: str) -> Type[AIMessage]:
    """Dynamically create a new message class for a specific agent."""
    return type(f"{agent_name}Message", (AIMessage,), {
        "__doc__": f"A message from the {agent_name} Agent.",
        "agent_name": agent_name
    })

class BaseAgent:
    def __init__(
        self,
        name: str,
        llm: BaseLanguageModel,
        tools: List[BaseTool] = None,
        system_message: Optional[str] = None,
        max_iterations: int = 10,
        checkpointer: Optional[BaseCheckpointSaver] = None,
        memory_store: Optional[BaseStore] = None,
        debug: bool = False,
        **kwargs: Any
    ):
        self.name = name
        self.MessageClass = create_agent_message_class(name)
        self.llm = llm
        self.tools = tools or [DuckDuckGoSearchRun()]
        self.system_message = system_message
        self.max_iterations = max_iterations
        self.checkpointer = checkpointer or MemorySaver()
        self.memory_store = memory_store or InMemoryStore()
        self.debug = debug
        self.kwargs = kwargs
        self.graph = self._create_graph()

    def create_message(self, content: str) -> AIMessage:
        return self.MessageClass(content=content, agent_name=self.name)

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