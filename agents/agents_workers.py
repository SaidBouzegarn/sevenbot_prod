from typing import List, Optional, Dict, Any, Union, Callable
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage
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
from typing import (
    Annotated,
    Sequence,
    TypedDict,
)
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_community.tools import DuckDuckGoSearchRun

class WorkerAgent:
    def __init__(
        self,
        llm: BaseLanguageModel,
        tools: List[BaseTool] = None,
        system_message: Optional[str] = None,
        max_iterations: int = 10,
        state_schema: Optional[BaseModel] = None,
        state_modifier: Optional[Union[str, SystemMessage, Callable, Runnable]] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
        memory_store: Optional[InMemoryStore] = None,
        interrupt_before: Optional[List[str]] = None,
        interrupt_after: Optional[List[str]] = None,
        debug: bool = False,
        **kwargs: Any
    ):
        self.llm = llm
        self.tools = tools or [DuckDuckGoSearchRun()]  # Use DuckDuckGo search as default tool
        self.system_message = system_message
        self.max_iterations = max_iterations
        self.state_schema = state_schema
        self.state_modifier = state_modifier
        self.checkpointer = checkpointer or MemorySaver()  # Default to MemorySaver if not provided
        self.memory_store = memory_store or InMemoryStore()
        self.interrupt_before = interrupt_before
        self.interrupt_after = interrupt_after
        self.debug = debug
        self.kwargs = kwargs
        self.graph = self._create_graph()

    def _create_graph(self) -> StateGraph:
        # Define the default state schema if not provided
        if self.state_schema is None:
            class AgentState(TypedDict):
                messages: Annotated[Sequence[BaseMessage], add_messages]
                is_last_step: bool = Field(default=False)
            self.state_schema = AgentState

        def should_continue(state: AgentState):
            messages = state["messages"]
            last_message = messages[-1]
            # If there is no function call, then we finish
            if not last_message.tool_calls:
                return "end"
            # Otherwise if there is, we continue
            else:
                return "continue"

        def call_model(state):
            # Apply state_modifier if provided
            if self.state_modifier:
                if isinstance(self.state_modifier, (str, SystemMessage)):
                    system_message = self.state_modifier if isinstance(self.state_modifier, SystemMessage) else SystemMessage(content=self.state_modifier)
                    state['messages'] = [system_message] + state['messages']
                elif callable(self.state_modifier):
                    state = self.state_modifier(state)
                elif isinstance(self.state_modifier, Runnable):
                    state = self.state_modifier.invoke(state)

            messages = state['messages']
            response = self.llm.invoke(messages)
            return {"messages": messages + [response]}

        tool_node = ToolNode(self.tools)

        workflow = StateGraph(self.state_schema)
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", tool_node)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", should_continue,    {
        # If `tools`, then we call the tool node.
        "continue": "tools",
        # Otherwise we finish.
            "end": END,
        }, 
        )
        workflow.add_edge("tools", "agent")

        return workflow.compile(checkpointer=self.checkpointer, store=self.memory_store)

    def run(self, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}
        if 'configurable' not in config:
            config['configurable'] = {}
        if 'thread_id' not in config['configurable']:
            config['configurable']['thread_id'] = 'default_thread'
        
        if 'messages' not in inputs:
            inputs = {'messages': inputs}
        
        return self.graph.invoke(inputs, config)

    def stream(self, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        return self.graph.stream(inputs, config)

    def arun(self, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        return self.graph.ainvoke(inputs, config)

    def astream(self, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        return self.graph.astream(inputs, config)

# Example usage:
if __name__ == "__main__":
    from langchain_core.tools import Tool
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    from dotenv import load_dotenv

    # Load environment variables from .env file
    load_dotenv()
    # Helper function for formatting the stream nicely
    def print_stream(stream):
        for s in stream:
            message = s["messages"][-1]
            if isinstance(message, tuple):
                print(message)
            else:
                message.pretty_print()

    tools = [DuckDuckGoSearchRun()]

    # Use ChatOpenAI instead of ChatAnthropic
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)
    agent = WorkerAgent(
        llm=llm,
        tools=tools,
        system_message="You are a helpful AI assistant.",
        max_iterations=5,
        checkpointer=MemorySaver(),  # Using MemorySaver as the checkpointer
        debug=True
    )

    # Generate the Mermaid graph
    mermaid_png = agent.graph.get_graph().draw_mermaid_png()

    # Convert the PNG data to a PIL Image
    img = Image.open(io.BytesIO(mermaid_png))

    # Save the image
    img.save('agent_graph.png')


    # Provide a config with a thread_id
    config = {"configurable": {"thread_id": "weather_thread"}}
    #result = agent.run({"messages": [HumanMessage(content="What's the weather in San Francisco?")]}, config=config)
    inputs = {
        "messages": [("user", "what is the weather in Paris")],
    }
    print_stream(agent.graph.stream(inputs, config, stream_mode="values"))
