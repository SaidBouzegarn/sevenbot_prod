from typing import List, Optional, Dict, Any, Union, Callable, Sequence, TypedDict, Annotated, Literal, Type
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.pydantic_v1 import BaseModel, Field, create_model
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
from agent_base import BaseAgent
from langchain_core.messages import trim_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from datetime import datetime
from dotenv import load_dotenv
import os
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
import json
import logging

# Set the logging level for the SageMaker SDK to WARNING or higher
logging.getLogger('sagemaker').setLevel(logging.WARNING)


set_llm_cache(InMemoryCache())

def pydantic_to_json(pydantic_obj):
    # Convert to dictionary and then to a compact JSON string
    obj_dict = pydantic_obj.dict()
    # Use separators to minimize whitespace: (',', ':') removes spaces after commas and colons
    compact_string = json.dumps(obj_dict, separators=(',', ':'))
    return compact_string

##########################################################################################
#################################### Level 1 agent #######################################
##########################################################################################


class Level1Decision(BaseModel):
    reasoning: str
    decision: Literal["search_more_information", "converse_with_superiors"]
    content: List[str] = Field(min_items=1, max_items=5)



# Define trimmer
# count each message as 1 "token" (token_counter=len) and keep only the last two messages
trimmer = trim_messages(strategy="last", max_tokens=3, token_counter=len)




class Level1Agent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_schema = self._create_dynamic_state_schema()
        self.attr_mapping = self._create_attr_mapping()
        self.prompt_dir = os.path.join(os.path.dirname(__file__), '..', 'prompts/level1', self.name)
        self.jinja_env = Environment(loader=FileSystemLoader(self.prompt_dir))
        
        # Generate the system prompt once during initialization
        system_prompt_template = self.jinja_env.get_template('system_prompt.j2')
        self.system_prompt = system_prompt_template.render(tools=self.tools)
        self.system_message = SystemMessage(content=self.system_prompt)
        self.graph = self._create_graph()

    def _create_dynamic_state_schema(self):
        return create_model(
            f'{self.name}_Level1State',
            **{
                "level1_2_conversation": (Annotated[List, add_messages], ...),
                "level1_3_conversation": (Annotated[List, add_messages], ...),
                f"{self.name}_is_first_execution": (bool, Field(default=True)),
                f"{self.name}_assistant_conversation": (Annotated[List, add_messages], ...),
                f"{self.name}_domain_knowledge": (Any, ...),
                f"{self.name}_mode": (Literal["research", "converse"], Field(default="research")),
                f"{self.name}_messages": (Annotated[List, add_messages], ...),
            },
            __base__=BaseModel
        )

    def _create_attr_mapping(self):
        return {
            "assistant_conversation": f"{self.name}_assistant_conversation",
            "domain_knowledge": f"{self.name}_domain_knowledge",
            "mode": f"{self.name}_mode",
            "messages": f"{self.name}_messages",
            "is_first_execution": f"{self.name}_is_first_execution",
        }

    def get_attr(self, state, attr_name):
        return getattr(state, self.attr_mapping.get(attr_name, attr_name))

    def set_attr(self, state, attr_name, value):
        setattr(state, self.attr_mapping.get(attr_name, attr_name), value)

    def _create_graph(self) -> StateGraph:
        
        def level1_node(state: self.state_schema) -> Dict[str, Any]:
            
            if not self.get_attr(state, "messages"):
                self.get_attr(state, "messages").append(self.system_message)
                
            #we want to get the last message from the level 3 conversation or level2 conversation based on the one that has last msg
            time1 = datetime.fromisoformat(self.get_attr(state, "level1_2_conversation")[-1].timestamp)
            time2 = datetime.fromisoformat(self.get_attr(state, "level1_3_conversation")[-1].timestamp)

            last_message = self.get_attr(state, "level1_2_conversation") if time1 > time2 else self.get_attr(state, "level1_3_conversation")
            last_message = last_message[-1].content if last_message else "No messages from level 2 or level 3 yet."

            if self.debug:
                print(f"Last message: {last_message}")

            decision_prompt = self.jinja_env.get_template('decision_prompt.j2').render(
                last_message=last_message,
                level1_2_conversation=trimmer.invoke(self.get_attr(state, "level1_2_conversation")) if self.get_attr(state, "level1_2_conversation") else "No messages from level 2 yet.",
                level1_3_conversation=trimmer.invoke(self.get_attr(state, "level1_3_conversation")) if self.get_attr(state, "level1_3_conversation") else "No messages from level 3 yet.",
                assistant_conversation=trimmer.invoke(self.get_attr(state, "assistant_conversation")) if self.get_attr(state, "assistant_conversation") else "No messages from assistant yet.",
                tools=self.tools
            )
            
            if self.get_attr(state, "is_first_execution"):
                self.get_attr(state, "messages").append(self.system_prompt)
                self.set_attr(state, "is_first_execution", False)

            # Use the decision prompt
            message = HumanMessage(content=decision_prompt)
            self.get_attr(state, "messages").append(message)
            structured_llm = self.llm.with_structured_output(Level1Decision)
            response = structured_llm.invoke(message)

            if self.debug:
                print(f"Reasoning: {response.reasoning}")
                print(f"Decision: {response.decision}")
                print(f"Content: {response.content}")

            response.content = " ".join(response.content)
            if response.decision == "search_more_information":
                questions = response.content
                message = self.create_message(questions)
                self.get_attr(state, "assistant_conversation").extend([message])
                state.mode = "research"
            elif response.decision == "converse_with_superiors":
                message = self.create_message(response.content)
                if time1 > time2:
                    self.get_attr(state, "level1_2_conversation").append(message)
                else:
                    self.get_attr(state, "level1_3_conversation").append(message)
                state.mode = "converse_with_superiors"
            
            response = self.create_message(response)
            self.get_attr(state, "messages").append(response)

            return state

        def assistant_node(state: self.state_schema) -> Dict[str, Any]:
            prompt = self.jinja_env.get_template('assistant_prompt.j2')

            # Get the last 5 messages from the conversation
            last_message = self.get_attr(state, "assistant_conversation")[-1]

            if last_message.agent_name == self.name:
                print(f"Processing question from {self.name}: {last_message.content}")
                
                response = self.assistant_llm.invoke(self.create_message(content=prompt.render(question=last_message.content)))

                response = self.create_message(response, agent_name=f"assistant_{self.name}")
                            
                self.get_attr(state, "assistant_conversation").extend(response)

            if self.debug:
                print(f"Answer: {response}")
            
            return state


        def should_continue(state: self.state_schema):
            # Check if there are any unanswered questions in the assistant conversation
            last_message = state.assistant_conversation[-1]

                # If the last message is from AI and has tool calls, continue to tools
            if last_message.tool_calls or last_message.content.tool_calls:
                return "continue"
            # If the last message is from AI but has no tool calls, it's an answer
            else:
                # If all questions are answered, go back to level1
                return "executive_agent"

        tool_node = ToolNode(self.tools)

        workflow = StateGraph(self.state_schema)
        workflow.add_node(f"agent_{self.name}", level1_node)
        workflow.add_node(f"assistant_{self.name}", assistant_node)
        workflow.add_node(f"tools_{self.name}", tool_node)

        workflow.set_entry_point(f"agent_{self.name}")
        
        workflow.add_conditional_edges(
            f"agent_{self.name}",
            lambda s: "assistant" if s.mode == "research" else END,
            {
                "assistant": f"assistant_{self.name}",
                END: END
            }
        )
        workflow.add_conditional_edges(f"assistant_{self.name}", should_continue,    {
        # If `tools`, then we call the tool node.
        "continue": f"tools_{self.name}",
        # Otherwise we finish.
            f"agent_{self.name}": f"agent_{self.name}",
        }, 
        )
        workflow.add_edge(f"tools_{self.name}", f"assistant_{self.name}")
        workflow.add_edge(f"assistant_{self.name}", f"agent_{self.name}")  

        return workflow.compile(checkpointer=self.checkpointer, store=self.memory_store)


##########################################################################################
#################################### Level 2 agent #######################################
##########################################################################################


    



class Level2Decision(BaseModel):
    reasoning: str
    decision: Literal["aggregate_for_ceo", "break_down_for_executives"]
    content: List[str] = Field(min_items=1, max_items=1)

class Level2Agent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_schema = self._create_dynamic_state_schema()
        self.prompt_dir = os.path.join(os.path.dirname(__file__), '..', 'prompts/level2', self.name)
        self.jinja_env = Environment(loader=FileSystemLoader(self.prompt_dir))
        self.subordinates = kwargs.get('subordinates', [])
        system_prompt_template = self.jinja_env.get_template('system_prompt.j2')
        self.system_prompt = system_prompt_template.render(tools=self.tools)
        self.system_message = SystemMessage(content=self.system_prompt)
        self.attr_mapping = self._create_attr_mapping()

        self.graph = self._create_graph()

    def _create_dynamic_state_schema(self):
        return create_model(
            f'{self.name}_Level2State',
            **{
                "level1_2_conversation": (Annotated[List, add_messages], ...),
                "level2_3_conversation": (Annotated[List, add_messages], ...),
                f"{self.name}_is_first_execution": (bool, Field(default=True)),
                f"{self.name}_messages": (Annotated[List, add_messages], ...),
                f"{self.name}_mode": (Literal["aggregate_for_ceo", "break_down_for_executives"], Field(default="break_down_for_executives")),
            },
            __base__=BaseModel
        )
    def _create_attr_mapping(self):
        return {
            "mode": f"{self.name}_mode",
            "messages": f"{self.name}_messages",
            "is_first_execution": f"{self.name}_is_first_execution",
        }

    def get_attr(self, state, attr_name):
        return getattr(state, self.attr_mapping.get(attr_name, attr_name))

    def set_attr(self, state, attr_name, value):
        setattr(state, self.attr_mapping.get(attr_name, attr_name), value)


    def supervisor_router(self, state):
        router_name = f"{self.name}_router"

        if getattr(state, f"{self.name}_mode") == "aggregate_for_ceo":
            return "CEO"
        elif getattr(state, f"{self.name}_mode") == "break_down_for_executives":
            return router_name
        else:
            return END

    def _create_graph(self) -> StateGraph:
        def level2_supervisor_node(state: self.state_schema) -> Dict[str, Any]:
            
            if self.get_attr(state, "is_first_execution"):
                self.get_attr(state, "messages").append(self.system_prompt)
                self.set_attr(state, "is_first_execution", False)   
            # Get the last 3 messages from both conversations
            level1_2_last_3 = state.level1_2_conversation[-3:]
            level2_3_last_3 = state.level2_3_conversation[-3:]

            decision_prompt = self.jinja_env.get_template('decision_prompt.j2').render(
                superior_message=level2_3_last_3,
                subordinate_messages=level1_2_last_3,
                subordinates_list=self.subordinates
            )
            
            structured_llm = self.llm.with_structured_output(Level2Decision)
            response = structured_llm.invoke([self.system_message, HumanMessage(content=decision_prompt)])


            if self.debug:
                print(f"Reasoning: {response.reasoning}")
                print(f"Decision: {response.decision}")
                print(f"Content: {response.content}")

            response.content = " ".join(response.content)
            message = self.create_message(content=response.content)
            if response.decision == "aggregate_for_ceo":
                state.level2_3_conversation.append(message)
                setattr(state, f"{self.name}_mode", "aggregate_for_ceo")
            elif response.decision == "break_down_for_executives":
                state.level1_2_conversation.append(message)
                setattr(state, f"{self.name}_mode", "break_down_for_executives")
            
            response = self.create_message(pydantic_to_json(response))
            self.get_attr(state, "messages").append(response)
            return state

        def should_continue(state: self.state_schema) -> Literal["aggregate_for_ceo", "break_down_for_executives", END]:
            if getattr(state, f"{self.name}_mode") == "aggregate_for_ceo":
                return "aggregate_for_ceo"
            elif getattr(state, f"{self.name}_mode") == "break_down_for_executives":
                return "break_down_for_executives"
            else:
                return END

        workflow = StateGraph(self.state_schema)
        workflow.add_node(f"{self.name}_supervisor", level2_supervisor_node)
        workflow.set_entry_point(f"{self.name}_supervisor")
        
        workflow.add_conditional_edges(
            f"{self.name}_supervisor",
            should_continue,
            {
                "aggregate_for_ceo": END,
                "break_down_for_executives": END,
            }
        )

        return workflow.compile(checkpointer=self.checkpointer, store=self.memory_store)


##########################################################################################
#################################### Level 3 agent #######################################
##########################################################################################

class Level3State(BaseModel):
    level2_3_conversation: Annotated[List, add_messages]
    level1_3_conversation: Annotated[List, add_messages]
    company_knowledge: str
    news_insights: List[str]
    digest: List[str]
    action: List[str]
    ceo_messages: Annotated[List, add_messages]
    ceo_assistant_conversation: Annotated[List, add_messages]
    ceo_mode: Literal["research_information", "write_to_digest", "communicate_with_directors", "communicate_with_executives", "end"]
    level2_agents: List[str]
    level1_agents: List[str]
    is_first_execution: bool = True

class CEODecision(BaseModel):
    reasoning: str
    decision: Literal["write_to_digest", "research_information", "communicate_with_directors", "communicate_with_executives", 'end']
    content: List[str]

class Level3Agent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_schema = Level3State
        self.prompt_dir = os.path.join(os.path.dirname(__file__), '..', 'prompts/level3', self.name)
        self.jinja_env = Environment(loader=FileSystemLoader(self.prompt_dir))
        # Generate the system prompt once during initialization
        system_prompt_template = self.jinja_env.get_template('system_prompt.j2')
        self.system_prompt = system_prompt_template.render()
        self.system_message = SystemMessage(content=self.system_prompt)
        self.graph = self._create_graph()

    def _create_graph(self) -> StateGraph:
        def ceo_node(state: self.state_schema) -> Dict[str, Any]:
            decision_prompt = self.jinja_env.get_template('decision_prompt.j2').render(
                news_insights=state.news_insights,
                level2_3_conversation=state.level2_3_conversation,
                level1_3_conversation=state.level1_3_conversation,
                digest=state.digest,
                company_knowledge=state.company_knowledge
            )
            
            if state.is_first_execution:
                state.ceo_messages.append(self.system_message)
                state.is_first_execution = False

            print('debug')

            state.ceo_messages.append(HumanMessage(content=decision_prompt, type = "human", name = self.name))
            structured_llm = self.llm.with_structured_output(CEODecision)
            response = structured_llm.invoke(state.ceo_messages)

            # Convert the list of strings to a single string
            response.content = " ".join(response.content)

            if self.debug:
                print(f"Reasoning: {response.reasoning}")
                print(f"Decision: {response.decision}")
                print(f"Content: {response.content}")
            
            if response.decision == "write_to_digest":
                state.digest.append(response.content)
                state.ceo_mode = "write_to_digest"
            elif response.decision == "research_information":
                state.ceo_assistant_conversation.append(HumanMessage(content=response.content, type = "human"))
                state.ceo_mode = "research_information"
            elif response.decision == "communicate_with_directors":
                state.level2_3_conversation.append(self.create_message(response.content, type = "human"))
                state.ceo_mode = "communicate_with_directors"
            elif response.decision == "communicate_with_executives":
                state.level1_3_conversation.append(self.create_message(response.content, type = "human"))
                state.ceo_mode = "communicate_with_executives"
            
            elif response.decision == "end":
                state.ceo_mode = "end"
            response = HumanMessage(content=pydantic_to_json(response), type = "human")

            #response = self.create_message(pydantic_to_json(response))
            state.ceo_messages.append(response)

            return state

        def assistant_node(state: self.state_schema) -> Dict[str, Any]:
            prompt = self.jinja_env.get_template('assistant_prompt.j2')
            last_message = state.ceo_assistant_conversation[-1]
            
            response = self.assistant_llm.invoke(self.create_message(content=prompt.render(
                question=last_message.content,
                company_knowledge=state.company_knowledge,
                digest=state.digest
            )))
            
            state.ceo_assistant_conversation.append(AIMessage(content=response))
        
            return state

        def should_continue(state: Level3State) -> Literal["assistant", "ceo", "directors", "executives", END]:
            if state.ceo_mode == "research_information":
                return "assistant"
            elif state.ceo_mode == "write_to_digest":
                return "ceo"
            elif state.ceo_mode == "communicate_with_directors":
                return "directors"
            elif state.ceo_mode == "communicate_with_executives":
                return "executives"
            elif state.ceo_mode == "end":
                return END
            else:
                raise ValueError(f"Unexpected CEO mode: {state.ceo_mode}")
                
        workflow = StateGraph(self.state_schema)
        workflow.add_node("ceo", ceo_node)
        workflow.add_node("ceo_assistant", assistant_node)

        workflow.set_entry_point("ceo")
        
        # Add conditional edges based on the should_continue function
        workflow.add_conditional_edges(
            "ceo",
            should_continue,
            {
                "assistant": "ceo_assistant",
                "ceo": "ceo",
                "directors": END,  # We'll handle this in the main graph
                "executives": END,  # We'll handle this in the main graph
                END: END
            }
        )
        workflow.add_edge("ceo_assistant", "ceo")

        return workflow.compile()

def create_agents_graph():
    load_dotenv()

    # Common configuration
    tools = [DuckDuckGoSearchRun()]
    checkpointer = MemorySaver()
    memory_store = InMemoryStore()
    debug = True

    # LLM configurations
    llm_config = {
        "model": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 1000
    }
    assistant_llm_config = {
        "model": "gpt-4",
        "temperature": 0.5,
        "max_tokens": 1000
    }

    level2_subordinates = {
        "supervisor1": ["agent1", "agent2"],
        "supervisor2": ["agent3", "agent4"],
    }

    # Function to get agent names from folder structure
    def get_agent_names(level):
        prompts_dir = os.path.join(os.path.dirname(__file__), '..', 'prompts', f'level{level}')
        return [name for name in os.listdir(prompts_dir) if os.path.isdir(os.path.join(prompts_dir, name))]

    # Create Level 3 agent (CEO)
    ceo_name = get_agent_names(3)[0]  # Assuming there's only one CEO
    ceo_agent = Level3Agent(
        name=ceo_name,
        llm="gpt-4",
        llm_params=llm_config,
        assistant_llm="gpt-4",
        assistant_llm_params=assistant_llm_config,
        tools=tools,  # Make sure to pass the tools here
        system_message="You are the CEO of the company.",
        max_iterations=15,
        checkpointer=checkpointer,
        memory_store=memory_store,
        debug=debug
    )

    # Create Level 2 agents
    level2_agents = []
    for name in get_agent_names(2):
        level2_agent = Level2Agent(
            name=name,
            llm="gpt-3.5-turbo",
            llm_params=llm_config,
            assistant_llm="gpt-4",
            assistant_llm_params=assistant_llm_config,
            tools=tools,
            system_message=f"You are a director and your name is {name}.",
            max_iterations=10,
            checkpointer=checkpointer,
            memory_store=memory_store,
            debug=debug,
            subordinates=level2_subordinates.get(name, [])  # Assign subordinates based on the dictionary
        )
        level2_agents.append(level2_agent)

    # Create Level 1 agents
    level1_agents = []
    for name in get_agent_names(1):
        level1_agent = Level1Agent(
            name=name,
            llm="gpt-3.5-turbo",
            llm_params=llm_config,
            assistant_llm="gpt-3.5-turbo",
            assistant_llm_params=assistant_llm_config,
            tools=tools,
            system_message=f"You are an executive in charge of {name}.",
            max_iterations=5,
            checkpointer=checkpointer,
            memory_store=memory_store,
            debug=debug
        )
        level1_agents.append(level1_agent)

    # Create subgraphs for Level 2 agents and their subordinates
    level2_subgraphs = {}
    for l2_agent in level2_agents:
        subgraph = StateGraph(l2_agent.state_schema)
        # Add Level 2 agent node
        subgraph.add_node(l2_agent.name, l2_agent.graph)
        
        
        subordinates = [l1_agent.name for l1_agent in level1_agents if l1_agent.name in l2_agent.subordinates]
        # Add Level 1 subordinate nodes
        for l1_agent in level1_agents:
            if l1_agent.name in l2_agent.subordinates:
                subgraph.add_node(l1_agent.name, l1_agent.graph)
        
        # Set entry point to Level 2 agent
        subgraph.set_entry_point(l2_agent.name)
        
        # Add router node
        router_name = f"{l2_agent.name}_router"
        
        def level2_router(state):
            # This function will be called when the router node is executed
            # It should return the names of all subordinates to execute them in parallel
            return state

        subgraph.add_node(router_name, level2_router)
        


        # Add conditional edges from Level 2 agent
        subgraph.add_conditional_edges(
            l2_agent.name,
            l2_agent.supervisor_router,
            {
                "CEO": END,
                router_name: router_name,
                END: END
            }
        )
        
        # Add edges from router to subordinates
        for subordinate in subordinates:
            subgraph.add_edge(router_name, subordinate)
        
        # Add edges from subordinates back to Level 2 agent
        for subordinate in subordinates:
            subgraph.add_edge(subordinate, l2_agent.name)
        
        # Compile the subgraph
        level2_subgraphs[l2_agent.name] = subgraph.compile()
        #mermaid_png = level2_subgraphs[l2_agent.name].get_graph().draw_mermaid_png()
        #img = Image.open(io.BytesIO(mermaid_png))
        #img.save(f'level2_agent_{l2_agent.name}_graph.png')


    # Create main graph using Level3State
    main_graph = StateGraph(Level3State)

    # Add CEO (Level 3) node
    main_graph.add_node("CEO", ceo_agent.graph)

 
   # Add ceo router node
    ceo_router_name = "ceo_router"
        
    def ceo_router_node(state):
        # This function will be called when the router node is executed
        # It should return the names of all subordinates to execute them in parallel
        return state

    main_graph.add_node(ceo_router_name, ceo_router_node)
    
#add created subgraphs as nodes in the main graph
    for l2_agent in level2_agents:
        main_graph.add_node(l2_agent.name, level2_subgraphs[l2_agent.name]) 

    def ceo_router(state: Level3State):
        if getattr(state, f"ceo_mode") == "communicate_with_directors":
            return ceo_router_name
        elif getattr(state, f"ceo_mode") == "communicate_with_executives":
            return ceo_router_name
        else:
            return END

    # Add conditional edges from CEO to Level 2 subgraphs
    main_graph.add_conditional_edges(
        "CEO",
        ceo_router,
        {
            ceo_router_name : ceo_router_name,
            END: END
        }
    )
    for l2_agent in level2_agents:
        main_graph.add_edge(ceo_router_name, l2_agent.name)
        main_graph.add_edge(l2_agent.name, "CEO")


    # Set the entry point
    main_graph.set_entry_point("CEO")

 
            
    # Compile the main graph
    final_graph = main_graph.compile()
    #mermaid_png = final_graph.get_graph().draw_mermaid_png()
    # img = Image.open(io.BytesIO(mermaid_png))
    #img.save(f'level3_agent_graph.png')
    return final_graph

if __name__ == "__main__":

    final_graph = create_agents_graph()
    # Now let's invoke the graph with some fake entry state
    initial_state = Level3State(
        level2_3_conversation=[],
        level1_3_conversation=[],
        company_knowledge="Our company is a tech startup focusing on AI solutions.",
        news_insights=["AI market is growing rapidly", "New regulations on data privacy"],
        digest=[],
        action=[],
        ceo_messages=[HumanMessage(content="What's our strategy for the next quarter?", type = "human")],
        ceo_assistant_conversation=[],
        ceo_mode="research_information",
        level2_agents=["Director1", "Director2"],
        level1_agents=["Agent1", "Agent2", "Agent3", "Agent4"],
    )
    
       # Helper function for formatting the stream nicely
    def print_stream(stream):
        for s in stream:
            print(s)

    # Invoke the graph with the initial state
    config = {"configurable": {"thread_id": "test_thread2"}}

    print_stream(final_graph.stream(initial_state, config, stream_mode="values"))
    
    
    
    
    
    
    
    
    
    
    
    
    #result = final_graph.invoke(initial_state)

    # Print the result
    #print("Graph execution result:", result)



