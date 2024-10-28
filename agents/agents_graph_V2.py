from typing import List, Optional, Dict, Any, Union, Callable, Sequence, Literal, Type, get_origin, get_args
from typing_extensions import Annotated, TypedDict
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, create_model, Field
from pydantic import BaseModel, Field
from langchain.schema.runnable import Runnable
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from PIL import Image
import io
import os
import operator
from langgraph.graph.message import add_messages
from langchain_community.tools import DuckDuckGoSearchRun
from jinja2 import Environment, FileSystemLoader
from langchain_openai import ChatOpenAI
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from langchain_core.messages import trim_messages
from langgraph.graph import START, MessagesState, StateGraph
from datetime import datetime
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
import json
import logging
import time  # Add this at the top with other imports
try:
    from .agent_base import BaseAgent
except:
    from agent_base import BaseAgent    


# Set the logging level for the SageMaker SDK to WARNING or higher
def setup_logging():
    log_filename = f"logs/agent_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=log_filename,
        filemode='w'
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger('').addHandler(console_handler)
    return logging.getLogger(__name__)

load_dotenv()


#set_llm_cache(InMemoryCache())

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
    content: Union[str, List[str], dict] = Field(default_factory=dict)

    def get_content_as_string(self) -> str:
        """Convert content to string regardless of input type"""
        if isinstance(self.content, dict):
            return json.dumps(self.content)
        elif isinstance(self.content, list):
            return " ".join(str(item) for item in self.content)
        return str(self.content)



# Define trimmer
# count each message as 1 "token" (token_counter=len) and keep only the last two messages
trimmer = trim_messages(strategy="last", max_tokens=5000, token_counter=ChatOpenAI(model="gpt-4o"), allow_partial=True)

def keep_last_n(existing: List, updates: List) -> List:
    """Keep only the last n items from the combined list."""
    combined = existing + updates
    return combined[-5:]

def keep_last_item(existing: List, updates: List) -> List:
    """Keep only the last item from the combined list."""
    combined = existing + updates
    return [combined[-1]] if combined else []
def keep_last_elem(existing, updates) -> List:
    return updates

class Level1Agent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.supervisor_name = kwargs.get('supervisor_name')

        self.state_schema = self._create_dynamic_state_schema()
        self.attr_mapping = self._create_attr_mapping()
        self.prompt_dir = os.path.join(kwargs.get('prompt_dir', ''), 'level1', self.name)
        self.jinja_env = Environment(loader=FileSystemLoader(self.prompt_dir))
        system_prompt_template = self.jinja_env.get_template('system_prompt.j2')
        self.system_prompt = system_prompt_template.render(tools=self.tools)
        self.system_message = SystemMessage(content=self.system_prompt)
        self.trimmer = trimmer
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{self.name}")

    def level1_node(self, state):
        self.logger.info(f"Executing level1_node for {self.name}")
        
        if not self.get_attr(state, "messages"):
            self.get_attr(state, "messages").append(self.system_message)
            
        # Initialize last_messages with default values
        last_messages = {
            "level1_2_conversation": "No messages from level 2 yet.",
            "level1_3_conversation": "No messages from level 3 yet."
        }
        
        # Safely get last messages if conversations exist and have messages
        if self.get_attr(state, f"{self.supervisor_name}_level1_2_conversation"):
            if len(self.get_attr(state, f"{self.supervisor_name}_level1_2_conversation")) > 0:
                last_messages["level1_2_conversation"] = self.get_attr(state, f"{self.supervisor_name}_level1_2_conversation")[-min(3, len(self.get_attr(state, f"{self.supervisor_name}_level1_2_conversation")))].content
        
        if self.get_attr(state, "level1_3_conversation"):
            if len(self.get_attr(state, "level1_3_conversation")) > 0:
                last_messages["level1_3_conversation"] = self.get_attr(state, "level1_3_conversation")[-min(3, len(self.get_attr(state, "level1_3_conversation")))].content
        
        if self.debug:
            print(f"Last message: {last_messages}")

        # Initialize empty lists for conversations if they don't exist
        if not self.get_attr(state, "level1_2_conversation"):
            self.get_attr(state, "level1_2_conversation").append(HumanMessage(
                content="Starting meeting between supervisor and executives. The executives will report on their findings and receive guidance from their supervisor."
            ))
        
        if not self.get_attr(state, "level1_3_conversation"):
            self.get_attr(state, "level1_3_conversation").append(HumanMessage(
                content="Starting meeting between CEO and executives. The executives will provide insights and receive strategic direction from the CEO."
            ))
        
        if not self.get_attr(state, "assistant_conversation"):
            self.get_attr(state, "assistant_conversation").append(HumanMessage(
                content="Starting research session between executive and assistant. The assistant will help gather information and analyze data to support executive decision-making."
            ))

        # Trim messages before rendering the decision prompt
        trimmed_level1_2_conversation = self.trimmer.invoke(self.get_attr(state, "level1_2_conversation"))
        trimmed_level1_3_conversation = self.trimmer.invoke(self.get_attr(state, "level1_3_conversation"))
        trimmed_assistant_conversation = self.trimmer.invoke(self.get_attr(state, "assistant_conversation"))

        decision_prompt = self.jinja_env.get_template('decision_prompt.j2').render(
            last_message=last_messages,
            level1_2_conversation=trimmed_level1_2_conversation,
            level1_3_conversation=trimmed_level1_3_conversation,
            assistant_conversation=trimmed_assistant_conversation,
            tools=self.tools
        )
        
        if state.ceo_runs_counter < 2:
            self.get_attr(state, "messages").append(self.system_prompt)

        # Use the decision prompt
        message = HumanMessage(content=decision_prompt)
        self.get_attr(state, "messages").append(message)
        trimmed_message = self.trimmer.invoke(self.get_attr(state, "messages"))
        structured_llm = self.llm.with_structured_output(Level1Decision)
        response = structured_llm.invoke(trimmed_message)

        if self.debug:
            print(f"Reasoning: {response.reasoning}")
            print(f"Decision: {response.decision}")
            print(f"Content: {response.content}")

        # Use the content_as_string method to safely convert any content type
        content_str = response.get_content_as_string()
        message = self.create_message(content=content_str)
        resp = self.create_message(content=pydantic_to_json(response))

        if response.decision == "search_more_information":
            return { f"{self.name}_assistant_conversation": [message],
                     f"{self.name}_mode": ["research"],
                     f"{self.name}_messages": [resp]
                    }
        else:
            return { f"{self.supervisor_name}_level1_2_conversation": [message],
                     f"{self.name}_mode": ["converse"],
                     f"{self.name}_messages": [resp]
            }

    def assistant_node(self, state) -> Dict[str, Any]:
        self.logger.info(f"Executing assistant_node for {self.name}")
        
        prompt = self.jinja_env.get_template('assistant_prompt.j2')

        # Safety check: ensure assistant_conversation exists and has messages
        assistant_conversation = self.get_attr(state, "assistant_conversation")
        if not assistant_conversation:
            # Initialize with a default message if empty
            assistant_conversation.append(HumanMessage(
                content="Starting research session. As your assistant, I'm here to help gather information, analyze data, and provide insights to support your decision-making process. What specific information would you like me to research?"
            ))
            return {f"{self.name}_assistant_conversation": [assistant_conversation[-1]]}

        try:
            last_message = assistant_conversation[-1]
            print(f"Processing question from {self.name}: {last_message.content}")
            assistant_message = self.create_message(content=prompt.render(question=last_message.content))
            
            try:
                response = self.assistant_llm.invoke(assistant_message)
            except Exception as e:
                self.logger.warning(f"Error invoking assistant_llm with message: {e}")
                response = self.assistant_llm.invoke(f"Question from executive: {last_message.content}.")
                self.logger.info(f"assistant answer: {response}")

            response = self.create_message(pydantic_to_json(response), agent_name=f"assistant_{self.name}")
            
            return {f"{self.name}_assistant_conversation": [response]}
        
        except Exception as e:
            self.logger.error(f"Error in assistant_node: {e}")
            # Return a safe default response
            default_response = self.create_message("I apologize, but I encountered an error processing the request.", 
                                                 agent_name=f"assistant_{self.name}")
            return {f"{self.name}_assistant_conversation": [default_response]}

    def should_continue(self, state):
        try:
            assistant_conversation = self.get_attr(state, "assistant_conversation")
            if not assistant_conversation:
                return "executive_agent"
            
            last_message = assistant_conversation[-1]
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "continue"
            else:
                return "executive_agent"
        except Exception as e:
            self.logger.error(f"Error in should_continue: {e}")
            return "executive_agent"

        
    def _create_dynamic_state_schema(self):
        return create_model(
            f'{self.name}_Level1State',
            **{
                f"{self.supervisor_name}_level1_2_conversation": (Annotated[List, add_messages], Field(default_factory=list)),
                f"level1_3_conversation": (Annotated[List, add_messages], Field(default_factory=list)),
                f"{self.name}_assistant_conversation": (Annotated[List, add_messages], Field(default_factory=list)),
                f"{self.name}_domain_knowledge": (Annotated[List[str], operator.add], Field(default_factory=lambda: [])),
                f"{self.name}_mode": (Annotated[List[Literal["research", "converse"]], operator.add], Field(default_factory=lambda: ["research"])),
                f"{self.name}_messages": (Annotated[List, add_messages], Field(default_factory=list)),
            },
            __base__=BaseModel
        )

    def _create_attr_mapping(self):
        return {
            "assistant_conversation": f"{self.name}_assistant_conversation",
            "domain_knowledge": f"{self.name}_domain_knowledge",
            "mode": f"{self.name}_mode",
            "messages": f"{self.name}_messages",
            "level1_2_conversation": f"{self.supervisor_name}_level1_2_conversation",
        }

    def get_attr(self, state, attr_name):
        return getattr(state, self.attr_mapping.get(attr_name, attr_name))

    def set_attr(self, state, attr_name, value):
        setattr(state, self.attr_mapping.get(attr_name, attr_name), value)

        
    



##########################################################################################
#################################### Level 2 agent #######################################
##########################################################################################


    



class Level2Decision(BaseModel):
    reasoning: str
    decision: Literal["aggregate_for_ceo", "break_down_for_executives"]
    content: Union[List[str], str] = Field(min_items=1)

class Level2Agent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_schema = self._create_dynamic_state_schema()
        self.prompt_dir = os.path.join(kwargs.get('prompt_dir', ''), 'level2', self.name)
        self.jinja_env = Environment(loader=FileSystemLoader(self.prompt_dir))
        self.subordinates = kwargs.get('subordinates', [])
        system_prompt_template = self.jinja_env.get_template('system_prompt.j2')
        self.system_prompt = system_prompt_template.render(tools=self.tools)
        self.system_message = SystemMessage(content=self.system_prompt)
        self.attr_mapping = self._create_attr_mapping()
        self.trimmer = trimmer
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{self.name}")
    

    def level2_supervisor_node(self, state):
        if not state.ceo_runs_counter > 1:
            self.get_attr(state, "messages").append(self.system_prompt)

        # Initialize conversations if they don't exist
        if not self.get_attr(state, "level1_2_conversation"):
            self.get_attr(state, "level1_2_conversation").append(HumanMessage(
                content=f"Starting supervisory session for {self.name}. As a director, I will coordinate between executives and CEO, ensuring departmental alignment and strategic implementation. Current focus is on reviewing reports from {', '.join(self.subordinates)}."
            ))

        if not self.get_attr(state, "level2_3_conversation"):
            self.get_attr(state, "level2_3_conversation").append(HumanMessage(
                content=f"Starting director-CEO alignment session for {self.name}. Ready to present consolidated insights from departments {', '.join(self.subordinates)} and receive strategic guidance."
            ))

        # Safely get the last 3 messages
        def get_last_n_messages(messages, n=3):
            if not messages:
                return []
            return messages[-min(n, len(messages)):]

        level1_2_last_3 = get_last_n_messages(self.get_attr(state, "level1_2_conversation"))
        level2_3_last_3 = get_last_n_messages(self.get_attr(state, "level2_3_conversation"))

        # Only trim if we have messages
        trimmed_level1_2_last_3 = self.trimmer.invoke(level1_2_last_3) if level1_2_last_3 else []
        trimmed_level2_3_last_3 = self.trimmer.invoke(level2_3_last_3) if level2_3_last_3 else []

        decision_prompt = self.jinja_env.get_template('decision_prompt.j2').render(
            superior_message=trimmed_level2_3_last_3,
            subordinate_messages=trimmed_level1_2_last_3,
            subordinates_list=self.subordinates
        )
        
        structured_llm = self.llm.with_structured_output(Level2Decision)
        response = structured_llm.invoke([self.system_message, HumanMessage(content=decision_prompt)])


        if self.debug:
            print(f"Reasoning: {response.reasoning}")
            print(f"Decision: {response.decision}")
            print(f"Content: {response.content}")

        response.content = " ".join(response.content)
        message = self.create_message(content=pydantic_to_json(response))

        if response.decision == "aggregate_for_ceo":

            
            return { f"level2_3_conversation": [message],
                        f"{self.name}_mode": ["aggregate_for_ceo"],
                        f"{self.name}_messages": [message]
            }
        
        elif response.decision == "break_down_for_executives":

            return { f"{self.name}_level1_2_conversation": [message],
                        f"{self.name}_mode": ["break_down_for_executives"],
                        f"{self.name}_messages": [message]
            }
            


    def should_continue(self, state) -> Literal["aggregate_for_ceo", "break_down_for_executives"]:
        current_mode = self.get_attr(state, "mode")
        if len(current_mode) < 1:
            return "break_down_for_executives"
        else:
            if current_mode[-1] == "aggregate_for_ceo" :
                return "aggregate_for_ceo"
            else:
                return "break_down_for_executives"


    def _create_dynamic_state_schema(self):
        return create_model(
            f'{self.name}_Level2State',
            **{
                f"{self.name}_level1_2_conversation": (Annotated[List, add_messages], ...),
                f"level2_3_conversation": (Annotated[List, add_messages], ...),
                f"{self.name}_messages": (Annotated[List, add_messages], ...),
                f"{self.name}_mode": (Annotated[List[Literal["aggregate_for_ceo", "break_down_for_executives"]], operator.add], Field(default_factory=lambda: ["break_down_for_executives"])),
            },
            __base__=BaseModel
        )
    def _create_attr_mapping(self):
        return {
            "mode": f"{self.name}_mode",
            "messages": f"{self.name}_messages",
            "level1_2_conversation": f"{self.name}_level1_2_conversation",
        }
        
    def get_attr(self, state, attr_name):
        return getattr(state, self.attr_mapping.get(attr_name, attr_name))

    def set_attr(self, state, attr_name, value):
        setattr(state, self.attr_mapping.get(attr_name, attr_name), value)


     




##########################################################################################
#################################### Level 3 agent #######################################
##########################################################################################

class Level3State(BaseModel):
    level2_3_conversation: Annotated[List, add_messages]
    level1_3_conversation: Annotated[List, add_messages]
    company_knowledge: Annotated[List[str], operator.add, Field(default_factory=lambda: [])]
    news_insights: Annotated[List[str], operator.add, Field(default_factory=lambda: [])]
    digest: Annotated[List[str], operator.add, Field(default_factory=lambda: [])]
    ceo_messages: Annotated[List, add_messages]
    ceo_assistant_conversation: Annotated[List, add_messages]
    ceo_mode: Annotated[List[Literal["research_information", "write_to_digest", "communicate_with_directors", "communicate_with_executives", "end"]], operator.add, Field(default_factory=lambda: ["communicate_with_executives"])]
    ceo_runs_counter: Annotated[int, operator.add, Field(default=0)]

class CEODecision(BaseModel):
    reasoning: str
    decision: Literal["write_to_digest", "research_information", "communicate_with_directors", "communicate_with_executives", 'end']
    content: Union[List[str], str] = Field(min_items=1)

class Level3Agent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_schema = Level3State
        self.prompt_dir = os.path.join(kwargs.get('prompt_dir', ''), 'level3', self.name)
        self.jinja_env = Environment(loader=FileSystemLoader(self.prompt_dir))
        # Generate the system prompt once during initialization
        system_prompt_template = self.jinja_env.get_template('system_prompt.j2')
        self.system_prompt = system_prompt_template.render()
        self.system_message = SystemMessage(content=self.system_prompt)
        self.trimmer = trimmer
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{self.name}")

    def ceo_node(self, state) -> Dict[str, Any]:
        state.ceo_runs_counter += 1

        trimmed_level2_3_conversation = self.trimmer.invoke(state.level2_3_conversation)
        trimmed_level1_3_conversation = self.trimmer.invoke(state.level1_3_conversation)
        trimmed_ceo_assistant_conversation = self.trimmer.invoke(state.ceo_assistant_conversation)

        decision_prompt = self.jinja_env.get_template('decision_prompt.j2').render(
            news_insights=state.news_insights,
            level2_3_conversation=state.level2_3_conversation,
            level1_3_conversation=state.level1_3_conversation,
            digest=state.digest,
            company_knowledge=state.company_knowledge
        )
        
        if not state.ceo_runs_counter > 1:
            state.ceo_messages.append(self.system_message)

        state.ceo_messages.append(HumanMessage(content=decision_prompt, type="human", name=self.name))
        trimmed_ceo_messages = self.trimmer.invoke(state.ceo_messages)
        structured_llm = self.llm.with_structured_output(CEODecision)
        response = structured_llm.invoke(trimmed_ceo_messages)

        # Convert the list of strings to a single string
        response.content = " ".join(response.content)

        if self.debug:
            print(f"Reasoning: {response.reasoning}")
            print(f"Decision: {response.decision}")
            print(f"Content: {response.content}")
        
        if response.decision == "write_to_digest":
            return { f"digest": [response.content],
                     f"ceo_mode": ["write_to_digest"],
                     f"ceo_messages": [HumanMessage(content=pydantic_to_json(response), type="human")]
            }
        elif response.decision == "research_information":
            return { f"ceo_assistant_conversation": [HumanMessage(content=response.content, type="human")],
                     f"ceo_mode": ["research_information"],
                     f"ceo_messages": [HumanMessage(content=pydantic_to_json(response), type="human")]
            }
        elif response.decision == "communicate_with_directors":
            return { f"level2_3_conversation": [self.create_message(pydantic_to_json(response), type="human")],
                     f"ceo_mode": ["communicate_with_directors"],
                     f"ceo_messages": [HumanMessage(content=pydantic_to_json(response), type="human")]
            }
        elif response.decision == "communicate_with_executives":
            return { f"level1_3_conversation": [self.create_message(pydantic_to_json(response), type="human")],
                     f"ceo_mode": ["communicate_with_executives"],
                     f"ceo_messages": [HumanMessage(content=pydantic_to_json(response), type="human")]
            }
        elif response.decision == "end":
            return { f"ceo_mode": ["end"],
                     f"ceo_messages": [HumanMessage(content=pydantic_to_json(response), type="human")]
            }
        

    def assistant_node(self, state) -> Dict[str, Any]:
        try:
            # Initialize conversation if empty
            if not state.ceo_assistant_conversation:
                state.ceo_assistant_conversation.append(HumanMessage(
                    content="Starting CEO advisory session. As your executive assistant, I'm here to help research market trends, analyze company data, and provide strategic insights. I have access to company knowledge and can help synthesize information for decision-making. What would you like me to analyze first?"
                ))
                return {"ceo_assistant_conversation": [state.ceo_assistant_conversation[-1]]}

            prompt = self.jinja_env.get_template('assistant_prompt.j2')
            last_message = state.ceo_assistant_conversation[-1]
            
            # Create and render the prompt
            prompt_content = prompt.render(
                question=last_message.content,
                company_knowledge=state.company_knowledge,
                digest=state.digest
            )
            
            # Invoke the assistant
            try:
                response = self.assistant_llm.invoke(self.create_message(content=prompt_content))
            except Exception as e:
                self.logger.warning(f"Error invoking assistant_llm: {e}")
                response = "I apologize, but I encountered an error processing your request."
            
            return {"ceo_assistant_conversation": [AIMessage(content=response)]}
            
        except Exception as e:
            self.logger.error(f"Error in assistant_node: {e}")
            return {"ceo_assistant_conversation": [AIMessage(content="I apologize, but I encountered an error.")]}

    def should_continue(self, state) -> Literal["assistant", "ceo", "directors", "executives", END]:
        current_mode = state.ceo_mode[-1] if state.ceo_mode else "research_information"
        if current_mode == "research_information" :
            return "assistant"
        elif current_mode == "write_to_digest" :
            return "ceo"
        elif current_mode == "communicate_with_directors":
            return "directors"
        elif current_mode == "communicate_with_executives" :
            return "executives"
        else:
            return END

    def should_continue_assistant(self, state):
        try:
            # Check if conversation exists and has messages
            if not state.ceo_assistant_conversation:
                return "ceo"
                
            last_message = state.ceo_assistant_conversation[-1]
            
            # Safely check for tool_calls
            has_tool_calls = False
            if hasattr(last_message, 'tool_calls'):
                has_tool_calls = bool(last_message.tool_calls)
            elif hasattr(last_message, 'content') and hasattr(last_message.content, 'tool_calls'):
                has_tool_calls = bool(last_message.content.tool_calls)
                
            return "continue" if has_tool_calls else "ceo"
            
        except Exception as e:
            self.logger.error(f"Error in should_continue_assistant: {e}")
            return "ceo"


##########################################################################################
#################################### Unified state #####################################
##################################### Final Graph ######################################
##########################################################################################

class StateMachines():
    def __init__(self, prompt_dir, interrupt_graph_before = True):
        self.logger = logging.getLogger(__name__)
        self.interrupt_graph_before = interrupt_graph_before
        
        def from_conn_stringx(cls, conn_string: str,) -> "SqliteSaver":
            return SqliteSaver(conn=sqlite3.connect(conn_string, check_same_thread=False))
        SqliteSaver.from_conn_stringx=classmethod(from_conn_stringx)

        from dotenv import load_dotenv
        load_dotenv()
        self.memory = SqliteSaver.from_conn_stringx(":memory:")
        self.prompt_dir = prompt_dir
        self.final_graph , self.unified_state_schema = self._create_agents_graph()
        self.config = {
            "recursion_limit": 50, 
            "configurable":{
                "thread_id": "2",
            }
        }

    def _create_unified_state_schema(self, level1_agents, level2_agents, ceo_agent):
        unified_fields = {
            "level2_3_conversation": (
                Annotated[List, add_messages], 
                Field(default_factory=lambda: [HumanMessage(
                    content="Starting strategic alignment meeting between directors and CEO. Directors will consolidate department reports and receive strategic guidance."
                )])
            ),
            "level1_3_conversation": (
                Annotated[List, add_messages], 
                Field(default_factory=lambda: [HumanMessage(
                    content="Starting executive briefing with CEO. Executives will share departmental insights and receive strategic direction."
                )])
            ),
            "ceo_messages": (
                Annotated[List, keep_last_n], 
                Field(default_factory=lambda: [HumanMessage(
                    content="Initiating CEO strategic review session. Focus areas include market analysis, organizational alignment, and strategic decision-making."
                )])
            ),
            "ceo_assistant_conversation": (
                Annotated[List, keep_last_n], 
                Field(default_factory=lambda: [HumanMessage(
                    content="Starting CEO advisory session. Ready to assist with market research, data analysis, and strategic insights compilation."
                )])
            ),
            "ceo_mode": (
                Annotated[List[Literal["research_information", "write_to_digest", "communicate_with_directors", "communicate_with_executives", "end"]], keep_last_item],
                Field(default_factory=lambda: ["research_information"])
            ),
            "company_knowledge": (
                Annotated[List[str], operator.add],
                Field(default_factory=list)
            ),
            "news_insights": (
                Annotated[List[str], keep_last_item],
                Field(default_factory=list)
            ),
            "digest": (
                Annotated[List[str], operator.add],
                Field(default_factory=list)
            ),
            "ceo_runs_counter": (
                Annotated[int, keep_last_elem],
                Field(default=0)
            ),
            "empty_channel": (
                Annotated[int, keep_last_elem],
                Field(default_factory=lambda: [])
            )
        }

        # Add default modes for all agents
        for agent in level1_agents:
            unified_fields[f"{agent.name}_mode"] = (
                Annotated[List[Literal["research", "converse"]], keep_last_item],
                Field(default_factory=lambda: ["research"])
            )
            unified_fields[f"{agent.name}_assistant_conversation"] = (
                Annotated[List, keep_last_n],
                Field(default_factory=lambda: [HumanMessage(
                    content=f"Starting research session for {agent.name}. Ready to assist with information gathering and analysis to support executive decision-making."
                )])
            )
            unified_fields[f"{agent.name}_messages"] = (
                Annotated[List, keep_last_n],
                Field(default_factory=lambda: [HumanMessage(content="")])
            )
            unified_fields[f"{agent.name}_domain_knowledge"] = (
                Annotated[List[str], operator.add],
                Field(default_factory=list)
            )

        for agent in level2_agents:
            unified_fields[f"{agent.name}_mode"] = (
                Annotated[List[Literal["aggregate_for_ceo", "break_down_for_executives"]], keep_last_item],
                Field(default_factory=lambda: ["break_down_for_executives"])
            )
            unified_fields[f"{agent.name}_messages"] = (
                Annotated[List, keep_last_n],
                Field(default_factory=lambda: [HumanMessage(content="")])
            )
            unified_fields[f"{agent.name}_level1_2_conversation"] = (
                Annotated[List, add_messages],
                Field(default_factory=lambda: [HumanMessage(
                    content=f"Starting departmental coordination meeting. Executives will report to their directors {agent.name} for guidance and alignment."
                )])
            )

        UnifiedState = create_model("UnifiedState", **unified_fields, __base__=BaseModel)
        return UnifiedState

    def _get_agent_names(self, level):
        base_path = os.path.join(self.prompt_dir, f'level{level}')
        return [name for name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, name))]

    def _load_agent_config(self, level, agent_name):
        config_path = os.path.join(self.prompt_dir, f'level{level}', agent_name, 'config.json')
        with open(config_path, 'r') as f:
            return json.load(f)

    def _create_agents_graph(self):
        # Common configuration
        tools = []
        debug = False

        # Create Level 3 agent (CEO)
        ceo_name = self._get_agent_names(3)[0]  # Assuming there's only one CEO
        ceo_config = self._load_agent_config(3, ceo_name)
        ceo_agent = Level3Agent(
            name=ceo_name,
            llm=ceo_config['llm_model'],
            llm_params=ceo_config['llm_config'],
            assistant_llm=ceo_config['assistant_llm_model'],
            assistant_llm_params=ceo_config['assistant_llm_config'],
            tools=tools,
            debug=debug,
            prompt_dir=self.prompt_dir
        )

        # Create Level 2 agents
        level2_agents = []
        for name in self._get_agent_names(2):
            level2_config = self._load_agent_config(2, name)
            level2_agent = Level2Agent(
                name=name,
                llm=level2_config['llm_model'],
                llm_params=level2_config['llm_config'],
                assistant_llm=level2_config['assistant_llm_model'],
                assistant_llm_params=level2_config['assistant_llm_config'],
                tools=tools,
                debug=debug,
                subordinates=level2_config.get('subordinates', []),
                prompt_dir=self.prompt_dir
            )
            level2_agents.append(level2_agent)

        # Create Level 1 agents
        level1_agents = []
        for name in self._get_agent_names(1):
            level1_config = self._load_agent_config(1, name)
            level1_agent = Level1Agent(
                name=name,
                llm=level1_config['llm_model'],
                llm_params=level1_config['llm_config'],
                assistant_llm=level1_config['assistant_llm_model'],
                assistant_llm_params=level1_config['assistant_llm_config'],
                tools=tools,
                debug=debug,
                supervisor_name=level1_config.get('supervisor_name', ''),
                prompt_dir=self.prompt_dir
            )
            level1_agents.append(level1_agent)

        # After creating all your agents, use this function to create the unified state schema
        unified_state_schema = self._create_unified_state_schema(level1_agents, level2_agents, ceo_agent)

        workflow = StateGraph(unified_state_schema)
        workflow.add_node("ceo", ceo_agent.ceo_node)
        workflow.add_node("ceo_assistant", ceo_agent.assistant_node)
        workflow.add_node("ceo_tool", ToolNode)
        workflow.set_entry_point("ceo")

        def create_ceo_router_up(level2_agents):
            def ceo_router_up(state):
                return "complete" if all([getattr(state, f"{l2_agent.name}_mode")[-1] == "aggregate_for_ceo" for l2_agent in level2_agents]) else "continue"
            return ceo_router_up
        
        def ceo_router_up_node(state):
            logging.info("CEO Router Up Node - Processing")
            time.sleep(5)  # Add 2-second delay
            return {"empty_channel": 1}
        
        workflow.add_node("ceo_router_up", ceo_router_up_node)
        
        def ceo_router_down(state):
            logging.info("CEO Router Down - Processing")
            time.sleep(2)  # Add 2-second delay
            return {"empty_channel": 1}

        workflow.add_node("ceo_router_down" , ceo_router_down  )

        for l1_agent in level1_agents:

            tool_node = ToolNode(l1_agent.tools)
            workflow.add_node(f"agent_{l1_agent.name}", l1_agent.level1_node)
            workflow.add_node(f"assistant_{l1_agent.name}", l1_agent.assistant_node)
            workflow.add_node(f"tools_{l1_agent.name}", tool_node)

        for l2_agent in level2_agents:
            # Add Level 2 agent node
            workflow.add_node(f"{l2_agent.name}_supervisor", l2_agent.level2_supervisor_node)

        workflow.add_node("END", lambda state: {"empty_channel": 1})
        # Add conditional edges based on the should_continue function
        workflow.add_conditional_edges(
            "ceo",
            ceo_agent.should_continue,
            {
                "assistant": "ceo_assistant",
                "ceo": "ceo",
                "directors": f"ceo_router_down" ,  
                "executives": f"ceo_router_down" , 
                END: "END"
            }
        )
        workflow.set_finish_point("END")

        workflow.add_conditional_edges(
            "ceo_assistant",
            ceo_agent.should_continue_assistant,
            {
                "continue": "ceo_tool",
                "ceo": "ceo",
            }
        )

        workflow.add_edge("ceo_tool", "ceo_assistant")

        for l2_agent in level2_agents:

            workflow.add_edge("ceo_router_down", f"{l2_agent.name}_supervisor")


            router_name_down = f"{l2_agent.name}_router_down"

            def create_level2_router_down(agent_name):
                def level2_router(state):
                    logging.info(f"{agent_name} Router Down - Processing")
                    time.sleep(1)  # Add 2-second delay
                    return {"empty_channel": 1}
                level2_router.__name__ = f"{agent_name}_router_down"
                return level2_router
        
            router_function_down = create_level2_router_down(l2_agent.name)

            workflow.add_node(router_name_down, router_function_down)

            router_name_up = f"{l2_agent.name}_router_up"

            def create_level2_router_up(agent_name, subordinates):
                def level2_router(state):
                    return "complete" if all([getattr(state, f"{sub}_mode")[-1] == "converse" 
                                              for sub in subordinates]) else "continue"
                level2_router.__name__ = f"{agent_name}_router_up"
                return level2_router
            
            def create_level2_router_up_node(agent_name):
                def level2_router_node(state):
                    logging.info(f"{agent_name} Router Up Node - Processing")
                    time.sleep(5)  # Add 2-second delay
                    return {"empty_channel": 1}
                level2_router_node.__name__ = f"{agent_name}_router_up"
                return level2_router_node

            router_function_up = create_level2_router_up(l2_agent.name, l2_agent.subordinates)
            router_node_up = create_level2_router_up_node(l2_agent.name)
            workflow.add_node(router_name_up, router_node_up)

            for l1_agent in level1_agents :
                if l1_agent.name in l2_agent.subordinates:
                    workflow.add_edge(router_name_down , f"agent_{l1_agent.name}")

                    workflow.add_conditional_edges(
                    f"agent_{l1_agent.name}",
                    #lambda s: "assistant" if l1_agent.get_attr(s, "mode")[-1] == "research" else "router",
                    lambda state: l1_agent.get_attr(state, "mode")[-1] if l1_agent.get_attr(state, "mode") else "converse",
                        {
                            "research" : f"assistant_{l1_agent.name}",
                            "converse": router_name_up
                        }
                    )
                    workflow.add_conditional_edges(f"assistant_{l1_agent.name}", l1_agent.should_continue,
                    {
                    # If `tools`, then we call the tool node.
                        "continue": f"tools_{l1_agent.name}",
                    # Otherwise we finish.
                        "executive_agent" : f"agent_{l1_agent.name}",
                    }, 
                    )
                    workflow.add_edge(f"tools_{l1_agent.name}", f"assistant_{l1_agent.name}")

            # Add conditional edges for the router
            workflow.add_conditional_edges(
                router_name_up,
                router_function_up,
                {
                    "complete": f"{l2_agent.name}_supervisor",
                    "continue": router_name_up  # Loop back if not all subordinates are ready
                }
            )
            workflow.add_conditional_edges(
                f"{l2_agent.name}_supervisor",
                l2_agent.should_continue,
                            {
                    "aggregate_for_ceo": "ceo_router_up",
                    "break_down_for_executives": router_name_down  # Loop back if not all subordinates are ready
                }


                
            )
        workflow.add_conditional_edges(
            "ceo_router_up",
            create_ceo_router_up(level2_agents),
            {
                "complete": "ceo",
                "continue": "ceo_router_up"  # Loop back if not all subordinates are ready
            }
        )

                
        # Compile the main graph
        if self.interrupt_graph_before:
            final_graph = workflow.compile(
                checkpointer=self.memory,
                interrupt_before=[
                    #"ceo",
                    *[f"{l2_agent.name}_supervisor" for l2_agent in level2_agents],
                    *[f"agent_{l1_agent.name}" for l1_agent in level1_agents]
            ]
            )
        else:
            final_graph = workflow.compile(
                checkpointer=self.memory,
            )
        
        self.logger.info("Agents graph created successfully")

        return final_graph , unified_state_schema

    def get_graph_image(self, name):   
        Image.open(io.BytesIO(self.final_graph.get_graph().draw_mermaid_png())).save(f'{name}.png')
    
    def start(self, initial_state):
        # Ensure recursion_limit is set before starting
        if "recursion_limit" not in self.config:
            self.config["recursion_limit"] = 50
            
        result = self.final_graph.invoke(initial_state, self.config)
        if result is None:
            values = self.final_graph.get_state(self.config).values
            last_state = next(iter(values))
            return values[last_state]
        return result
    
    def resume(self, new_state: dict):
        # Ensure recursion_limit is set before resuming
        if "recursion_limit" not in self.config:
            self.config["recursion_limit"] = 50
            
        # Get the current state values
        current_state = self.final_graph.get_state(self.config).values
        # Update the current state with new values from new_state
        if new_state:  # This checks if new_state is not empty
            state = {}
            for key, value in current_state.items():
                for k, v in new_state.items():
                    if k == key:
                        state[key] = v
                    else:
                        state[key] = value
        else:
            state = current_state

        # Update the state in the graph
        if state != current_state:
            self.final_graph.update_state(self.config, state)
        
        # Invoke the graph with the updated state
        result = self.final_graph.invoke(None, self.config)
        
        if result is None:
            print("this is the result",result)
            values = self.final_graph.get_state(self.config).values
            last_state = next(iter(values))
            return self.final_graph.get_state(self.config).values[last_state]
        
        return result

    def update_config(self, new_config: dict):
        """
        Update the current configuration with new values.
        
        :param new_config: A dictionary containing the new configuration values.
        """
        self.config.update(new_config)
        self.logger.info(f"Configuration updated: {self.config}")

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info("Starting script execution")

    logger.info("About to create StateMachines instance")
    try:
        state_machines = StateMachines("Data/Prompts")
        logger.info("StateMachines instance created successfully")
    except Exception as e:
        logger.error(f"Error creating StateMachines instance: {str(e)}", exc_info=True)
        sys.exit(1)

    # Create an initial state with default values using the unified state schema
    initial_state = state_machines.unified_state_schema()
    save_graph_img = state_machines.get_graph_image("agents_graph")
    # Set specific initial values
    initial_state.company_knowledge = ["Our company is a leading tech firm specializing in AI and machine learning solutions."]
    initial_state.news_insights = ["Recent advancements in natural language processing have opened new opportunities in the market."]
    initial_state.ceo_mode = ["research_information"]

    # Start the graph execution
    logger.info("Starting graph execution...")
    result = state_machines.start(initial_state)
    
    while result is not None:
        logger.info(f"Current state: {result}")
        # Here you can add logic to handle the current state and provide new values
        # For example:
        new_values = {}  # Add any necessary updates to the state
        result = state_machines.resume(new_values)

    logger.info("Graph execution completed.")


















