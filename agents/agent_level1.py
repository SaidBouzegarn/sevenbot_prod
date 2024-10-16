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
from agents.agent_base import BaseAgent
from langchain_core.messages import trim_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

set_llm_cache(InMemoryCache())


##########################################################################################
#################################### Level 1 agent #######################################
##########################################################################################

class Level1State(BaseModel):
    level1_2_conversation: Annotated[List[Union[AIMessage, BaseMessage]], add_messages]
    level1_3_conversation: Annotated[List[Union[AIMessage, BaseMessage]], add_messages]
    assistant_conversation: Annotated[List[Union[AIMessage, BaseMessage]], add_messages]
    domain_knowledge: Any
    mode: Literal["research", "converse"]

class Level1Decision(BaseModel):
    reasoning: str
    decision: Literal["search_more_information", "converse_with_superiors"]
    content: List[str]



# Define trimmer
# count each message as 1 "token" (token_counter=len) and keep only the last two messages
trimmer = trim_messages(strategy="last", max_tokens=3, token_counter=len)




class Level1Agent(BaseAgent):
    def __init__(self, *args, **kwargs):
        self.state_schema = Level1State
        super().__init__(*args, **kwargs)
        self.prompt_dir = os.path.join(os.path.dirname(__file__), '..', 'prompts/level1', self.name)
        self.jinja_env = Environment(loader=FileSystemLoader(self.prompt_dir))

    def _create_graph(self) -> StateGraph:
        
        def level1_node(state: Level1State) -> Dict[str, Any]:
            
            #we want to get the last message from the level 3 conversation or level2 conversation based on the one that has last msg
            time1 = datetime.fromisoformat(state.level1_2_conversation[-1].timestamp)
            time2 = datetime.fromisoformat(state.level1_3_conversation[-1].timestamp)

            last_message = state.level1_2_conversation if time1 > time2 else state.level1_3_conversation
            last_message = last_message[-1].content if last_message else "No messages from level 2 or level 3 yet."

            if self.debug:
                print(f"Last message: {last_message}")

            decision_prompt = self.jinja_env.get_template('decision_prompt.j2').render(
                last_message=last_message,
                level1_2_conversation=trimmer.invoke(state.level1_2_conversation) if state.level1_2_conversation else "No messages from level 2 yet.",
                level1_3_conversation=trimmer.invoke(state.level1_3_conversation) if state.level1_3_conversation else "No messages from level 3 yet.",
                assistant_conversation=trimmer.invoke(state.assistant_conversation) if state.assistant_conversation else "No messages from assistant yet.",
                tools=self.tools
            )
            system_prompt = self.jinja_env.get_template('system_prompt.j2').render(
                tools=self.tools
            )
            messages = [SystemMessage(content=system_prompt)] + [BaseMessage(content=decision_prompt)]

            structured_llm = self.llm.with_structured_output(Level1Decision)
            response = structured_llm.invoke(messages)

            if self.debug:
                print(f"Reasoning: {response.reasoning}")
                print(f"Decision: {response.decision}")
                print(f"Content: {response.content}")
                
            if response.decision == "search_more_information":
                questions = response.content
                state.assistant_conversation.extend([BaseMessage(content=q) for q in questions])
                state.mode = "research"
            elif response.decision == "converse_with_superiors":
                message = self.create_message(response.content[0] if response.content else "")
                if time1 > time2:
                    state.level1_2_conversation.append(message)
                else:
                    state.level1_3_conversation.append(message)
                state.mode = "converse_with_superiors"
            return state

        def assistant_node(state: Level1State) -> Dict[str, Any]:
            new_messages = []
            prompt = self.jinja_env.get_template('assistants_prompt.j2')

            for msg in state.assistant_conversation:
                if isinstance(msg, HumanMessage) and not any(isinstance(r, AIMessage) for r in new_messages):
                    print(f"Processing question: {msg.content}")
                
                    answer = self.llm.invoke([HumanMessage(content=prompt.render(question=msg.content))])
                    answer_content = answer.content if isinstance(answer.content, str) else str(answer.content)
                    
                    new_messages.append(msg)
                    new_messages.append(self.create_message(answer_content))
                    
                    if self.debug:
                        print(f"Answer: {answer_content}")
            
            state.assistant_conversation.extend(new_messages)
            return state

        def should_continue(state: Level1State):
            # Check if there are any unanswered questions in the assistant conversation
            unanswered_questions = [msg for msg in state.assistant_conversation if isinstance(msg, HumanMessage)]
            
            if unanswered_questions:
                # If there are unanswered questions, check the last message
                last_message = state.assistant_conversation[-1]
                
                if isinstance(last_message, AIMessage):
                    # If the last message is from AI and has tool calls, continue to tools
                    if last_message.tool_calls:
                        return "continue"
                    # If the last message is from AI but has no tool calls, it's an answer
                    # Remove the corresponding question and continue checking
                    state.assistant_conversation.pop(-2)  # Remove the question
                    return should_continue(state)  # Recursive call to check next Q&A pair
                else:
                    # If the last message is a question, continue to tools
                    return "continue"
            else:
                # If all questions are answered, go back to level1
                return "level1"

        tool_node = ToolNode(self.tools)

        workflow = StateGraph(self.state_schema)
        workflow.add_node("level1", level1_node)
        workflow.add_node("assistant", assistant_node)
        workflow.add_node("tools", tool_node)

        workflow.set_entry_point("level1")
        
        workflow.add_conditional_edges(
            "level1",
            lambda s: "assistant" if s.mode == "research" else END,
            {
                "assistant": "assistant",
                END: END
            }
        )
        workflow.add_conditional_edges("assistant", should_continue,    {
        # If `tools`, then we call the tool node.
        "continue": "tools",
        # Otherwise we finish.
            "level1": "level1",
        }, 
        )
        workflow.add_edge("tools", "assistant")
        #workflow.add_edge("assistant", "level1")  # Changed 'tools' to 'level1'

        return workflow.compile(checkpointer=self.checkpointer, store=self.memory_store)


##########################################################################################
#################################### Level 2 agent #######################################
##########################################################################################



##########################################################################################
#################################### Level 3 agent #######################################
##########################################################################################

class Level3State(BaseModel):
    news: List[str]
    insights: List[str]
    domain_insights: List[str]
    level3_conversation: Annotated[List[Union[AIMessage, BaseMessage]], add_messages]
    level2_conversation: Annotated[List[Union[AIMessage, BaseMessage]], add_messages]
    assistant_conversation: Annotated[List[Union[AIMessage, BaseMessage]], add_messages]
    domain_knowledge: Any
    mode: Literal["research", "converse_with_level2", "converse_with_level4"]

class Level3Decision(BaseModel):
    reasoning: str
    decision: Literal["search_more_information", "converse_with_level2", "converse_with_level4"]
    questions: List[str]

class Level3Agent(BaseAgent):
    def __init__(self, *args, **kwargs):
        self.state_schema = Level3State
        super().__init__(*args, **kwargs)
        self.prompt_dir = os.path.join(os.path.dirname(__file__), '..', 'prompts', self.name)
        self.jinja_env = Environment(loader=FileSystemLoader(self.prompt_dir))

    def _create_graph(self) -> StateGraph:
        def level3_node(state: Level3State) -> Dict[str, Any]:
            decision = self._make_decision(state)
            
            if decision.decision == "search_more_information":
                state.assistant_conversation.extend([HumanMessage(content=q) for q in decision.questions])
                state.mode = "research"
            elif decision.decision == "converse_with_level2":
                message = self._generate_level2_message(state)
                state.level3_conversation.append(message)
                state.mode = "converse_with_level2"
            elif decision.decision == "converse_with_level4":
                message = self._generate_level4_message(state)
                state.level3_conversation.append(message)
                state.mode = "converse_with_level4"
            return state

        def assistant_node(state: Level3State) -> Dict[str, Any]:
            for msg in state.assistant_conversation:
                if isinstance(msg, HumanMessage) and not any(isinstance(r, AIMessage) for r in state.assistant_conversation[state.assistant_conversation.index(msg):]):
                    answer = self._answer_question(msg.content)
                    state.assistant_conversation.append(answer)
            return state

        def moderator_node(state: Level3State) -> Dict[str, Any]:
            return state

        workflow = StateGraph(self.state_schema)
        workflow.add_node("level3", level3_node)
        workflow.add_node("assistant", assistant_node)
        workflow.add_node("moderator", moderator_node)

        workflow.set_entry_point("level3")
        
        workflow.add_conditional_edges(
            "level3",
            lambda s: s.mode,
            {
                "research": "assistant",
                "converse_with_level2": "moderator",
                "converse_with_level4": END
            }
        )

        workflow.add_edge("assistant", "level3")
        workflow.add_edge("moderator", END)

        return workflow.compile(checkpointer=self.checkpointer, store=self.memory_store)

    def _make_decision(self, state: Level3State) -> Level3Decision:
        prompt = self.jinja_env.get_template('level3_decision_prompt.j2').render(
            level3_conversation=state.level3_conversation,
            level2_conversation=state.level2_conversation,
            assistant_conversation=state.assistant_conversation,
            news=state.news,
            insights=state.insights,
            domain_insights=state.domain_insights
        )   
        structured_llm = self.llm.with_structured_output(Level3Decision)
        response = structured_llm.invoke([HumanMessage(content=prompt)])

        if self.debug:
            print(f"Decision: {response.decision}")
            print(f"Reasoning: {response.reasoning}")
            print(f"Questions: {response.questions}")
        return response 
        
    def _generate_level2_message(self, state: Level3State) -> AIMessage:
        prompt = self.jinja_env.get_template('level3_to_level2_message_prompt.j2').render(
            level3_conversation=state.level3_conversation,
            level2_conversation=state.level2_conversation,
            assistant_conversation=state.assistant_conversation,
            news=state.news,
            insights=state.insights,
            domain_insights=state.domain_insights
        )
        response = self.llm.invoke([HumanMessage(content=prompt)])

        if self.debug:
            print(f"Level2 Message: {response.content}")

        return self.create_message(response.content)

    def _generate_level4_message(self, state: Level3State) -> AIMessage:
        prompt = self.jinja_env.get_template('level3_to_level4_message_prompt.j2').render(
            level3_conversation=state.level3_conversation,
            level2_conversation=state.level2_conversation,
            assistant_conversation=state.assistant_conversation,
            news=state.news,
            insights=state.insights,
            domain_insights=state.domain_insights
        )       
        response = self.llm.invoke([HumanMessage(content=prompt)])

        if self.debug:
            print(f"Level4 Message: {response.content}")

        return self.create_message(response.content)

    def _answer_question(self, question: str) -> AIMessage:
        prompt = f"""
        Answer the following question using available tools and knowledge graphs:

        Question:
        {question}
        """
        answer = self.llm.bind_tools(self.tools).invoke([HumanMessage(content=prompt)])
        if self.debug:
            print(f"Answer: {answer.content}")
        return self.create_message(answer.content.strip())

# Example usage:
if __name__ == "__main__":
    from langchain_openai import ChatOpenAI
    from dotenv import load_dotenv

    load_dotenv()

    tools = [DuckDuckGoSearchRun()]

    llm_eagent = ChatOpenAI(
        model="gpt-4",
        temperature=0.8,
        max_tokens=400,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        n=1,
        request_timeout=90,
        streaming=True,
    )
    
    agent1 = Level1Agent(
        name="agent1",
        llm=llm_eagent,
        tools=tools,
        system_message="You are the level 1 agent doing your work.",
        max_iterations=5,
        checkpointer=MemorySaver(),
        debug=True
    )

    level1_2 = Level1Agent(
        name="Level1_2",
        llm=llm_eagent,
        tools=tools,
        system_message="You are another level 1 agent doing your work.",
        max_iterations=5,
        checkpointer=MemorySaver(),
        debug=True
    )

    director = Level3Agent(
        name="director1",
        llm=llm_eagent,
        tools=tools,
        system_message="You are the level 3 agent overseeing the level 1 agents.",
        max_iterations=5,
        checkpointer=MemorySaver(),
        debug=True
    )

    initial_inputs = {
        "news": [
            "Eli Lilly announces 70% price cut on insulin products",
            "FDA approves Bigfoot Unity, a new smart insulin pen cap system",
            "Novo Nordisk launches Ozempic for type 2 diabetes in additional markets",
            "Researchers develop glucose-responsive 'smart' insulin",
            "Medtronic introduces new extended-wear infusion set for insulin pumps"
        ],
        "insights": [
            "Price cuts could significantly reduce hospital pharmacy costs",
            "Smart insulin delivery systems may improve patient outcomes and reduce hospital readmissions",
            "Extended-wear infusion sets could decrease nursing workload and improve patient satisfaction"
        ],
        "domain_insights": [
            "Hospitals are increasingly adopting closed-loop insulin delivery systems",
            "There's a growing trend towards personalized diabetes management using AI and big data",
            "Telemedicine is becoming crucial for remote insulin management and patient monitoring"
        ],
        "level1_conversation": [
            director.create_message("Hello, I need an update on recent insulin news and its impact on hospitals. Please provide information on two significant developments."),
            director.create_message("Certainly! I'll focus on Eli Lilly's recent insulin price cuts and new insulin delivery technologies being adopted by hospitals.")
        ],
        "assistant_conversation": [],
        "domain_knowledge": {},
        "mode": "research"
    }

    # Visualize the graph
    mermaid_png = agent1.graph.get_graph().draw_mermaid_png()
    img = Image.open(io.BytesIO(mermaid_png))
    img.save('level1_agent_graph.png')

    result = agent1.run(initial_inputs)
    print(result)

    # Example of processing the conversation
    for message in result['level1_conversation']:
        if isinstance(message, level1_1.MessageClass):
            print(f"Level1_1: {message.content}")
        elif isinstance(message, level1_2.MessageClass):
            print(f"Level1_2: {message.content}")
        elif isinstance(message, level3.MessageClass):
            print(f"Level3: {message.content}")
        else:
            print(f"Unknown: {message.content}")

    domain_knowledge = {
    "key_players": ["Eli Lilly", "Novo Nordisk", "Sanofi", "Medtronic", "Dexcom", "Abbott"],
    "emerging_technologies": ["Artificial pancreas", "Smart insulin pens", "Continuous glucose monitors"],
    "regulatory_bodies": ["FDA", "EMA", "WHO"],
    "market_trends": ["Increasing prevalence of diabetes", "Growing demand for non-invasive insulin delivery"]
}
