from typing import List, Optional, Dict, Any, Union, Callable, Sequence, TypedDict, Annotated
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
from langgraph.graph.message import add_messages
from langchain_community.tools import DuckDuckGoSearchRun
from neo4j import GraphDatabase  # Import Neo4j driver


class QAEntry(TypedDict):
    question: str
    answer: Optional[str] = None


class EAgentState(TypedDict):
    branch_insights: List[str]
    branch_knowledge_graph: Any
    domain_insights: List[str]
    domain_knowledge_graph: Any
    QA_h: List[QAEntry]
    QA_l: List[QAEntry]
    answers: Dict[str, Any]


class EAgent:
    def __init__(
        self,
        llm: BaseLanguageModel,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        tools: List[BaseTool] = None,
        system_message: Optional[str] = None,
        max_iterations: int = 10,
        state_schema: Optional[BaseModel] = None,
        state_modifier: Optional[Union[str, SystemMessage, Callable, Runnable]] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
        memory_store: Optional[BaseStore] = None,
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
        # self.neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.graph = self._create_graph()

    def _create_graph(self) -> StateGraph:
        # Define the state schema if not provided
        if self.state_schema is None:
            self.state_schema = EAgentState

        def should_continue_eagent(state: EAgentState) -> str:
            # Logic to determine if EAgent should continue the loop
            if not self.is_satisfied(state['QA_h']):
                return "continue"
            else:
                return "end"

        def call_model_eagent(state: EAgentState) -> Dict[str, Any]:
            branch_insights = state['branch_insights']
            domain_insights = state['domain_insights']
            QA_h = state['QA_h']
            QA_l = state['QA_l']

            # Analyze current insights and QA
            analysis = self.analyze_state(branch_insights, domain_insights, QA_h, QA_l)

            # Generate new QA_l questions for the worker
            new_QA_l = self.generate_questions(analysis)

            # Update state with new QA_l
            state['QA_l'].extend(new_QA_l)
            return state

        def should_continue_worker(state: EAgentState) -> str:
            # Continue if there are unanswered questions in QA_l
            unanswered = [qa for qa in state['QA_l'] if qa['answer'] is None]
            if unanswered:
                return "continue"
            else:
                return "end"

        def call_model_worker(state: EAgentState) -> Dict[str, Any]:
            QA_l = state['QA_l']
            for qa in QA_l:
                if qa['answer'] is None:
                    answer = self.answer_question(qa['question'])
                    qa['answer'] = answer
            return state

        # Define WorkerAgent as a ToolNode
        worker_tool_node = ToolNode(self.tools)

        # Reference the local function without 'self.'
        e_agent_node = call_model_eagent

        # Define WorkerAgent node
        worker_agent_node = call_model_worker

        # Initialize the main workflow graph
        workflow = StateGraph(self.state_schema)
        workflow.add_node("E_Agent", call_model_eagent)
        workflow.add_node("WorkerAgent", worker_agent_node)
        workflow.set_entry_point("E_Agent")
        workflow.add_conditional_edges("E_Agent", should_continue_eagent, {
            "continue": "WorkerAgent",
            "end": END,
        })
        workflow.add_conditional_edges("WorkerAgent", should_continue_worker, {
            "continue": "E_Agent",
            "end": END,
        })

        return workflow.compile(checkpointer=self.checkpointer, store=self.memory_store)

    def _initialize_state(self, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Initializes the state and configuration based on the provided inputs.
        """
        if config is None:
            config = {}
        if 'configurable' not in config:
            config['configurable'] = {}
        if 'thread_id' not in config['configurable']:
            config['configurable']['thread_id'] = 'default_thread'

        if ('branch_insights' in inputs and 
            'branch_knowledge_graph' in inputs and 
            'QA_h' in inputs):
            # Inherit state from another node
            state = {
                "branch_insights": inputs['branch_insights'],
                "branch_knowledge_graph": inputs['branch_knowledge_graph'],
                "domain_insights": inputs.get('domain_insights', []),
                "domain_knowledge_graph": inputs.get('domain_knowledge_graph', {}),
                "QA_h": inputs['QA_h'],
                "QA_l": [],
                "answers": {}
            }
        else:
            state = {
                "branch_insights": [],
                "branch_knowledge_graph": {},
                "domain_insights": [],
                "domain_knowledge_graph": {},
                "QA_h": [],
                "QA_l": [],
                "answers": {}
            }

        return state, config

    def analyze_state(self, branch_insights: List[str], domain_insights: List[str],
                     QA_h: List[QAEntry], QA_l: List[QAEntry]) -> str:
        prompt = f"""
        Analyze the following data to determine the next steps:

        Branch Insights:
        {branch_insights}

        Domain Insights:
        {domain_insights}

        High-Level Q&A:
        {QA_h}

        Low-Level Q&A:
        {QA_l}

        Provide an analysis of the current state.
        """
        analysis_response = self.llm.invoke([HumanMessage(content=prompt)])
        return analysis_response.content  # Access the 'content' attribute

    def generate_questions(self, analysis: str) -> List[QAEntry]:
        prompt = f"""
        Based on the analysis below, generate new relevant questions to further explore the topic:

        Analysis:
        {analysis}

        Generate questions in the following format:
        - Question: <your question here>
        """
        questions_response = self.llm.invoke([HumanMessage(content=prompt)])
        questions = questions_response.content.strip().split('\n')  # Access the 'content' attribute
        qa_entries = []
        for q in questions:
            if q.startswith("- Question:"):
                question_text = q.replace("- Question:", "").strip()
                qa_entries.append({"question": question_text, "answer": None})
        return qa_entries

    def answer_question(self, question: str) -> str:
        prompt = f"""
        Answer the following question using available tools and knowledge graphs:

        Question:
        {question}
        """
        answer = self.llm.bind_tools(self.tools).invoke([HumanMessage(content=prompt)])
        return answer.content.strip()  # Access the 'content' attribute

    def evaluate_answers(self, QA_h: List[QAEntry], QA_l: List[QAEntry]) -> bool:
        # Implement logic to compare QA_l answers with QA_h questions
        for qa_h in QA_h:
            relevant_answers = [qa_l['answer'] for qa_l in QA_l if qa_l['question'] == qa_h['question']]
            # Placeholder condition: Check if answer is satisfactory
            if not relevant_answers or not relevant_answers[0]:
                return False
        return True

    def is_satisfied(self, QA_h: List[QAEntry]) -> bool:
        # Define logic to determine if QA_h questions are satisfactorily answered
        return self.evaluate_answers(QA_h, self.state['QA_l'])

    def run(self, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        state, config = self._initialize_state(inputs, config)
        self.state = state
        return self.graph.invoke(self.state, config)

    def stream(self, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        state, config = self._initialize_state(inputs, config)
        self.state = state
        return self.graph.stream(self.state, config)

    def arun(self, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        state, config = self._initialize_state(inputs, config)
        self.state = state
        return self.graph.ainvoke(self.state, config)

    def astream(self, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        state, config = self._initialize_state(inputs, config)
        self.state = state
        return self.graph.astream(self.state, config)


# Example usage:
if __name__ == "__main__":
    from langchain_openai import ChatOpenAI
    from dotenv import load_dotenv

    # Load environment variables from .env file
    load_dotenv()

    # Helper function for formatting the stream nicely
    import pprint
    
    def print_stream(stream):
        for s in stream:
            pprint.pprint(s)


    tools = [DuckDuckGoSearchRun()]

    # Initialize EAgent with Neo4j credentials
    llm_eagent = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    e_agent = EAgent(
        llm=llm_eagent,
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        tools=tools,
        system_message="You are the executive agent coordinating the workflow.",
        max_iterations=5,
        checkpointer=MemorySaver(),
        debug=True
    )

    # Example initial state inherited from another node
    initial_inputs = {
        "branch_insights": ["Company Y has launched a new wind turbine model."],
        "branch_knowledge_graph": {"nodes": [], "relationships": []},
        "QA_h": [
            {"question": "What are the recent advancements in wind technology?", "answer": None}
        ]
    }

    # Run the EAgent
    #result = e_agent.run(initial_inputs)
    #print(result)

    # Example usage of stream
    stream_result = e_agent.stream(initial_inputs)
    print_stream(stream_result)
"""
    # Example usage of arun (asynchronous run)
    import asyncio

    async def async_run():
        async_result = await e_agent.arun(initial_inputs)
        print(async_result)

    asyncio.run(async_run())

    # Example usage of astream (asynchronous stream)
    async def async_stream():
        async_stream_result = await e_agent.astream(initial_inputs)
        print_stream(async_stream_result)

    asyncio.run(async_stream())"""