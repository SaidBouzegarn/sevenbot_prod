from typing import Dict, TypedDict, Sequence, Union, List
from langchain_core.messages import HumanMessage, AIMessage
from llama_index import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.node_parser import SimpleNodeParser
from langgraph.graph import StateGraph, END
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

# Define our state
class ChatState(TypedDict):
    """State for the chat system."""
    messages: Sequence[Union[HumanMessage, AIMessage]]
    current_question: str
    retrieved_documents: List[Document]
    search_query: str

class RAGChatbot:
    def __init__(self):
        # Initialize LLM
        self.llm = ChatOpenAI(temperature=0)
        
        # Initialize vector store and index
        self.setup_vector_store()
        
        # Create the graph
        self.graph = self.create_graph()

    def setup_vector_store(self):
        """Initialize the vector store with documents"""
        # Load documents
        documents = SimpleDirectoryReader('data/').load_data()
        
        # Parse into nodes
        parser = SimpleNodeParser()
        nodes = parser.get_nodes_from_documents(documents)
        
        # Create vector store index
        self.index = VectorStoreIndex(nodes)
        self.retriever = self.index.as_retriever(similarity_top_k=3)

    def generate_search_query(self, state: ChatState) -> ChatState:
        """Generate an optimized search query based on the question"""
        if len(state["messages"]) <= 1:  # First question
            state["search_query"] = state["current_question"]
            return state
            
        # Use LLM to generate better search query
        chat_history = state["messages"][:-1]  # Exclude current question
        response = self.llm.predict(
            f"""Given the chat history and current question, generate a search query that will help find relevant information.
            Chat History: {chat_history}
            Current Question: {state["current_question"]}
            Search Query:"""
        )
        state["search_query"] = response
        return state

    def retrieve_documents(self, state: ChatState) -> ChatState:
        """Retrieve relevant documents using the search query"""
        docs = self.retriever.retrieve(state["search_query"])
        state["retrieved_documents"] = docs
        return state

    def generate_response(self, state: ChatState) -> ChatState:
        """Generate final response using retrieved documents"""
        # Format context from retrieved documents
        context = "\n".join([doc.text for doc in state["retrieved_documents"]])
        
        response = self.llm.predict(
            f"""Answer the question based on the provided context. If you cannot answer from the context, say so.
            Context: {context}
            
            Chat History: {state["messages"]}
            Question: {state["current_question"]}
            
            Answer:"""
        )
        
        state["messages"].append(AIMessage(content=response))
        return state

    def create_graph(self) -> StateGraph:
        """Create the LangGraph workflow"""
        # Initialize graph
        workflow = StateGraph(ChatState)
        
        # Add nodes
        workflow.add_node("generate_query", self.generate_search_query)
        workflow.add_node("retrieve", self.retrieve_documents)
        workflow.add_node("generate", self.generate_response)
        
        # Add edges
        workflow.add_edge("generate_query", "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        
        # Set entry point
        workflow.set_entry_point("generate_query")
        
        return workflow.compile()

    def chat(self, question: str, chat_history: List[Union[HumanMessage, AIMessage]] = None) -> str:
        """Main chat interface"""
        if chat_history is None:
            chat_history = []
            
        # Prepare state
        state = {
            "messages": chat_history + [HumanMessage(content=question)],
            "current_question": question,
            "retrieved_documents": [],
            "search_query": ""
        }
        
        # Execute graph
        result = self.graph.invoke(state)
        
        # Return the last AI message
        return result["messages"][-1].content
