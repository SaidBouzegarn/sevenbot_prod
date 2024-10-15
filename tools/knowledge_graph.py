import os
from py2neo import Graph, Node, Relationship
from dotenv import load_dotenv, set_key
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Optional

# Load environment variables from .env file
load_dotenv()

def create_knowledge_graph():
    """
    Create an empty knowledge graph and write the necessary information to the .env file.
    """
    # Retrieve necessary information from .env file
    neo4j_host = os.getenv('NEO4J_HOST', 'localhost')
    neo4j_port = os.getenv('NEO4J_PORT', '7687')
    neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
    neo4j_password = os.getenv('NEO4J_PASSWORD', 'password')

    # Initialize Neo4j graph
    graph = Graph(host=neo4j_host, port=neo4j_port, user=neo4j_user, password=neo4j_password)
    print("Empty knowledge graph created successfully.")

    # Write the graph information to the .env file
    env_file_path = '.env'
    set_key(env_file_path, 'NEO4J_HOST', neo4j_host)
    set_key(env_file_path, 'NEO4J_PORT', neo4j_port)
    set_key(env_file_path, 'NEO4J_USER', neo4j_user)
    set_key(env_file_path, 'NEO4J_PASSWORD', neo4j_password)
    print(f"Neo4j connection details set in '{env_file_path}'.")

def populate_knowledge_graph(graph: Graph, texts: List[str], 
                             allowed_nodes: List[str] = [],
                             allowed_relationships: List[str] = [],
                             strict_mode: bool = True,
                             node_properties: bool | List[str] = False,
                             relationship_properties: bool | List[str] = False,
                             ignore_tool_usage: bool = False,
                             prompt: Optional[ChatPromptTemplate] = None):
    """
    Populate the Neo4j knowledge graph with information extracted from the given texts.

    Args:
        graph (Graph): The Neo4j graph object to populate.
        texts (List[str]): List of strings, each containing up to 80 words of text.
        allowed_nodes (List[str]): List of allowed node types.
        allowed_relationships (List[str]): List of allowed relationship types.
        strict_mode (bool): Whether to strictly adhere to allowed nodes and relationships.
        node_properties (bool | List[str]): Whether to extract node properties or a list of specific properties.
        relationship_properties (bool | List[str]): Whether to extract relationship properties or a list of specific properties.
        ignore_tool_usage (bool): Whether to ignore tool usage in the LLM.
        prompt (Optional[ChatPromptTemplate]): Custom prompt template for the LLMGraphTransformer.
    """

    custom_prompt = ChatPromptTemplate.from_template(
    "Extract key information from the following text and represent it as a graph. "
    "Focus on identifying people, their fields of study, and major discoveries or contributions. "
    "Text: {text}"
)
    # Initialize the LLM and LLMGraphTransformer
    llm = ChatOpenAI(temperature=0, model_name="gpt-4-turbo")
    llm_transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=allowed_nodes,
        allowed_relationships=allowed_relationships,
        strict_mode=strict_mode,
        node_properties=node_properties,
        relationship_properties=relationship_properties,
        ignore_tool_usage=ignore_tool_usage,
        prompt=prompt
    )

    # Convert texts to Documents
    documents = [Document(page_content=text) for text in texts]

    # Convert documents to graph documents
    graph_documents = llm_transformer.convert_to_graph_documents(documents)

    # Add graph documents to the Neo4j graph
    for graph_doc in graph_documents:
        # Add nodes
        for node in graph_doc.nodes:
            neo4j_node = Node(node.type, **node.properties)
            graph.merge(neo4j_node, node.type, "id")

        # Add relationships
        for rel in graph_doc.relationships:
            start_node = graph.nodes.match(rel.source.type, id=rel.source.id).first()
            end_node = graph.nodes.match(rel.target.type, id=rel.target.id).first()
            if start_node and end_node:
                neo4j_rel = Relationship(start_node, rel.type, end_node, **rel.properties)
                graph.merge(neo4j_rel)

    print(f"Added {len(graph_documents)} documents to the Neo4j knowledge graph.")



