import os
from dotenv import load_dotenv, set_key
from neo4j import GraphDatabase
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.chains import GraphCypherQAChain
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
import subprocess
import logging
from langchain_core.documents import Document
from langchain.tools import BaseTool
from jinja2 import Environment, FileSystemLoader
import chromadb
from chromadb import Settings
from langchain_openai import OpenAIEmbeddings
from langchain.schema import SystemMessage, HumanMessage
import uuid
import json

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

class DuplicateEntities(BaseModel):
    entities: List[str] = Field(
        description="List of at least 2 entities or entity relationships that represent the same object or real-world entity, but are not identical in their spelling and should be merged. The entity or relationship name to keep should be the first one in the list.",
        min_items=2
    )   

class Disambiguate(BaseModel):
    merge_entities: Optional[List[DuplicateEntities]] = Field(
        description="Lists of entities that represent the same object or real-world entity but are not identical in their spelling and should be merged in the graph to avoid confusion"
    )

class GraphKnowledgeManager:
    def __init__(
        self, 
        name: str, 
        level: str,
        prompt_dir: str, 
        llm_models: str = "gpt-4-turbo", 
        cypher_llm_model: str = "gpt-4",
        qa_llm_model: str = "gpt-3.5-turbo",
        cypher_llm_params: Dict[str, Any] = None,
        qa_llm_params: Dict[str, Any] = None,
        chain_verbose: bool = True,
        chain_callback_manager: Optional[Any] = None,
        chain_memory: Optional[Any] = None,
        similarity_threshold: float = 0.7,
        max_iterations: int = 3,
        execution_timeout: int = 60,
        max_retries: int = 2,
        return_intermediate_steps: bool = False,
        handle_retries: bool = True,
        allowed_nodes: List[str] = None,
        allowed_relationships: List[str] = None,
        strict_mode: bool = True,
        node_properties: Union[bool, List[str]] = False,
        relationship_properties: Union[bool, List[str]] = False,
        ignore_tool_usage: bool = False,
        **llm_params
    ):
        load_dotenv()
        self.name = name
        self.level = level
        self.prompt_dir = prompt_dir
        self.llm_models = llm_models
        self.llm_params = llm_params
        self.cypher_llm_model = cypher_llm_model
        self.qa_llm_model = qa_llm_model
        self.cypher_llm_params = cypher_llm_params if cypher_llm_params is not None else {}
        self.qa_llm_params = qa_llm_params if qa_llm_params is not None else {}
        self.chain_verbose = chain_verbose
        self.chain_callback_manager = chain_callback_manager
        self.chain_memory = chain_memory
        self.similarity_threshold = similarity_threshold
        self.max_iterations = max_iterations
        self.execution_timeout = execution_timeout
        self.max_retries = max_retries
        self.return_intermediate_steps = return_intermediate_steps
        self.handle_retries = handle_retries
        self.allowed_nodes = allowed_nodes
        self.allowed_relationships = allowed_relationships
        self.strict_mode = strict_mode
        self.node_properties = node_properties
        self.relationship_properties = relationship_properties
        self.ignore_tool_usage = ignore_tool_usage
        self.aura_instance_name = f"aura-{self.name}"
        self.neo4j_graph = None
        self.jinja_env = Environment(loader=FileSystemLoader(os.path.join(self.prompt_dir)))
        self.graph_system_prompt = self._load_graph_system_prompt()
        self.schema = self._load_schema()
        
        self.neo4j_uri = os.getenv('NEO4J_URI')
        self.neo4j_user = os.getenv('NEO4J_USERNAME')
        self.neo4j_password = os.getenv('NEO4J_PASSWORD')
        
        self.ensure_db_exists_and_connect()
        
        # Initialize GraphCypherQAChain
        self.cypher_chain = GraphCypherQAChain.from_llm(
            cypher_llm=self._create_cypher_llm(),
            qa_llm=self._create_qa_llm(),
            graph=self.neo4j_graph,
            verbose=self.chain_verbose,
            callback_manager=self.chain_callback_manager,
            memory=self.chain_memory,
            prompt_template=self._create_prompt_template(),
            similarity_threshold=self.similarity_threshold,
            max_iterations=self.max_iterations,
            execution_timeout=self.execution_timeout,
            max_retries=self.max_retries,
            return_intermediate_steps=self.return_intermediate_steps,
            handle_retries=self.handle_retries,
            allow_dangerous_requests=True  # Add this line
        )
        
        # Initialize LLMGraphTransformer
        graph_llm = ChatOpenAI(
            model_name=self.llm_models,
            **self.llm_params
        )
        
        graph_prompt = ChatPromptTemplate.from_template(self.graph_system_prompt)
        
        self.llm_transformer = LLMGraphTransformer(
            llm=graph_llm,
            allowed_nodes=self.allowed_nodes,
            allowed_relationships=self.allowed_relationships,
            prompt=graph_prompt,
            strict_mode=self.strict_mode,
            node_properties=self.node_properties,
            relationship_properties=self.relationship_properties,
            ignore_tool_usage=self.ignore_tool_usage
        )

    def _create_prompt_template(self) -> ChatPromptTemplate:
        schema = self.schema
        
        # Format the schema for better readability
        formatted_schema = "Nodes:\n"
        for node in schema["nodes"]:
            formatted_schema += f"- {node['label']}: {', '.join(node['properties'])}\n"
        
        formatted_schema += "\nRelationships:\n"
        for rel in schema["relationships"]:
            formatted_schema += f"- {rel['start_node']}-[{rel['type']}]->{rel['end_node']}: {', '.join(rel['properties'])}\n"

        template = f"""
        You are an AI assistant for querying a Neo4j graph database about PEPFAR (President's Emergency Plan for AIDS Relief) and its impact. Translate the user's questions into Cypher queries.
        Only provide the Cypher query without any explanations or additional text.
        Ensure that the queries are optimized and follow best practices for graph databases.

        The current database schema is:
        {{formatted_schema}}

        Use this schema information to construct your Cypher queries.

        User question: {{question}}

        Cypher query:
        """
        return ChatPromptTemplate.from_messages([
            ("system", template),
            ("human", "{{question}}")
        ])

    def _create_cypher_llm(self) -> ChatOpenAI:
        return ChatOpenAI(
            temperature=self.cypher_llm_params.get("temperature", 0),
            model_name=self.cypher_llm_model,
            **self.cypher_llm_params
        )

    def _create_qa_llm(self) -> ChatOpenAI:
        return ChatOpenAI(
            temperature=self.qa_llm_params.get("temperature", 0),
            model_name=self.qa_llm_model,
            **self.qa_llm_params
        )

    def ensure_db_exists_and_connect(self):
        if not all([self.neo4j_uri, self.neo4j_user, self.neo4j_password]):
            self.create_neo4j_instance()
        
        try:
            self.neo4j_graph = Neo4jGraph(
                url=self.neo4j_uri,
                username=self.neo4j_user,
                password=self.neo4j_password
            )
            logger.info(f"Connected to Neo4j instance: {self.neo4j_uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j instance: {e}")
            raise

    def create_neo4j_instance(self):
        # For demonstration purposes, we'll use a local Neo4j instance
        # In a real-world scenario, you'd interact with Neo4j Aura's API to create a new instance
        self.neo4j_uri = "bolt://localhost:7687"
        self.neo4j_user = "neo4j"
        self.neo4j_password = str(uuid.uuid4())  # Generate a random password
        
        # Save the connection details to .env file
        env_file_path = '.env'
        set_key(env_file_path, 'NEO4J_URI', self.neo4j_uri)
        set_key(env_file_path, 'NEO4J_USERNAME', self.neo4j_user)
        set_key(env_file_path, 'NEO4J_PASSWORD', self.neo4j_password)
        logger.info(f"Neo4j connection details saved to '{env_file_path}'.")

    def query_graph(self, question: str) -> Optional[Any]:
        try:
            response = self.cypher_chain.invoke(question)
            logger.info(f"Query result: {response}")
            return response
        except Exception as e:
            logger.error(f"Error executing GraphCypherQAChain: {e}")
            return None

    def populate_knowledge_graph(self, texts: List[str], batch_size: int = 100):
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            documents = [Document(page_content=text) for text in batch]
            graph_documents = self.llm_transformer.convert_to_graph_documents(documents)
            
            self.neo4j_graph.add_graph_documents(graph_documents)
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")

    def delete_node_or_relationship(self, identifier: str):
        try:
            self.neo4j_graph.query(f"MATCH (n) WHERE n.id = '{identifier}' DETACH DELETE n")
            logger.info(f"Deleted node or relationship with identifier '{identifier}'")
        except Exception as e:
            logger.error(f"Error deleting node or relationship: {e}")

    def delete_database(self):
        try:
            with GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password)) as driver:
                with driver.session() as session:
                    session.run("MATCH (n) DETACH DELETE n")
            logger.info("Deleted all nodes and relationships in the database.")
        except Exception as e:
            logger.error(f"Error deleting database: {e}")

    def query_graph_tool(self) -> BaseTool:
        return BaseTool(
            name="query_graph",
            func=self.query_graph,
            description="Query the Neo4j graph database"
        )

    def populate_knowledge_graph_tool(self) -> BaseTool:
        return BaseTool.from_function(self.populate_knowledge_graph)

    def delete_node_or_relationship_tool(self) -> BaseTool:
        return BaseTool.from_function(self.delete_node_or_relationship)

    def delete_database_tool(self) -> BaseTool:
        return BaseTool.from_function(self.delete_database)

    def get_tools(self) -> List[BaseTool]:
        return [
            self.query_graph_tool(),
            self.populate_knowledge_graph_tool(),
            self.delete_node_or_relationship_tool(),
            self.delete_database_tool(),
        ]

    def _load_graph_system_prompt(self):
        try:
            template = self.jinja_env.get_template('graph_system_prompt.j2')
            return template.render()
        except Exception as e:
            logger.error(f"Error loading graph system prompt: {e}")
            # Fallback to a default prompt if the template can't be loaded
            return """
            You are an AI assistant specialized in transforming unstructured text into graph structures for a Neo4j database.
            Identify entities, relationships, and properties from the given text and represent them in a graph format.
            """

    def check_aura_instance(self):
        try:
            result = subprocess.run(
                ["aura", "instances", "list", "--format", "json"],
                capture_output=True, text=True, check=True
            )
            instances = json.loads(result.stdout)
            for instance in instances:
                if instance['name'] == self.aura_instance_name:
                    logger.info(f"Aura instance '{self.aura_instance_name}' found.")
                    return True
            logger.warning(f"Aura instance '{self.aura_instance_name}' not found.")
            return False
        except subprocess.CalledProcessError as e:
            logger.error(f"Error checking Aura instance: {e}")
            logger.error(f"Command output: {e.stdout}")
            logger.error(f"Error output: {e.stderr}")
            return False

    def _load_schema(self):
        schema_path = os.path.join(self.prompt_dir, 'schema.json')
        try:
            with open(schema_path, 'r') as schema_file:
                return json.load(schema_file)
        except FileNotFoundError:
            logger.warning(f"Schema file not found at {schema_path}. Using default schema.")
            return ''


class VectorDBManager:
    def __init__(
        self,
        name: str,
        prompt_dir: str,
        llm_models: str = "gpt-4-turbo",
    ):
        self.name = name
        self.prompt_dir = prompt_dir
        self.llm_models = llm_models
        self.vector_db = self._ensure_vector_db_exists_and_connect()
        self.embeddings = OpenAIEmbeddings()

    def _ensure_vector_db_exists_and_connect(self):
        try:
            persist_directory = f"./chroma_db_{self.name}"
            os.makedirs(persist_directory, exist_ok=True)
            vector_db = chromadb.Client(Settings(persist_directory=persist_directory))
            collection_name = f"{self.name}_collection"
            try:
                vector_db.get_collection(collection_name)
            except ValueError:
                vector_db.create_collection(collection_name)
            logger.info(f"Connected to Chroma vector database: {self.name}")
            return vector_db
        except Exception as e:
            logger.error(f"Failed to connect to Chroma vector database '{self.name}': {e}")
            raise

    def populate_vector_db(self, documents: List[Dict[str, Any]], chunk_size: int = 1000, chunk_overlap: int = 200):
        collection = self.vector_db.get_collection(f"{self.name}_collection")
        for doc in documents:
            embedding = self.embeddings.embed_query(doc['content'])
            collection.add(
                embeddings=[embedding],
                documents=[doc['content']],
                metadatas=[doc['metadata']]
            )
        logger.info(f"Populated vector database with {len(documents)} documents")

    def query_chroma_db(self, question: str, top_k: int = 5) -> Optional[str]:
        try:
            collection = self.vector_db.get_collection(f"{self.name}_collection")
            query_embedding = self.embeddings.embed_query(question)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            return results['documents'][0] if results['documents'] else None
        except Exception as e:
            logger.error(f"Error querying Chroma DB: {e}")
            return None

    def populate_vector_db_tool(self) -> BaseTool:
        return BaseTool.from_function(self.populate_vector_db)

    def query_vector_db_tool(self) -> BaseTool:
        return BaseTool.from_function(self.query_chroma_db)

    def get_tools(self) -> List[BaseTool]:
        return [
            self.populate_vector_db_tool(),
            self.query_vector_db_tool(),
        ]

# Test the code
def main():
    prompt_dir = "prompts/level1/agent1"
    name = "agent1"
    gkm = None
    
    # Initialize GraphKnowledgeManager
    gkm = GraphKnowledgeManager(
        name=name,
        level="level1",
        prompt_dir=prompt_dir,
        temperature=0.2,
        max_tokens=4000,
    )
    
    # Populate the knowledge graph with insights
    insights = [
        "PEPFAR, started in 2003, has saved over 25 million lives through antiretroviral treatments.",
        "Dr. Emily Kainne Dokubo, working for PEPFAR, emphasizes partnerships in addressing disparities in pediatric HIV treatment.",
    ]
    gkm.populate_knowledge_graph(insights, batch_size=25)
    
    # Query the graph database
    result = gkm.query_graph("How many lives has PEPFAR saved?")
    print(f"Answer from graph DB: {result}")
    

        
        # Optionally, clean up the database
        # gkm.delete_database()

if __name__ == "__main__":
    main()
