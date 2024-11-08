import os
from dotenv import load_dotenv, set_key
from neo4j import GraphDatabase
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_cohere import ChatCohere
from langchain_groq import ChatGroq
from langchain_google_vertexai import ChatVertexAI
from langchain_community.chat_models import ChatOllama
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_anthropic import ChatAnthropic
from langchain_fireworks import ChatFireworks
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.chains import GraphCypherQAChain, RetrievalQA
from langchain_community.vectorstores import Neo4jVector

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
import subprocess
import logging
from langchain_core.documents import Document
from langchain.tools import BaseTool, Tool
from jinja2 import Environment, FileSystemLoader
import chromadb
from chromadb import Settings
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
import uuid
import json
from langchain.prompts.chat import ChatPromptTemplate
from langchain.base_language import BaseLanguageModel
import requests
import re
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

class SubQueries(BaseModel):
    """Model representing a list of sub-queries."""
    queries: List[str] = Field(
        description="List of sub-queries that break down the original question",
        min_items=1
    )

class GraphKnowledgeManager:
    def __init__(
        self, 
        name: str, 
        level: str,
        prompt_dir: str,
        aura_instance_id: str,  # Add this parameter
        aura_instance_name: str,  # Add this parameter
        neo4j_uri: str,  # Add this parameter
        neo4j_username: str,  # Add this parameter
        neo4j_password: str,  # Add this parameter
        llm_models: str = "gpt-4", 
        cypher_llm_model: str = "gpt-4",
        qa_llm_model: str = "gpt-4",
        cypher_llm_params: Dict[str, Any] = None,
        qa_llm_params: Dict[str, Any] = None,
        chain_verbose: bool = True,
        chain_callback_manager: Optional[Any] = None,
        chain_memory: Optional[Any] = None,
        similarity_threshold: float = 0.85,
        max_iterations: int = 5,
        execution_timeout: int = 30,
        max_retries: int = 3,
        return_intermediate_steps: bool = False,
        handle_retries: bool = True,
        allowed_nodes: Optional[List[str]] = None,
        allowed_relationships: Optional[List[str]] = None,
        strict_mode: bool = False,
        node_properties: Union[bool, List[str]] = True,
        relationship_properties: Union[bool, List[str]] = True,
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
        
        # Store Neo4j instance details
        self.aura_instance_id = aura_instance_id
        self.aura_instance_name = aura_instance_name
        self.neo4j_uri = neo4j_uri
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password
        
        self.neo4j_graph = None
        self.jinja_env = Environment(loader=FileSystemLoader(os.path.join(self.prompt_dir)))
        self.graph_system_prompt = None
        self.schema = None
        
        self.ensure_connection()
        
        # Initialize GraphCypherQAChain with default values
        self.cypher_chain = None
        
        # Initialize LLMGraphTransformer        
        self.llm_transformer = None

    def _escape_single_brackets(self, text: str) -> str:
        """
        Replace single brackets with double brackets to escape them properly.
        Ignores already doubled brackets.
        """
        # Replace single { with {{ if not already {{
        text = re.sub(r'(?<!\{)\{(?!\{)', '{{', text)
        # Replace single } with }} if not already }}
        text = re.sub(r'(?<!\})\}(?!\})', '}}', text)
        return text

    def _create_query_prompt_template(self, context: List[str]) -> ChatPromptTemplate:
        # First ensure schema is loaded
        
        template = f"""T    Task:Generate Cypher statement to query a graph database.
        Instructions:
        Use only the provided relationship types and properties in the schema.
        Do not use any other relationship types or properties that are not provided.

        Note: Do not include any explanations or apologies in your responses.
        Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
        Do not include any text except the generated Cypher statement.
        
        A context is provided from a vector search in a form of tuple ('a..', 'W..') 
        Use the second element of the tuple as a node id, e.g 'W..... 
        Here are the contexts: {context}

        Using node id from the context above, create cypher statements and use that to query with the graph.
            """
        return ChatPromptTemplate.from_template(template)
        

    def _construct_llm(self, llm_name: str, llm_params: Dict[str, Any]) -> BaseLanguageModel:
        """Construct the appropriate LLM based on the input string and parameters."""
        OPENAI_MODELS = ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]
        MISTRAL_MODELS = ["mistral-tiny", "mistral-small", "mistral-medium"]
        COHERE_MODELS = ["command", "command-light", "command-nightly"]
        GROQ_MODELS = ["llama2-70b-4096", "mixtral-8x7b-32768"]
        VERTEXAI_MODELS = ["chat-bison", "chat-bison-32k"]
        OLLAMA_MODELS = ["llama2", "mistral", "dolphin-phi"]
        NVIDIA_MODELS = ["mixtral-8x7b", "llama2-70b"]
        ANTHROPIC_MODELS = ["claude-2", "claude-instant-1"]
        FIREWORKS_MODELS = ["llama-v2-7b", "llama-v2-13b", "llama-v2-70b"]

        if llm_name in OPENAI_MODELS:
            return ChatOpenAI(model_name=llm_name, **llm_params)
        elif llm_name in MISTRAL_MODELS:
            return ChatMistralAI(model=llm_name, **llm_params)
        elif llm_name in COHERE_MODELS:
            return ChatCohere(model=llm_name, **llm_params)
        elif llm_name in GROQ_MODELS:
            return ChatGroq(model=llm_name, **llm_params)
        elif llm_name in VERTEXAI_MODELS:
            return ChatVertexAI(model_name=llm_name, **llm_params)
        elif llm_name in OLLAMA_MODELS:
            return ChatOllama(model=llm_name, **llm_params)
        elif llm_name in NVIDIA_MODELS:
            return ChatNVIDIA(model=llm_name, **llm_params)
        elif llm_name in ANTHROPIC_MODELS:
            return ChatAnthropic(model=llm_name, **llm_params)
        elif llm_name in FIREWORKS_MODELS:
            return ChatFireworks(model=llm_name, **llm_params)
        else:
            raise ValueError(f"Unsupported model: {llm_name}")

    def ensure_connection(self):
        """Connect to the existing Neo4j instance."""

        # Connect using Neo4jGraph for LangChain operations
        self.neo4j_graph = Neo4jGraph(
            url=self.neo4j_uri,
            username=self.neo4j_username,
            password=self.neo4j_password,
            #enhanced_schema=True  
        )
        logger.info(f"Connected to Neo4j Aura instance: {self.aura_instance_name}")


    def query_graph(self, question: str, context: Optional[str] = None) -> Optional[Any]:
        """Query the graph database with better error handling and logging"""
        try:
            vector_search_result = self.vector_search(question)
            context = [vector_search_result["article_ids"], vector_search_result["documents"]]
            # Create custom prompt template
            prompt = self._create_query_prompt_template(context)
            
            if self.cypher_chain is None:
                self.cypher_chain = GraphCypherQAChain.from_llm(
                    cypher_llm=self._construct_llm(self.cypher_llm_model, self.cypher_llm_params),
                    qa_llm=self._construct_llm(self.qa_llm_model, self.qa_llm_params),
                    graph=self.neo4j_graph,
                    verbose=self.chain_verbose,
                    prompt=prompt,
                    return_intermediate_steps=True,
                    validate_cypher=True,
                    top_k=10,
                    allow_dangerous_requests=True
                )

            # Execute query
            response = self.cypher_chain.invoke({
                "query": question,
            })
            
            # Log intermediate steps for debugging
            if "intermediate_steps" in response:
                logger.debug(f"Generated Cypher: {response['intermediate_steps']}")
                
            logger.info(f"Query result: {response}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error executing GraphCypherQAChain: {e}")
            return None

    async def populate_knowledge_graph(self, texts: List[str], batch_size: int = 100):
        self.graph_system_prompt = self._load_graph_system_prompt()
        graph_prompt = ChatPromptTemplate.from_template(self.graph_system_prompt)

        if self.llm_transformer is None:
            self.llm_transformer = LLMGraphTransformer(
                llm=self._construct_llm(self.llm_models, self.llm_params),
                allowed_nodes=self.allowed_nodes,
                allowed_relationships=self.allowed_relationships,
                #prompt=graph_prompt,
                strict_mode=self.strict_mode,
                node_properties=self.node_properties,
                relationship_properties=self.relationship_properties,
                ignore_tool_usage=self.ignore_tool_usage
                
            )

        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                documents = [Document(page_content=text) for text in batch]
                graph_documents = await self.llm_transformer.aconvert_to_graph_documents(documents)
                
                # Log the graph documents before adding them
                logger.info(f"Converting batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                logger.info(f"Generated graph documents: {graph_documents}")
                
                # Add explicit node properties
                for doc in graph_documents:
                    if hasattr(doc, 'nodes') and doc.nodes:  # Check if doc has nodes attribute
                        for node in doc.nodes:
                            if not hasattr(node, 'properties'):
                                node.properties = {}
                            if hasattr(node, 'id'):
                                node.properties['name'] = node.id
                
                self.neo4j_graph.add_graph_documents(graph_documents, include_source=True)
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            self.neo4j_graph.refresh_schema()
        except Exception as e:
            logger.error(f"Error in populate_knowledge_graph: {e}")
            raise

    def delete_node_or_relationship(self, identifier: str):
        try:
            self.neo4j_graph.query(f"MATCH (n) WHERE n.id = '{identifier}' DETACH DELETE n")
            logger.info(f"Deleted node or relationship with identifier '{identifier}'")
        except Exception as e:
            logger.error(f"Error deleting node or relationship: {e}")

    def delete_database(self):
        """Delete all nodes and relationships in the database."""
        try:
            with GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_username, self.neo4j_password)) as driver:
                with driver.session() as session:
                    session.run("MATCH (n) DETACH DELETE n")
            logger.info("Deleted all nodes and relationships in the database.")
        except Exception as e:
            logger.error(f"Error deleting database: {e}")

    def query_graph_tool(self) -> BaseTool:
        """Create a tool for querying the graph database."""
        return Tool(
            name="query_graph",
            description="Query the Neo4j graph database with natural language questions",
            func=self.query_graph
        )

    def populate_knowledge_graph_tool(self) -> BaseTool:
        """Create a tool for populating the knowledge graph."""
        return Tool(
            name="populate_knowledge_graph",
            description="Add new information to the knowledge graph from text",
            func=self.populate_knowledge_graph
        )

    def delete_node_or_relationship_tool(self) -> BaseTool:
        """Create a tool for deleting nodes or relationships."""
        return Tool(
            name="delete_node_or_relationship",
            description="Delete a specific node or relationship from the graph",
            func=self.delete_node_or_relationship
        )

    def delete_database_tool(self) -> BaseTool:
        """Create a tool for deleting the entire database."""
        return Tool(
            name="delete_database",
            description="Delete all nodes and relationships in the database",
            func=self.delete_database
        )

    def vector_search_tool(self) -> BaseTool:
        """Create a tool for vector similarity search."""
        return Tool(
            name="vector_search",
            description="Search the knowledge graph using vector similarity",
            func=self.vector_search
        )

    def get_tools(self) -> List[BaseTool]:
        """Get all available tools."""
        return [
            self.query_graph_tool(),
            self.populate_knowledge_graph_tool(),
            self.delete_node_or_relationship_tool(),
            self.delete_database_tool(),
            self.vector_search_tool(),
        ]

    def _load_graph_system_prompt(self):
        template = self.jinja_env.get_template('graph_system_prompt.j2')
        return template.render()



    def disambiguate(self):
        """
        Resolve duplicate entities and relationships in the graph by identifying and merging
        nodes and relationships that represent the same concepts.
        """
        try:
            # First connect to the specific database
            logger.info(f"Starting disambiguation process for database: {self.aura_instance_name}")
            
            # Get all nodes
            nodes_query = """
            MATCH (n) 
            RETURN DISTINCT n.name as name, labels(n) as labels, 
            properties(n) as properties
            """
            nodes = self.neo4j_graph.query(nodes_query)
            
            # Get all relationships
            rels_query = """
            MATCH ()-[r]->() 
            RETURN DISTINCT type(r) as type, 
            startNode(r).name as start_name, 
            endNode(r).name as end_name,
            properties(r) as properties
            """
            relationships = self.neo4j_graph.query(rels_query)

            # Process nodes in batches
            batch_size = 15  # Adjust based on your needs
            node_batches = [nodes[i:i+batch_size] for i in range(0, len(nodes), batch_size)]
            
            for i, node_batch in enumerate(node_batches, 1):
                self._merge_similar_nodes(node_batch)
                logger.info(f"Processed node batch {i}/{len(node_batches)}")

            # Process relationships in batches
            rel_batches = [relationships[i:i+batch_size] for i in range(0, len(relationships), batch_size)]
            
            for i, rel_batch in enumerate(rel_batches, 1):
                self._merge_similar_relationships(rel_batch)
                logger.info(f"Processed relationship batch {i}/{len(rel_batches)}")

        except Exception as e:
            logger.error(f"Error during disambiguation: {e}")
            raise

    def _merge_similar_nodes(self, nodes):
        """Merge nodes that represent the same entity."""
        try:
            # Create a prompt for the LLM to identify similar nodes
            node_data = [
                {
                    "name": node["name"],
                    "labels": node["labels"],
                    "properties": node["properties"]
                }
                for node in nodes if node["name"]  # Filter out nodes without names
            ]
            
            if not node_data:
                return

            messages = [
                SystemMessage(content="""You are a data processing assistant specialized in identifying duplicate entities in Neo4j graphs.
                Your task is to analyze nodes and identify which ones represent the same real-world entity despite having different representations.

                Rules for identifying duplicates:
                1. Consider nodes with minor spelling variations or typographical differences as duplicates
                   Example: "John Smith" and "Jon Smith" might be the same person
                2. Consider nodes with different formats but same semantic meaning as duplicates
                   Example: "USA" and "United States of America" refer to the same country
                3. Consider nodes that refer to the same real-world entity as duplicates, even if described differently
                   Example: "NYC" and "New York City" refer to the same place
                4. Do NOT merge nodes if they represent:
                   - Different time periods or dates
                   - Different numerical values
                   - Different specific instances of similar things
                   Example: "Report 2023" and "Report 2024" should remain separate

                Return your response as a JSON array where each element is an array of duplicate nodes.
                The first node in each array should be the canonical form (the preferred version to keep).
                Only include nodes that have duplicates - ignore unique nodes.
                """),
                HumanMessage(content=f"Analyze these nodes for duplicates:\n{json.dumps(node_data, indent=2)}")
            ]

            # Get LLM response
            response = self.cypher_llm.invoke(messages)
            merge_groups = json.loads(response.content)

            # Merge similar nodes
            for group in merge_groups:
                if len(group) > 1:
                    primary = group[0]
                    for secondary in group[1:]:
                        merge_query = """
                        MATCH (primary {name: $primary_name}), (secondary {name: $secondary_name})
                        CALL apoc.merge.nodes([primary, secondary]) YIELD node
                        RETURN node
                        """
                        self.neo4j_graph.query(
                            merge_query,
                            {"primary_name": primary["name"], "secondary_name": secondary["name"]}
                        )
                        logger.info(f"Merged node '{secondary['name']}' into '{primary['name']}'")

        except Exception as e:
            logger.error(f"Error merging similar nodes: {e}")
            raise

    def _merge_similar_relationships(self, relationships):
        """Merge relationships that represent the same connection."""
        try:
            # Create a prompt for the LLM to identify similar relationships
            rel_data = [
                {
                    "type": rel["type"],
                    "start": rel["start_name"],
                    "end": rel["end_name"],
                    "properties": rel["properties"]
                }
                for rel in relationships
            ]
            
            if not rel_data:
                return

            messages = [
                SystemMessage(content="""
                    Identify relationships that represent the same connection but are written differently.
                    Return a list of groups where each group contains similar relationships.
                    The first relationship in each group should be the canonical form to keep.
                """),
                HumanMessage(content=f"Analyze these relationships:\n{json.dumps(rel_data, indent=2)}")
            ]

            # Get LLM response
            response = self.cypher_llm.invoke(messages)
            merge_groups = json.loads(response.content)

            # Merge similar relationships
            for group in merge_groups:
                if len(group) > 1:
                    primary = group[0]
                    for secondary in group[1:]:
                        merge_query = """
                        MATCH (s1 {name: $primary_start})-[r1:$primary_type]->(e1 {name: $primary_end}),
                              (s2 {name: $secondary_start})-[r2:$secondary_type]->(e2 {name: $secondary_end})
                        CALL apoc.merge.relationships([r1, r2]) YIELD rel
                        RETURN rel
                        """
                        self.neo4j_graph.query(
                            merge_query,
                            {
                                "primary_type": primary["type"],
                                "primary_start": primary["start"],
                                "primary_end": primary["end"],
                                "secondary_type": secondary["type"],
                                "secondary_start": secondary["start"],
                                "secondary_end": secondary["end"]
                            }
                        )
                        logger.info(f"Merged relationship '{secondary['type']}' into '{primary['type']}'")

        except Exception as e:
            logger.error(f"Error merging similar relationships: {e}")
            raise

    def decompose_user_query(self, question: str) -> List[str]:
        """
        Decompose a complex user query into simpler sub-queries using structured output.
        """
        # Construct dedicated LLM for query decomposition
        decomposition_llm = self._construct_llm("gpt-4o-mini", {
            "temperature": 0.1,
            "max_tokens": 500
        })
        
        # Bind the LLM with the SubQueries tool
        llm_with_structured_output = decomposition_llm.with_structured_output(SubQueries)
        
        messages = [
            SystemMessage(content="""You are an expert at converting user questions into Neo4j Cypher queries.
            Break down complex questions into simpler sub-queries that can be executed sequentially.
            Each sub-query should focus on a specific aspect of the original question.
            
            Example:
            Question: "What companies are investing in AI and what are their market trends?"
            Sub-queries:
            1. "Find companies investing in AI technology"
            2. "What are the market trends for companies"
            """),
            HumanMessage(content=f"Break this question into sub-queries: {question}")
        ]
        
        try:
            # Get structured response from LLM
            response = llm_with_structured_output.invoke(messages)
            
            logger.info(f"Decomposed query '{question}' into: {response.queries}")
            return response.queries
            
        except Exception as e:
            logger.error(f"Error decomposing query: {e}")
            return [question]  # Return original question if decomposition fails

    def vector_search(self, query: str, k: int = 3) -> Dict[str, Any]:
        """Improved vector search with better error handling"""
        try:
            # Initialize embeddings
            embeddings = OpenAIEmbeddings()
            
            # Create vector store with specific node labels and properties
            vector_store = Neo4jVector.from_existing_graph(
                embedding=embeddings,
                url=self.neo4j_uri,
                username=self.neo4j_username,
                password=self.neo4j_password,
                index_name=f"{self.name}_vector_index",
                node_label="Geographic_region",  # Search across all labels
                text_node_properties=["name", "content", "description", "location", "role"],
                embedding_node_property="embedding",
                search_type="hybrid",  # Enable hybrid search
                graph=self.neo4j_graph
            )
            
            # Perform similarity search directly first
            search_results = vector_store.similarity_search_with_score(
                query,
                k=k
            )
            
            logger.debug(f"Vector search results: {search_results}")
            
            # Create retrieval chain
            retrieval_chain = RetrievalQA.from_chain_type(
                llm=self._construct_llm(self.qa_llm_model, self.qa_llm_params),
                chain_type="stuff",
                retriever=vector_store.as_retriever(
                    search_kwargs={'k': k}
                ),
                return_source_documents=True
            )
            
            # Execute search
            results = retrieval_chain.invoke({"query": query})
            
            # Convert the result to a list of DocumentModel instances
            documents = [DocumentModel(**doc.dict()) for doc in results['source_documents']]
            extracted_data = [{"title": doc.extract_title(), "article_id": doc.metadata.article_id} for doc in documents]
            article_ids = [("article_id", doc.metadata.article_id) for doc in documents]
    
            logger.info(f"Vector search completed for query: {query}")
            return {"article_ids": article_ids, "documents": extracted_data, "question":query}

        except Exception as e:
            logger.error(f"Error in vector search: {e}", exc_info=True)
            raise



# Test the code
def main():
    from langchain.agents import AgentExecutor, create_openai_tools_agent
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    
    prompt_dir = "Data/Prompts/level1/Datavor"
    name = "agent1"
    load_dotenv()

    # Get environment variables
    aura_instance_id = os.getenv('AURA_INSTANCE_ID')
    aura_instance_name = os.getenv('AURA_INSTANCENAME')
    neo4j_uri = os.getenv('NEO4J_URI')
    neo4j_username = os.getenv('NEO4J_USERNAME')
    neo4j_password = os.getenv('NEO4J_PASSWORD')   

    # Initialize GraphKnowledgeManager
    gkm = GraphKnowledgeManager(
        name=name,
        level="level1",
        prompt_dir=prompt_dir,
        temperature=0.2,
        max_tokens=4000,
        aura_instance_id=aura_instance_id,
        aura_instance_name=aura_instance_name,
        neo4j_uri=neo4j_uri,
        neo4j_username=neo4j_username,
        neo4j_password=neo4j_password,
        allowed_nodes=[
            "Company", "Market_Segment", "Product", "Technology",
            "Consumer_Group", "Regulatory_Body", "Market_Trend",
            "Statistic", "Timeline", "Investment", "Clinical_Trial",
            "Patient_Outcome", "Healthcare_Provider", "Research_Institution",
            "Geographic_Region", "Cost", "Regulation"
        ],
        allowed_relationships=[
            "AFFECTS", "CORRELATES_WITH", "COMPETES_WITH", "REGULATES",
            "COLLABORATES_WITH", "SUPPLIES_TO", "INVESTS_IN", "RESEARCHES",
            "OPERATES_IN", "GROWS_BY", "COMPLIES_WITH", "INTEGRATES_WITH",
            "RESULTS_IN", "BELONGS_TO", "PROVIDES_SERVICE_TO", "IMPLEMENTS"
        ],
        node_properties=True,
        relationship_properties=True,
        chain_verbose=True
    )

    # Initialize LangChain components
    llm = ChatOpenAI(temperature=0.2, model="gpt-4")
    
    # Create simple prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant that uses tools to interact with a knowledge graph."),
        ("human", "{input}"),
        ("human", "Assistant: Let me help you with that. {agent_scratchpad}")

    ])

    # Create tools list
    tools = gkm.get_tools()

    # Create agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True
    )

    # Test single query
    test_query = "What are the main challenges facing the German MedTech industry?"
    print(f"\nTesting query: {test_query}")
    try:
        response = agent_executor.invoke({"input": test_query, "agent_scratchpad": "    "})
        print(f"Response: {response['output']}")
    except Exception as e:
        print(f"Error processing query: {e}")

    test_text = """put this knowledge into the graph: the German MedTech industry is facing challenges in the areas of regulation, reimbursement, and market access. The industry is also facing challenges in the areas of regulation, reimbursement, and market access.
    the main challenges are with the following companies and regulatory bodies:
    """
    print(f"\nTesting graph population with new text: {test_text}")
    try:
        response = agent_executor.invoke({"input": test_text, "agent_scratchpad": "    "})
        print(f"Response: {response['output']}")
    except Exception as e:
        print(f"Error processing query: {e}")
        
if __name__ == "__main__":
    main()

