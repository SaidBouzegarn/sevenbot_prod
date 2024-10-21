import os
from py2neo import Graph
from dotenv import load_dotenv, set_key
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate  # Updated import pathfrom langchain_core.output_parsers import StrOutputParser
from langchain_experimental.graph_transformers import LLMGraphTransformer
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import subprocess
import boto3
from datetime import datetime
import tempfile
import json  # Ensure this import is present
import logging
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DuplicateEntities(BaseModel):
    entities: List[str] = Field(
        description="List of at least 2 entities or entity relationships that represent the same object or real-world entity and should be merged. The entity or relationship name to keep should be the first one in the list.",
        min_items=2
    )   
    


class Disambiguate(BaseModel):
    merge_entities: Optional[List[DuplicateEntities]] = Field(
        description="Lists of entities that represent the same object or real-world entity and should be merged"
    )


class KnowledgeGraphManager:
    def __init__(self, db_name: str, graph_name: str, llm_model: str = "gpt-4-turbo", **llm_params):
        load_dotenv()
        self.db_name = db_name
        self.graph_name = graph_name
        self.llm_model = llm_model
        self.llm_params = llm_params
        self.llm = self._create_llm()
        self.neo4j_graph = None

    def _create_llm(self):
        return ChatOpenAI(model_name=self.llm_model, **self.llm_params)

    def update_llm_config(self, model: str = None, **params):
        """
        Update the LLM configuration with new parameters.
        
        :param model: New model name to use (optional)
        :param params: Additional parameters to update or add
        """
        if model:
            self.llm_model = model
        self.llm_params.update(params)
        self.llm = self._create_llm()
        print(f"LLM configuration updated. Model: {self.llm_model}, Params: {self.llm_params}")

    def create_aura_instance(self):
        """Create a Neo4j Aura instance using the Aura CLI."""
        try:
            # Add credentials to Aura CLI
            subprocess.run(
                ["aura", "credentials", "add", "--name", "my-credentials", "--client-id", os.getenv('AURA_CLIENT_ID'), "--client-secret", os.getenv('AURA_CLIENT_SECRET'), "--use"],
                capture_output=True, text=True, check=True
            )
            # Create the Aura instance
            result = subprocess.run(
                ["aura", "instances", "create", "--name", self.db_name, "--cloud-provider", "gcp", "--region", "europe-west1", "--type", "free"],
                capture_output=True, text=True, check=True
            )
            print("Aura instance created successfully.")
            self._parse_and_save_connection_details(result.stdout)
            self.connect_to_neo4j_aura()
        except subprocess.CalledProcessError as e:
            print(f"Error creating Aura instance: {e}")
            print(f"Command output: {e.stdout}")
            print(f"Error output: {e.stderr}")
            raise

    def delete_aura_instance(self):
        """Delete the Neo4j Aura instance."""
        try:
            result = subprocess.run(
                ["aura", "instances", "delete", "--name", self.db_name, "--force"],
                capture_output=True, text=True, check=True
            )
            print("Aura instance deleted successfully.")
            # Clear environment variables
            os.environ.pop('NEO4J_AURA_URI', None)
            os.environ.pop('NEO4J_AURA_USERNAME', None)
            os.environ.pop('NEO4J_AURA_PASSWORD', None)
            # Update .env file
            env_file_path = '.env'
            set_key(env_file_path, 'NEO4J_AURA_URI', '')
            set_key(env_file_path, 'NEO4J_AURA_USERNAME', '')
            set_key(env_file_path, 'NEO4J_AURA_PASSWORD', '')
            print(f"Neo4j Aura connection details removed from '{env_file_path}'.")
            self.neo4j_graph = None
        except subprocess.CalledProcessError as e:
            print(f"Error deleting Aura instance: {e}")
            print(f"Command output: {e.stdout}")
            print(f"Error output: {e.stderr}")
            raise

    def _parse_and_save_connection_details(self, output):
        """Parse the CLI output and save connection details to .env file."""
        lines = output.split('\n')
        uri = next((line.split(': ')[1] for line in lines if 'Connection URI' in line), None)
        username = next((line.split(': ')[1] for line in lines if 'Username' in line), None)
        password = next((line.split(': ')[1] for line in lines if 'Password' in line), None)

        if not all([uri, username, password]):
            raise ValueError("Failed to parse all required connection details from CLI output.")

        env_file_path = '.env'
        set_key(env_file_path, 'NEO4J_URI', uri)
        set_key(env_file_path, 'NEO4J_USERNAME', username)
        set_key(env_file_path, 'NEO4J_PASSWORD', password)
        print(f"Neo4j Aura connection details saved to '{env_file_path}'.")

    def connect_to_neo4j_aura(self):
        """Connect to Neo4j Aura instance using environment variables."""
        aura_uri = os.getenv('NEO4J_URI')
        aura_username = os.getenv('NEO4J_USERNAME')
        aura_password = os.getenv('NEO4J_PASSWORD')

        if not all([aura_uri, aura_username, aura_password]):
            raise ValueError("Neo4j Aura connection details are missing in the .env file.")

        self.neo4j_graph = Neo4jGraph(
            url=aura_uri,
            username=aura_username,
            password=aura_password
        )
        print("Connected to Neo4j Aura instance.")

    def query_graph(self, query, parameters=None):
        """Execute a Cypher query on the Neo4j Aura instance."""
        if not self.neo4j_graph:
            raise ValueError("Not connected to a Neo4j Aura instance. Call connect_to_neo4j_aura() first.")
        return self.neo4j_graph.run(query, parameters)

    def populate_knowledge_graph(self, texts: List[str], batch_size: int = 100, **kwargs):
        """
        Populate the Neo4j knowledge graph with information extracted from the given texts.
        Process texts in batches to manage memory and API usage.
        """
        llm_transformer = LLMGraphTransformer(llm=self.llm, **kwargs)

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            documents = [Document(page_content=text) for text in batch]
            graph_documents = llm_transformer.convert_to_graph_documents(documents)
            
            # Store graph documents to the graph database
            self.neo4j_graph.add_graph_documents(graph_documents)

            print(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")

    def resolve_entities(self, batch_size: int = 15):
        """
        Resolve duplicate entities in the graph using LLM.
        Process entities in batches to manage memory and API usage.
        """
        entities = self._get_entities()
        relationships = self._get_relationships()

        entity_batches = [entities[i:i+batch_size] for i in range(0, len(entities), batch_size)]
        relationship_batches = [relationships[i:i+batch_size] for i in range(0, len(relationships), batch_size)]

        for i, entity_batch in enumerate(entity_batches, 1):
            self._resolve_entity_batch(entity_batch)
            print(f"Processed entity batch {i}/{len(entity_batches)}")

        for i, relationship_batch in enumerate(relationship_batches, 1):
            self._resolve_relationship_batch(relationship_batch)
            print(f"Processed relationship batch {i}/{len(relationship_batches)}")

    def _get_entities(self):
        """Retrieve all entities from the graph."""
        return self.neo4j_graph.query("MATCH (n) RETURN n")  # This already returns a list of dicts

    def _get_relationships(self):
        """Retrieve all relationships from the graph."""
        return self.neo4j_graph.query("MATCH ()-[r]->() RETURN r")  # This already returns a list of dicts

    def _resolve_entity_batch(self, entities):
        """Resolve a batch of entities using LLM."""
        # Extract entity names, handling potential missing keys
        entity_names = [
            entity["n"].get("name", str(entity["n"])) for entity in entities if "n" in entity
        ]

        if not entity_names:
            logger.info("No valid entities found in this batch.")
            logger.info(f"Original entities attempted: {entities}")
            return

        prompt = self._create_entity_resolution_prompt(entity_names)
        messages = prompt.format_messages(
            entities=json.dumps(entity_names, ensure_ascii=False, indent=2)
        )

        structured_llm = self.llm.with_structured_output(Disambiguate)
        response = structured_llm.invoke(messages)
        duplicates = response.merge_entities
        self._merge_duplicate_entities(duplicates, entity_names)

    def _resolve_relationship_batch(self, relationships):
        """Resolve a batch of relationships using LLM."""
        print("Debugging: Relationships received:")
        for i, rel in enumerate(relationships):
            print(f"Relationship {i}: {rel}")
        
        print("\nDebugging: Attempting to extract relationship types:")
        relationship_types = []
        for i, rel in enumerate(relationships):
            try:
                # The relationship type is the second element of the tuple
                rel_type = rel['r'][1]
                relationship_types.append(rel_type)
                print(f"Relationship {i} type: {rel_type}")
            except Exception as e:
                print(f"Error processing relationship {i}: {e}")
                print(f"Relationship {i} content: {rel}")

        if not relationship_types:
            print("No valid relationships found in this batch.")
            print(f"Original relationships attempted: {relationships}")
            return

        prompt = self._create_relationship_resolution_prompt(relationship_types)
        messages = prompt.format_messages(
            relationships=json.dumps(relationship_types, ensure_ascii=False, indent=2)
        )
        structured_llm = self.llm.with_structured_output(Disambiguate)
        response = structured_llm.invoke(messages)
        duplicates = response.merge_entities
        self._merge_duplicate_relationships(duplicates, relationship_types)

    def _create_entity_resolution_prompt(self, entities):
        """Create a prompt for entity resolution."""
        system_prompt = """You are a data processing assistant. Your task is to identify duplicate entities in a list and decide which of them should be merged.
The entities might be slightly different in format or content, but essentially refer to the same thing. Use your analytical skills to determine duplicates.

Here are the rules for identifying duplicates:
1. Entities with minor typographical differences should be considered duplicates.
2. Entities with different formats but the same content should be considered duplicates.
3. Entities that refer to the same real-world object or concept, even if described differently, should be considered duplicates.
4. If it refers to different numbers, dates, or products, do not merge results
"""

        user_template = """Here is the list of entities to process:
```json
{entities}
```

Please identify duplicates, merge them, and provide the merged list.
"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", user_template)
        ])

        # Serialize entities to a JSON-formatted string to avoid format issues
        #serialized_entities = json.dumps(entities, ensure_ascii=False, indent=2)

        # Format the prompt with the serialized_entities
        #prompt = prompt.format_prompt(entities=serialized_entities)

        return prompt

    def _create_relationship_resolution_prompt(self, relationships):
        """Create a prompt for relationship resolution."""
        system_prompt = """You are a data processing assistant. Your task is to identify duplicate relationships in a list and decide which of them should be merged.
The relationships might be slightly different in format or content, but essentially refer to the same thing. Use your analytical skills to determine duplicates.

Here are the rules for identifying duplicates:
1. Relationships with minor typographical differences should be considered duplicates.
2. Relationships with different formats but the same content should be considered duplicates.
3. Relationships that refer to the same real-world object or concept, even if described differently, should be considered duplicates.
4. If it refers to different numbers, dates, or products, do not merge results
"""

        user_template = """Here is the list of relationships to process:
```json
{relationships}
```

Please identify duplicates, merge them, and provide the merged list.
"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", user_template)
        ])

        # Serialize relationships to a JSON-formatted string to avoid format issues
        #serialized_relationships = json.dumps(relationships, ensure_ascii=False, indent=2)

        # Format the prompt with the serialized_relationships
        #prompt = prompt.format_prompt(relationships=serialized_relationships)

        return prompt

    def _merge_duplicate_entities(
        self, duplicates: Optional[List[DuplicateEntities]], original_entities: List[str]
    ):
        """Merge duplicate entities in the Neo4j graph."""
        if not duplicates:
            logger.info(
                f"No duplicates found to merge. Original entities provided: {original_entities}"
            )
            return

        for duplicate_group in duplicates:
            entities_to_merge = duplicate_group.entities
            if len(entities_to_merge) < 2:
                continue  # Skip if there's only one entity or none
            primary_entity = entities_to_merge[0]
            secondary_entities = entities_to_merge[1:]

            for secondary in secondary_entities:
                try:
                    self.neo4j_graph.run(
                        """
                        MATCH (a {name: $primary}), (b {name: $secondary})
                        CALL apoc.merge.node(['Entity'], {name: $primary}, {name: $primary}) YIELD node as merged
                        WITH merged, b
                        CALL apoc.refactor.mergeNodes([merged, b], {properties: 'combine', mergeRels: true})
                        YIELD node
                        RETURN node
                        """,
                        {"primary": primary_entity, "secondary": secondary},
                    )
                    logger.info(f"Merged entity '{secondary}' into '{primary_entity}'")
                except Exception as e:
                    logger.error(
                        f"Error merging entity '{secondary}' into '{primary_entity}': {str(e)}"
                    )
    def _merge_duplicate_relationships(
        self, duplicates: Optional[List[DuplicateEntities]], original_relationships: List[str]
    ):
        """Merge duplicate relationships in the Neo4j graph."""
        if not duplicates:
            logger.info(
                f"No duplicates found to merge. Original relationships provided: {original_relationships}"
            )
            return

        for duplicate_group in duplicates:
            relationships_to_merge = duplicate_group.entities
            if len(relationships_to_merge) < 2:
                continue  # Skip if there's only one relationship or none
            primary_relationship = relationships_to_merge[0]
            secondary_relationships = relationships_to_merge[1:]

            for secondary in secondary_relationships:
                try:
                    self.neo4j_graph.run(
                        """
                        MATCH ()-[r1:`$primary`]->(), ()-[r2:`$secondary`]->()
                        WITH r1, r2, startNode(r1) AS s1, endNode(r1) AS e1, startNode(r2) AS s2, endNode(r2) AS e2
                        WHERE s1 = s2 AND e1 = e2
                        CALL apoc.refactor.mergeRelationships([r1, r2], {properties: 'combine'})
                        YIELD rel
                        RETURN rel
                        """,
                        {"primary": primary_relationship, "secondary": secondary},
                    )
                    logger.info(
                        f"Merged relationship '{secondary}' into '{primary_relationship}'"
                    )
                except Exception as e:
                    logger.error(
                        f"Error merging relationship '{secondary}' into '{primary_relationship}': {str(e)}"
                    )
    def add_element_summarization(self, batch_size: int = 100):
        """
        Add summaries to elements (nodes and relationships) in the graph.
        Process elements in batches to manage memory and API usage.
        """
        elements = self._get_elements()
        batches = [elements[i:i+batch_size] for i in range(0, len(elements), batch_size)]

        for batch in batches:
            summaries = self._summarize_batch(batch)
            self._add_summaries_to_graph(summaries)

    def _get_elements(self):
        """Retrieve all elements (nodes and relationships) from the graph."""
        return self.neo4j_graph.query("""
        MATCH (n)
        OPTIONAL MATCH (n)-[r]->()
        RETURN n, r
        """).data()  # Ensures it returns a list of dicts

    def _summarize_batch(self, batch):
        """Summarize a batch of elements using LLM."""
        summaries = []
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self._summarize_element, element): element for element in batch}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Summarizing elements"):
                summaries.append(future.result())
        return summaries

    def _summarize_element(self, element):
        """Summarize a single element using LLM."""
        prompt = self._create_summarization_prompt(element).messages
        response = self.llm.invoke(prompt)
        summary_text = response.content  # Assuming response is a message with 'content'
        return {"element": element, "summary": summary_text}

    def _create_summarization_prompt(self, element):
        """Create a prompt for element summarization."""
        system_prompt = "You are a graph element summarization assistant."
        # Serialize the element to a JSON-formatted string
        serialized_element = json.dumps(element, ensure_ascii=False, indent=2)

        user_template = """Summarize the following graph element:
```json
{element}
```

Provide a concise summary that captures the key information and relationships.
"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", user_template)
        ])

        # Format the prompt with the serialized_element
        formatted_prompt = prompt.format_prompt(element=serialized_element)

        return formatted_prompt

    def _add_summaries_to_graph(self, summaries: List[Dict[str, Any]]):
        """Add generated summaries to the graph elements."""
        for summary in summaries:
            element = summary["element"]
            summary_text = summary["summary"]
            try:
                if "n" in element:
                    node_id = element["n"]["id"]  # Adjust based on actual data structure
                    self.neo4j_graph.run(
                        """
                        MATCH (n)
                        WHERE id(n) = $node_id
                        SET n.summary = $summary
                        """,
                        {"node_id": node_id, "summary": summary_text},
                    )
                elif "r" in element:
                    rel_id = element["r"]["id"]  # Adjust based on actual data structure
                    self.neo4j_graph.run(
                        """
                        MATCH ()-[r]->()
                        WHERE id(r) = $rel_id
                        SET r.summary = $summary
                        """,
                        {"rel_id": rel_id, "summary": summary_text},
                    )
                logger.info(
                    f"Added summary to {'node' if 'n' in element else 'relationship'} with ID '{node_id if 'n' in element else rel_id}'"
                )
            except Exception as e:
                logger.error(f"Error adding summary to element: {str(e)}")

    def save_db_image(self):
        """Save an image of the current database state to S3."""
        if self.neo4j_graph is not None:
            try:
                # Get S3 bucket name and graph prefix from environment variables
                bucket_name = os.getenv('BUCKET_NAME')
                graph_prefix = os.getenv('GRAPH_PREFIX')

                if not bucket_name or not graph_prefix:
                    raise ValueError("BUCKET_NAME or GRAPH_PREFIX not found in environment variables.")

                # Generate a unique filename for the database dump
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{graph_prefix}_backup_{timestamp}.dump"

                # Create a temporary file to store the database dump
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_filename = temp_file.name

                # Dump the database to the temporary file
                result = subprocess.run(
                    ["neo4j-admin", "dump", "--database=neo4j", f"--to={temp_filename}"],
                    capture_output=True, text=True, check=True
                )

                # Upload the temporary file to S3
                s3_client = boto3.client('s3')
                s3_client.upload_file(temp_filename, bucket_name, filename)

                # Remove the temporary file
                os.unlink(temp_filename)

                print(f"Database image saved to S3 bucket '{bucket_name}' with key '{filename}'")
            except subprocess.CalledProcessError as e:
                print(f"Error saving database image: {e}")
            except Exception as e:
                print(f"Error uploading to S3: {e}")
        else:
            print("No graph connection available. Please connect to a Neo4j instance first.")

    def resolve_relationships(self, batch_size: int = 500):
        """
        Resolve duplicate relationships in the graph using LLM.
        Process relationships in batches to manage memory and API usage.
        """
        relationships = self._get_relationships()

        relationship_batches = [relationships[i:i+batch_size] for i in range(0, len(relationships), batch_size)]

        for i, relationship_batch in enumerate(relationship_batches, 1):
            self._resolve_relationship_batch(relationship_batch)
            print(f"Processed relationship batch {i}/{len(relationship_batches)}")


# Let's write a main function to test the class methods using some dummy texts
def main():
    db_name = "my-aura-instance"
    graph_name = "my-graph"

    # Initialize with custom LLM parameters
    kgm = KnowledgeGraphManager(db_name, graph_name, temperature=0.2, max_tokens=4000)
    kgm.connect_to_neo4j_aura()
    
    # This will create and connect to a Neo4j Aura instance
    
    insights = [
        "Since 2003, PEPFAR has saved over 25 million lives and significantly reduced HIV infection rates through antiretroviral treatments. It is the leading global funder for HIV prevention and aims to end HIV as a public health threat by 2030.",
        "Dr. Emily Kainne Dokubo emphasizes the role of partnerships in PEPFAR's mission, particularly in addressing disparities in pediatric HIV treatment, as children remain disproportionately affected by HIV/AIDS.",
        "PEPFAR has supported the development of health systems in over 55 countries, helping nations achieve the UNAIDS 95-95-95 targets for HIV testing, treatment, and viral suppression by 2025.",
        "In the 21 years since its establishment, PEPFAR has provided antiretroviral treatments to nearly 21 million people and facilitated the birth of 5.5 million babies free of HIV.",
        "The short-term goal of PEPFAR is to help high HIV-burden countries achieve the UNAIDS 95-95-95 targets by 2025, which will improve individual health and reduce HIV transmission globally.",
        "PEPFAR uses a data-driven approach to prioritize high-burden countries for intervention, predominantly in sub-Saharan Africa, but also extending to Asia, the Caribbean, and Central and South America.",
        "PEPFAR's funding has typically enjoyed bipartisan support, but recent reauthorization for only one year complicates long-term planning. A five-year reauthorization would provide stability for HIV prevention and treatment efforts.",
        "PEPFAR remains committed to serving key populations at increased risk of HIV infection, particularly through a person-centered, differentiated care model that adjusts services to meet individual needs.",
        "Partnerships are essential to PEPFAR's work, including collaborations with other US government agencies, international organizations like WHO and UNAIDS, and private sector entities to develop new HIV prevention tools such as long-acting PrEP.",
        "PEPFAR helps pharmaceutical companies rapidly introduce HIV-related commodities to countries with limited healthcare resources by guaranteeing bulk purchases, facilitating access to new prevention and treatment tools.",
        "A lack of infrastructure and healthcare delivery systems continues to hinder broader access to HIV prevention and treatment, which is why PEPFAR is also focused on health system strengthening, including the training of 346,000 healthcare workers.",
        "Despite successes in adult populations, pediatric HIV remains a challenge, with many children and adolescents lacking access to treatment. PEPFAR has launched initiatives to improve early infant diagnosis and treatment for HIV-positive mothers.",
        "PEPFAR's $20 million Youth Initiative aims to increase HIV awareness and prevention among adolescents, empowering the younger generation to manage their health and prevent new HIV infections.",
        "Although global efforts have transformed HIV from a death sentence to a manageable disease, 1.5 million new infections occur annually, and 600,000 people die from HIV-related complications. Continued efforts are essential to ending HIV as a public health threat by 2030."
    ]
    # Populate the knowledge graph with insights
    kgm.populate_knowledge_graph(insights, batch_size=25)
    
    # Resolve entities to merge potential duplicates
    kgm.resolve_entities(batch_size=10)
    
    # Test resolving relationships
    kgm.resolve_relationships(batch_size=10)

    # Test adding summaries
    kgm.add_element_summarization(batch_size=10)

    # Example query
    result = kgm.query_graph("MATCH (n) RETURN count(n) as node_count")
    print(f"Number of nodes: {result.evaluate()}")
    
    # Delete the Aura instance when done
    kgm.delete_aura_instance()

    # Example of updating LLM configuration
    kgm.update_llm_config(model="gpt-4o-mini", temperature=0.1, top_p=1)

if __name__ == "__main__":
    main()