import os
import asyncio
from typing import List, Dict
import boto3
import logging
from dotenv import load_dotenv, set_key
from chromadb import HttpClient
from chromadb.utils import embedding_functions
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import (
    SentenceSplitter,
    SentenceWindowNodeParser,
    SemanticSplitterNodeParser
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up OpenAI embedding function
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-ada-002"
)

def create_chroma_db(collection_name: str):
    """
    Create a Chroma DB collection and write the collection name to the .env file.

    Args:
        collection_name (str): The name of the Chroma collection to create.
    """
    # Initialize Chroma client
    chroma_client = HttpClient(host='localhost', port=8000)

    # Create Chroma collection
    chroma_collection = chroma_client.create_collection(name=collection_name)
    logger.info(f"Chroma collection '{collection_name}' created successfully.")

    # Write the collection name to the .env file
    env_file_path = '.env'
    set_key(env_file_path, 'CHROMA_COLLECTION_NAME', collection_name)
    logger.info(f"CHROMA_COLLECTION_NAME set to '{collection_name}' in .env file.")

    return chroma_collection

def get_parser(parser_type: str):
    if parser_type == 'SentenceSplitter':
        return SentenceSplitter(chunk_size=512)
    elif parser_type == 'SentenceWindowNodeParser':
        return SentenceWindowNodeParser(chunk_size=512, window_size=100)
    elif parser_type == 'SemanticSplitterNodeParser':
        embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
        return SemanticSplitterNodeParser(chunk_size=512, embed_model=embed_model)
    else:
        logger.warning(f"Unknown parser type: {parser_type}. Using SentenceSplitter by default.")
        return SentenceSplitter(chunk_size=512)

async def add_articles_chroma_db(articles: List[Dict], collection_name: str, parser_type: str = 'SentenceSplitter', batch_size: int = 100):
    """
    Add articles to the Chroma DB collection.

    Args:
        articles (List[Dict]): List of article dictionaries containing 'text' and metadata.
        collection_name (str): Name of the Chroma collection to use.
        parser_type (str): Type of parser to use for text splitting.
        batch_size (int): Number of articles to process in each batch.
    """
    chroma_client = HttpClient(host='localhost', port=8000)
    chroma_collection = chroma_client.get_collection(name=collection_name)
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    parser = get_parser(parser_type)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

    for i in range(0, len(articles), batch_size):
        batch = articles[i:i+batch_size]
        documents = [
            Document(text=article['article_body'], metadata={
            'Author': article['Author'],
            'date_published': article['date_published'],
            'url': article['url']
        })
            for article in batch
        ]


        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            transformations=[parser]
        )

        logger.info(f"Added batch of {len(batch)} articles to Chroma (total: {i + len(batch)})")

    logger.info(f"All articles added to Chroma collection '{collection_name}'")

async def add_insights_chroma_db(insights_data: List[Dict], collection_name: str, batch_size: int = 100):
    """
    Add insights to the Chroma DB collection.

    Args:
        insights_data (List[Dict]): List of dictionaries containing insights and metadata.
        collection_name (str): Name of the Chroma collection to use.
        batch_size (int): Number of insights to process in each batch.
    """
    chroma_client = HttpClient(host='localhost', port=8000)
    chroma_collection = chroma_client.get_collection(name=collection_name)

    for i in range(0, len(insights_data), batch_size):
        batch = insights_data[i:i+batch_size]
        documents = []
        ids = []
        metadatas = []

        for idx, item in enumerate(batch):
            for insight in item['insights']:
                documents.append(insight)
                ids.append(f"{i + idx}_{len(documents)}")
                metadatas.append({
                    'Author': item.get('Author', 'Unknown'),
                    'date_published': item.get('date_published', 'Unknown'),
                    'url': item.get('url', 'Unknown')
                })

        chroma_collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )

        logger.info(f"Added batch of {len(documents)} insights to Chroma (total: {i + len(documents)})")

    logger.info(f"All insights added to Chroma collection '{collection_name}'")

def save_copy_chroma_db_s3(collection_name: str, s3_bucket: str, s3_key: str):
    """
    Save a copy of the Chroma DB to S3.

    Args:
        collection_name (str): Name of the Chroma collection to save.
        s3_bucket (str): Name of the S3 bucket to save to.
        s3_key (str): S3 key (path) to save the DB copy to.
    """
    chroma_client = HttpClient(host='localhost', port=8000)
    chroma_collection = chroma_client.get_collection(name=collection_name)

    # Export the collection data
    collection_data = chroma_collection.export()

    # Initialize S3 client
    s3_client = boto3.client('s3')

    # Upload to S3
    s3_client.put_object(
        Bucket=s3_bucket,
        Key=s3_key,
        Body=collection_data
    )

    logger.info(f"Chroma DB '{collection_name}' saved to S3: s3://{s3_bucket}/{s3_key}")