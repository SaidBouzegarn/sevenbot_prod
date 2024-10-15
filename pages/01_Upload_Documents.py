import streamlit as st
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
import chromadb
from chromadb.utils import embedding_functions
import json

# Load environment variables
load_dotenv()

# Set up OpenAI embedding function
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-ada-002"
)

# Initialize Chroma client and collection
chroma_client = chromadb.Client()
chroma_collection_name = os.getenv("CHROMA_COLLECTION_NAME")
chroma_collection = chroma_client.get_or_create_collection(
    name=chroma_collection_name,
    embedding_function=openai_ef
)

# Set up vector store and index
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(vector_store)

def preprocess_and_add_document(file_content, file_name):
    try:
        # Parse the JSON content
        article_data = json.loads(file_content)
        
        # Create a Document object
        document = Document(
            text=article_data['article_body'],
            metadata={
                'Author': article_data.get('Author', 'Unknown'),
                'date_published': article_data.get('date_published', 'Unknown'),
                'url': article_data.get('url', 'Unknown'),
                'file_name': file_name
            }
        )
        
        # Parse the document into nodes
        parser = SentenceSplitter(chunk_size=512)
        nodes = parser.get_nodes_from_documents([document])
        
        # Add nodes to the index
        index.insert_nodes(nodes)
        
        return True, "Document successfully added to the database."
    except Exception as e:
        return False, f"Error processing document: {str(e)}"

# Streamlit app
st.title("Upload Documents")

# File uploader
uploaded_file = st.file_uploader("Choose a JSON file to upload", type="json")
if uploaded_file is not None:
    # Read file content
    file_content = uploaded_file.read().decode("utf-8")
    
    # Preprocess and add document
    success, message = preprocess_and_add_document(file_content, uploaded_file.name)
    
    if success:
        st.success(message)
    else:
        st.error(message)