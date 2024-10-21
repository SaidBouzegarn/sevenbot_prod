import streamlit as st
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext, ServiceContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.mistralai import MistralAI
import chromadb
from chromadb.utils import embedding_functions

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
index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

# Create the query engine
query_engine = index.as_query_engine()

# Streamlit app
st.set_page_config(page_title="7bot", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Start", "View Graph DB"])

# Home page
if page == "Home":
    st.markdown("<h1 style='text-align: center; font-size: 72px;'>7bot</h1>", unsafe_allow_html=True)

# Start page
elif page == "Start":
    st.title("Start")
    upload_option = st.radio("Choose upload method:", ["Upload S3 Links", "Upload URLs"])
    
    if upload_option == "Upload URLs":
        # Call scrape function
        scraped_data = scrape()  # Assume scrape() is defined elsewhere and returns JSON-like data
        st.write("Scraped Data:", scraped_data)
    
    if st.button("Populate Vector DB"):
        # Call function to add data to vector DB
        populate_vector_db(scraped_data)  # Assume populate_vector_db() is defined elsewhere
    
    if st.button("Populate Graph"):
        # Call function to add data to graph DB
        populate_graph_db(scraped_data)  # Assume populate_graph_db() is defined elsewhere

# View Graph DB page
elif page == "View Graph DB":
    st.title("Graph DB Viewer")
    # Display and interact with the graph DB
    # Assume view_graph_db() is defined elsewhere
    view_graph_db()

    # Chatbot interface
    st.sidebar.title("Chat with LLaMA")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to know?"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            retrieval_results = query_engine.retrieve(prompt)
            context = ""
            if retrieval_results:
                context = "Based on the following information:\n\n"
                for i, result in enumerate(retrieval_results[:3], 1):
                    context += f"{i}. {result.node.get_content()}\n\n"
            
            llm_prompt = f"{context}Human: {prompt}\n\nAssistant: Let me answer your question based on the information provided and my general knowledge."
            llm_response = llm.complete(llm_prompt)
            
            with st.chat_message("assistant"):
                st.markdown(llm_response.text)
            
            st.session_state.messages.append({"role": "assistant", "content": llm_response.text})
        
        except Exception as e:
            st.error(f"An error occurred while processing your query: {str(e)}")
            st.exception(e)
