import streamlit as st
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext, ServiceContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.llama import LLaMA
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
st.title("Chat with LLaMA")

# Model selection
model_options = ["gpt-3.5-turbo", "gpt-4", "llama", "mistralai"]
selected_model = st.selectbox("Select model:", model_options)

# Set up LLM based on selected model
if selected_model in ["gpt-3.5-turbo", "gpt-4"]:
    llm = OpenAI(model=selected_model)
elif selected_model == "llama":
    llm = LLaMA()
elif selected_model == "mistralai":
    llm = MistralAI()
else:
    st.error("Selected model is not supported.")
    st.stop()

Settings.llm = llm

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What would you like to know?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        # Query the index for relevant documents
        retrieval_results = query_engine.retrieve(prompt)
        
        # Prepare context from retrieved documents
        context = ""
        if retrieval_results:
            context = "Based on the following information:\n\n"
            for i, result in enumerate(retrieval_results[:3], 1):
                context += f"{i}. {result.node.get_content()}\n\n"
        
        # Prepare the prompt for the LLM
        llm_prompt = f"{context}Human: {prompt}\n\nAssistant: Let me answer your question based on the information provided and my general knowledge."
        
        # Generate response using the LLM
        llm_response = llm.complete(llm_prompt)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(llm_response.text)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": llm_response.text})
        
        # Display debug information
        st.sidebar.subheader("Debug Information")
        st.sidebar.write("Retrieved Documents:")
        for i, result in enumerate(retrieval_results[:3], 1):
            st.sidebar.write(f"{i}. {result.node.get_content()[:100]}...")
        
    except Exception as e:
        st.error(f"An error occurred while processing your query: {str(e)}")
        st.exception(e)