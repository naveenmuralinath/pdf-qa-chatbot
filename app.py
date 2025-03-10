import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory
import os
import tempfile
import torch
import time
from dotenv import load_dotenv

#setting HF TOken
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")


# Set Streamlit page configuration
st.set_page_config(page_title="PDF Question Answering Bot", layout="wide")

# Initialize session state variables if they don't exist
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_pdfs" not in st.session_state:
    st.session_state.processed_pdfs = []
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# Function to process an uploaded PDF file
def process_pdf(uploaded_file):
    # Create a temporary file to store uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Load PDF content
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        
        # Split text into chunks for better processing
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
        chunks = text_splitter.split_documents(documents)
        
        # Generate embeddings using a HuggingFace model
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
        
        # Store document chunks in a FAISS vector database
        vectorstore = FAISS.from_documents(chunks, embeddings)
        os.unlink(tmp_path)  # Delete temporary file
        
        return vectorstore
    except Exception as e:
        os.unlink(tmp_path)  # Ensure temp file is deleted in case of failure
        st.error(f"Error processing PDF: {str(e)}")
        return None

# Function to set up a conversational retrieval chain
def setup_conversation_chain(vectorstore):
    # Define the language model
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        HUGGINGFACE_API_TOKEN="HUGGINGFACE_API_TOKEN",
        temperature=0.5, top_p=0.9, top_k=50, repetition_penalty=1.2
    )
    
    # Memory to store chat history
    memory = ConversationSummaryMemory(llm=llm,memory_key="chat_history", return_messages=True)
    
    # Create a conversational retrieval chain
    conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        get_chat_history=lambda h: h,
        verbose=True
    )
    
    return conversation

# Sidebar UI for uploading PDFs
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.processed_pdfs:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    vectorstore = process_pdf(uploaded_file)
                    if vectorstore:
                        if not st.session_state.conversation:
                            st.session_state.conversation = setup_conversation_chain(vectorstore)
                        else:
                            st.session_state.conversation = setup_conversation_chain(vectorstore)
                        
                        st.session_state.processed_pdfs.append(uploaded_file.name)
                        st.success(f"Processed {uploaded_file.name}")
    
    # Display list of processed PDFs
    if st.session_state.processed_pdfs:
        st.write("Processed PDFs:")
        for pdf in st.session_state.processed_pdfs:
            st.write(f"- {pdf}")
    
    # Button to reset the chatbot conversation
    if st.button("Reset Conversation"):
        st.session_state.conversation = None
        st.session_state.chat_history = []
        st.session_state.processed_pdfs = []
        st.session_state.last_query = ""
        time.sleep(0.5)
        st.rerun()

# Main chat interface
st.title("📄 PDF Question Answering Bot")
st.header("Ask any questions about your documents")

# Display chat history
for i, message in enumerate(st.session_state.chat_history):
    if i % 2 == 0:
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**AI:** {message}")

# Input field for user query
user_question = st.text_input("Type your question here:")
submit_button = st.button("Ask")

if submit_button and user_question:
    if not st.session_state.conversation:
        st.warning("Please upload a PDF first.")
    elif user_question == st.session_state.last_query:
        st.warning("You already asked this question.")
    else:
        with st.spinner("Thinking..."):
            # Get response from the conversation model
            response = st.session_state.conversation.run(user_question)
            
            # Update chat history and last query
            st.session_state.chat_history.append(user_question)
            st.session_state.chat_history.append(response)
            st.session_state.last_query = user_question
            st.rerun()

