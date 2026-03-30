import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings # Free alternative for embeddings

def process_pdfs(uploaded_files):
    all_docs = []
    for uploaded_file in uploaded_files:
        # 1. Extract Text (Requirement 2.1)
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        loader = PyPDFLoader("temp.pdf")
        all_docs.extend(loader.load())

    # 2. Chunking (Requirement 2.1)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(all_docs)

    # 3. Embed & Store in FAISS (Requirement 2.1 & 2.5)
    # Using HuggingFace because it's free and doesn't require an extra API key
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    return vector_store