from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings():
    """Returns the embedding model used for RAG."""
    # This model is lightweight and highly effective for document search
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    return HuggingFaceEmbeddings(model_name=model_name)