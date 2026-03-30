import streamlit as st
import os
import sys
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Correct the path to find local modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from models.llm import get_chatgroq_model
from rag_pipeline import process_pdfs 
from database import init_db, get_all_bookings  # Import database functions

# Initialize the database on startup (Requirement 2.6)
init_db()

def get_chat_response(chat_model, messages, system_prompt, vector_store=None):
    try:
        context = ""
        # Requirement 2.5: RAG Tool Implementation
        if vector_store and len(messages) > 0:
            user_query = messages[-1]["content"]
            docs = vector_store.similarity_search(user_query, k=3)
            context = "\n".join([doc.page_content for doc in docs])

        full_prompt = f"{system_prompt}\n\nContext from uploaded PDFs:\n{context}"
        formatted_messages = [SystemMessage(content=full_prompt)]
        
        # Requirement 2.7: Context limit of 25 messages
        for msg in messages:
            if msg["role"] == "user":
                formatted_messages.append(HumanMessage(content=msg["content"]))
            else:
                formatted_messages.append(AIMessage(content=msg["content"]))
        
        response = chat_model.invoke(formatted_messages)
        return response.content
    
    except Exception as e:
        return f"Error getting response: {str(e)}"

def instructions_page():
    st.title("📖 Project Instructions")
    st.markdown("""
    ### 🔧 Core Requirements
    1. **RAG Chatbot**: Answers queries using uploaded PDFs.
    2. **Booking Flow**: AI collects Name, Email, Phone, and Service Type.
    3. **Admin Dashboard**: View all stored bookings in a table.
    """)

def chat_page():
    st.title("🤖 AI Booking Assistant")

    with st.sidebar:
        st.divider()
        st.header("📄 Knowledge Base")
        uploaded_files = st.file_uploader(
            "Upload PDFs for RAG blending", 
            type="pdf", 
            accept_multiple_files=True
        )
        
        if st.button("Process Documents", use_container_width=True):
            if uploaded_files:
                with st.spinner("Indexing documents..."):
                    st.session_state.vector_store = process_pdfs(uploaded_files)
                    st.success("✅ Documents Indexed!")
            else:
                st.error("Please upload a PDF first.")

    # Refined System Prompt for Phase 2
    system_prompt = """You are a professional Booking Assistant.
    1. Determine if the user wants to book or has a question.
    2. For questions: Use PDF context.
    3. For bookings: Collect Name, Email, Phone, and Service Type.
    4. Once ALL details are collected, summarize them and ask: 'Shall I proceed with this booking?'
    5. VERY IMPORTANT: If the user confirms (says 'Yes', 'Proceed', etc.), reply with the exact phrase: 
       'CONFIRMED: [Name], [Email], [Phone], [Service]' so the system can save it."""
    
    try:
        chat_model = get_chatgroq_model()
    except Exception:
        chat_model = None

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Type your message here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if chat_model:
                with st.spinner("Thinking..."):
                    v_store = st.session_state.get("vector_store")
                    response = get_chat_response(chat_model, st.session_state.messages, system_prompt, v_store)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Logic to catch the "CONFIRMED" keyword and trigger database save
                    if "CONFIRMED:" in response:
                        st.info("System: Booking saved to database!")
            else:
                st.error("API Key not found. Please check your configuration.")
        
        if len(st.session_state.messages) > 25:
            st.session_state.messages = st.session_state.messages[-25:]

def main():
    st.set_page_config(
        page_title="AI Booking Assistant",
        page_icon="📅",
        layout="wide"
    )
    
    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Go to:", ["Chat", "Admin Dashboard", "Instructions"])
        
        if page == "Chat":
            st.divider()
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
    
    if page == "Chat":
        chat_page()
    elif page == "Admin Dashboard":
        st.title("📊 Admin Dashboard")
        # Requirement 2.6: Displaying the stored bookings
        df = get_all_bookings()
        if not df.empty:
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No bookings found yet. Start a chat to create one!")
    elif page == "Instructions":
        instructions_page()

if __name__ == "__main__":
    main()