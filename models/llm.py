import os
from langchain_groq import ChatGroq

def get_chatgroq_model():
    # The .strip() ensures no hidden spaces or newline characters break the key
    raw_key = "gsk_oC3frp0SG8AqghsOejWWWGdyb3FYvawxAnFY1R5HmVdO41wVIyq8THI"
    api_key = raw_key.strip()
    
    return ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-70b-versatile",
        temperature=0
    )