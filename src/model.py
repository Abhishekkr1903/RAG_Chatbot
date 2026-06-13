import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# def get_llm():

#     return ChatGoogleGenerativeAI(
#         model="gemini-2.5-flash",
#         temperature=0
#     )

##change for streamlit.
import streamlit as st

def get_llm():

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )
