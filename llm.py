import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

llm = ChatGoogleGenerativeAI(
    google_api_key=st.secrets["GOOGLE_API_KEY"],
    model=st.secrets["GEMINI_MODEL"],
)

# Create the Embedding model

embeddings = GoogleGenerativeAIEmbeddings(
    google_api_key=st.secrets["GOOGLE_API_KEY"],
    model="models/embedding-001"
)