import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    openai_api_key=st.secrets["OPENAI_API_KEY"],
    model=st.secrets["OPENAI_MODEL"],
)

# Create the Embedding model

embeddings = OpenAIEmbeddings(
    openai_api_key=st.secrets["OPENAI_API_KEY"]
)