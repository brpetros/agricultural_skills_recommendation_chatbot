import streamlit as st
from utils import write_message
from agent_graph import generate_response
from datetime import datetime

st.set_page_config("Skill Recommender",page_icon=":seedling:")
st.title("Skill Recommender")

# Set up Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, how can I help you?"},
    ]

# Submit handler
def handle_submit(message):
    # Handle the response
    with st.spinner('Thinking...'):
        # Call the agent
        response = generate_response(message)
        write_message('assistant', response)


# Display messages in Session State
for message in st.session_state.messages:
    write_message(message['role'], message['content'], save=False)

# Handle any user input
if question := st.chat_input("Ask me about agricultural skills and jobs."):
    # Display user message in chat message container
    write_message('user', question)

    # Generate a response
    handle_submit(question)