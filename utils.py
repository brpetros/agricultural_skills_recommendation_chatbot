import streamlit as st
from streamlit.runtime.scriptrunner.script_runner import get_script_run_ctx

def write_message(role, content, save = True):
    """
    This is a helper function that saves a message to the
     session state and then writes a message to the UI
    """
    # Append to session state
    if save:
        st.session_state.messages.append({"role": role, "content": content})

    # Write to UI
    with st.chat_message(role):
        st.markdown(content)

def get_session_id():
    ctx = get_script_run_ctx()
    if ctx:
        return ctx.session_id
    return "test-session"

def extract_text_safely(llm_response, fallback_text: str = "") -> str:
    """
    Safely extracts response text from an LLM output object (LangChain or raw)
    without triggering 'list index out of range' errors.
    """
    if llm_response is None:
        return fallback_text

    # Case 1: LangChain Message Objects (AIMessage, BaseMessage)
    if hasattr(llm_response, "content"):
        content = llm_response.content
        
        # If content is a list (e.g., Anthropic multi-part block formats)
        if isinstance(content, list):
            if len(content) > 0:
                first_item = content[0]
                if isinstance(first_item, dict) and "text" in first_item:
                    return first_item["text"]
                return str(first_item)
            return fallback_text
            
        # If content is a standard string
        if isinstance(content, str):
            return content.strip()

    # Case 2: Response is a standard Python dictionary
    if isinstance(llm_response, dict):
        if "content" in llm_response:
            content = llm_response["content"]
            if isinstance(content, list) and len(content) > 0:
                return content[0].get("text", fallback_text) if isinstance(content[0], dict) else str(content[0])
            return str(content).strip()
        if "text" in llm_response:
            return llm_response["text"].strip()

    # Case 3: Response is already a raw string
    if isinstance(llm_response, str):
        return llm_response.strip()

    # Catch-all fallback
    return str(llm_response) if llm_response else fallback_text