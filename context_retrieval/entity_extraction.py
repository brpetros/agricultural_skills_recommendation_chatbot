from llm import llm
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import TypedDict, List
from context_retrieval.schema import InputEntity

prompt = ChatPromptTemplate.from_messages([
    ("system",
    """
    You are an entity extraction system for a graph database.

    Extract occupations, skills, jobs, or locations from the current user query, 
    or the message history, if there are references to it.
    Save each entity exactly as stated by the user. 

    Assign a category ONLY if the user provides it. 
    Example 1: if the user says that 
    the entity is a skill, you should save "skill" as the entity category. 
    Example 2: if the user says "I need information about the following occupations", 
    the entities that follow this phrase should be categorized as occupations.
    If the user does not specify a category, you MUST save "unknown" as the entity's category.
    A category can have one of the following values: 
    - "job"
    - "skill"
    - "occupation"
    - "location"
    - "unknown"

    If the query contains references such as
    'it', 'they', 'that occupation', 'the previous job', 'this skill'
    which cannot be found in the current question,
    you MUST use the conversation history to resolve those references.
    RESOLUTION STRATEGY: If you need to see the history, scan the conversation history backward, starting from the IMMEDIATELY PRECEDING MESSAGE (the very last assistant turn) up to the oldest.
    Do NOT save entities like "this skill", "first occupation". 

    Return ONLY structured output.
    If you do not spot a relevant entity in the current query or in the history, return an empty list. 
    """),
    MessagesPlaceholder("history"),
    ("human", "{user_query}")
    ]
)

class Entities(TypedDict):
    """the structure of the LLM output - required to be in this form for the llm"""
    entities: List[InputEntity]

# chain to extract job, occuppation and skill labels from the user's query
extraction_chain = prompt | llm.with_structured_output(Entities)

def extract_entities(user_input,history):
    return extraction_chain.invoke({"user_query":user_input,"history":history})