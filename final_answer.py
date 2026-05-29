from llm import llm
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system",
     """
    You are an expert in agriculture, providing information about Occupations, Jobs and Skills related to agriculture. 
    Be accurate and return as much information as possible.
    Do not answer any questions that are not relevant to agricultural skills, occupations and jobs.
    Do not answer any questions using your pre-trained knowledge. Use only the information provided by the context.

    #RETRIEVED ENTITIES
    {entities}

    #RETRIEVED CONTEXT
    {context}

    #TASK
    Answer the user's query based only on the information about the retrieved entities and context.
    Do not introduce any information that is not relevant to the provided information.
    """
    ),
    ("human","{query}")
])

final_answer_chain = prompt | llm

def get_final_answer(query,entities,context):
    return final_answer_chain.invoke({
        "query":query,
        "entities":entities,
        "context":context
    })