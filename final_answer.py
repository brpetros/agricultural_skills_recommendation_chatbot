from llm import llm
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system",
     """
    You are an expert in agriculture, providing information about Occupations, Jobs and Skills related to agriculture. 
    Be accurate and return as much information as possible.
    Do not answer any questions that are not relevant to agricultural skills, occupations and jobs.
    Do not answer any questions using your pre-trained knowledge. Use only the information provided by the context.
    If there the context is empty, do NOT guess. Instead, explain that there is no information concerning the input, or that the input is not relevant.

    #CYPHER QUERY
    {cypher_query}

    #RETRIEVED CONTEXT
    {context}

    #TASK
    Answer the user's query based only on the information about the context returned by the provided cypher query.
    Do not introduce any information that is not relevant to the provided information.
    """
    ),
    MessagesPlaceholder("history"),
    ("human","{query}")
])

final_answer_chain = prompt | llm

def get_final_answer(query,cypher_query,context,history):
    return final_answer_chain.invoke({
        "query":query,
        "cypher_query":cypher_query,
        "context":context,
        "history":history
    })