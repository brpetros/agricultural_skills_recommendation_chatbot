from skills_graph import graph 
from llm import llm
from langchain_neo4j import GraphCypherQAChain
from langchain.prompts.prompt import PromptTemplate


cypher_prompt = PromptTemplate.from_template("""
    You are a neo4j graph developer translating user questions into Cypher to answer questions about agricultural skills and their relation to the job market.
    Convert the user's question based on the schema. 

    Use only the provided relationship types and properties in the schema.
    Do not use any other relationship types or properties that are not provided.

    Do not return entire nodes or embedding properties.
    
    Schema:
    {schema}

    Question:
    {question}

    Cypher Query:
    """
)

cypher_qa = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True,
    cypher_prompt=cypher_prompt,
    allow_dangerous_requests=True, # This allows the chain to execute any Cypher query, including those that modify the database. Use with caution!
)