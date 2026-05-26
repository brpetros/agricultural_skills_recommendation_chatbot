from skills_graph import graph 
from llm import llm
from langchain_neo4j import GraphCypherQAChain
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool

cypher_prompt = PromptTemplate.from_template("""
    You are a neo4j graph developer translating user questions into Cypher to answer questions about agricultural skills and their relation to the job market.
    Convert the user's question based on the schema. 

    Use only the provided relationship types and properties in the schema.
    Do not use any other relationship types or properties that are not provided.
    If the question cannot be answered using this schema, return "This question cannot be answered based on the available data."
    Do not return entire nodes or embedding properties.

                                             
    Schema:
    {schema}

    Question:
    {question}

    Cypher Query:
    """
)

cypher_prompt = PromptTemplate.from_template(
    """
        You are an expert Neo4j Cypher query generator.
        Your task is to convert a natural language question into a valid, safe Cypher query.

        ---

        # GRAPH SCHEMA
        You are working with the following graph schema:

        {schema}

        ---

        # EXTRACTED ENTITIES (already grounded to the graph when possible)

        These entities were extracted from the user question and correspond to nodes in the graph:

        {entities}

        Use entity IDs when available for precise matching.

        ---

        # USER QUESTION
        {question}

        # RULES
        Output ONLY the cypher query.
        ONLY generate READ-ONLY Cypher queries:
        - Allowed: MATCH, WHERE, RETURN, WITH, OPTIONAL MATCH, ORDER BY, LIMIT
        - NOT allowed: CREATE, DELETE, MERGE, SET, DROP, CALL, or any write operation
        Prefer using entity IDs when provided
    """
)

cypher_qa = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True,
    cypher_prompt=cypher_prompt,
    return_intermediate_steps=True,
    allow_dangerous_requests=True, # This allows the chain to execute any Cypher query, including those that modify the database. Use with caution!
)


cypher_chain = cypher_prompt | llm

#@tool
def get_cypher(question:str, entities):
    """generates cypher based on the graph schema and the user's question"""
    return cypher_chain.invoke(
        {
            "schema":graph.get_structured_schema,
            "entities":entities,
            "question":question
        }
    )
     
