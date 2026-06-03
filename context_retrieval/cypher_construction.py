from typing import List
from skills_graph import graph 
from llm import llm
from langchain_neo4j import GraphCypherQAChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from context_retrieval.schema import RetrievedEntity



original_cypher_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are an expert Neo4j Cypher query generator.

        Your task is to convert a natural language user question into a valid, safe Cypher query, based on the given schema and the entities.

        ONLY generate READ-ONLY Cypher queries:
        - Allowed: MATCH, WHERE, RETURN, WITH, OPTIONAL MATCH, ORDER BY, LIMIT
        - NOT allowed: CREATE, DELETE, MERGE, SET, DROP, CALL, or any write operation

        # GRAPH SCHEMA
        {schema}

        # EXTRACTED ENTITIES
        {entities}

        STRICT REQUIREMENTS:
        - The cypher search query should be based only on the information from the retrieved entities that are relevant to the user's query. 
        - Each entity contains the original user's input, which was used for the entity to be retrieved. Use this to relate the user's question with the retrieved entities.
        - Prefer entity IDs when available
        - Output ONLY the corrected Cypher query
        - Do NOT explain anything
        - Ensure all labels, relationships, and properties exist in the schema
        - If the query contains references such as 'it', 'they', 'that occupation', 'the previous job', 'the last skill', use the conversation history to resolve those references.
        - Do NOT make the query return whole nodes when they include embedding properties like titleEmbedding, labelEmbedding or descriptionEmbedding. Chose the 
        node properties that do not contain embeddings.
        - Prefer entity IDs when available
        - If unsure, simplify the query instead of guessing
        """
    ),
    MessagesPlaceholder("history"),
    (
        "human",
        """
        # USER QUESTION
        {question}
        """
    )
])
# prompt to retry cypher generation

retry_cypher_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are an expert Neo4j Cypher query generator specialized in correcting failed queries.

        Your task is to FIX broken Cypher queries and produce correct, executable, and SAFE READ-ONLY Cypher queries.

        ONLY READ operations are allowed.

        Allowed clauses:
        - MATCH
        - OPTIONAL MATCH
        - WHERE
        - WITH
        - RETURN
        - ORDER BY
        - LIMIT

        Forbidden operations and clauses:
        - CREATE
        - MERGE
        - DELETE
        - SET
        - REMOVE
        - DROP
        - CALL
        - LOAD CSV
        - APOC procedures
        - db.*
        - schema modifications
        - writes of any kind

        # GRAPH SCHEMA
        {schema}

        # EXTRACTED ENTITIES
        {entities}


        # PREVIOUS FAILED CYPHER QUERY
        {previous_cypher}

        # ERROR MESSAGE 
        {error}

        # TASK
        Analyze why the query failed and generate a corrected Cypher query.

        Possible failure causes include:
        - syntax issues
        - invalid labels
        - invalid relationships
        - invalid properties
        - incorrect filtering
        - incorrect entity usage
        - schema mismatch
        - overconstrained conditions

        Correction strategy:
        - fix syntax issues
        - align query with schema
        - correct labels/relationships/properties
        - simplify overly complex queries
        - relax filters slightly if appropriate
        - use grounded entity IDs when available

        Return ONLY the corrected Cypher query.
        
        STRICT REQUIREMENTS:
        - The cypher search query should be based only on the information from the retrieved entities
        - Prefer entity IDs when available
        - Output ONLY the corrected Cypher query
        - Do NOT explain anything
        - Ensure all labels, relationships, and properties exist in the schema
        - If the query contains references such as 'it', 'they', 'that occupation', 'the previous job', 'the last skill', use the conversation history to resolve those references.
        - Do NOT make the query return whole nodes when they include embedding properties like titleEmbedding, labelEmbedding or descriptionEmbedding. Chose the 
        node properties that do not contain embeddings.
        - If unsure, simplify the query instead of guessing
        """
    ),
    MessagesPlaceholder("history"),
    (
        "human",
        """
        # USER INPUT
        {question}
        """
    )
])


cypher_chain = original_cypher_prompt | llm
retry_cypher_chain = retry_cypher_prompt | llm

def get_cypher(question:str, entities:List[RetrievedEntity], history, retry:bool, previous_cypher:str="", error:str=""):
    """generates cypher based on the graph schema and the user's question"""
    if retry:
        return retry_cypher_chain.invoke(
            {
                "previous_cypher":previous_cypher,
                "error":error,
                "schema":graph.get_structured_schema,
                "entities":entities,
                "question":question,
                "history":history
            }
        )
    
    return cypher_chain.invoke(
        {
            "schema":graph.get_structured_schema,
            "entities":entities,
            "question":question,
            "history":history
        }
    )
     

