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
        - Prefer entity IDs when available.
        - Output ONLY the Cypher query. Do NOT wrap it in markdown code blocks, do NOT write markdown formatting, and do NOT explain anything.
        - Ensure all labels, relationships, and properties exist in the schema.
        - If the query contains references such as 'it', 'they', 'that occupation', 'the previous job', 'the last skill', use the conversation history to resolve those references.
        - RESOLUTION STRATEGY: If you need to see the history, scan the conversation history backward, starting from the IMMEDIATELY PRECEDING MESSAGE (the very last assistant turn) up to the oldest.
        - Do NOT make the query return whole nodes when they include embedding properties like titleEmbedding, labelEmbedding or descriptionEmbedding. Choose specific node properties that do not contain embeddings.
        - If unsure, simplify the query instead of guessing.

        RELATIONSHIP DIRECTION RULES (CRITICAL):
        - Analyze the relationship directions defined in the # GRAPH SCHEMA (e.g., `(:NodeA)-[:REL_TYPE]->(:NodeB)`).
        - Ensure the arrows in your MATCH patterns strictly match the schema's canonical direction. For example, if the schema specifies `(:User)-[:HAS_SKILL]->(:Skill)`, your query must match `(u:User)-[:HAS_SKILL]->(s:Skill)`. Do NOT write `(u:User)<-[:HAS_SKILL]-(s:Skill)`.
        - SAFE FALLBACK: If the direction of traversal is ambiguous in the user's question, or if you are unsure of the correct semantic flow, use an UNDIRECTED relationship pattern by omitting arrowheads (e.g., `(a)-[:REL_TYPE]-(b)`). This prevents returning 0 results due to directional mismatches while remaining safe.
        """
    ),
    MessagesPlaceholder("history"),
    (
        "human",
        """
        {question}
        """
    )
])

# =====================================================================
# 2. RETRY CYPHER GENERATION PROMPT
# =====================================================================
# Enhanced to handle both syntax/schema errors and logical errors (like returning 0 results)
retry_cypher_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are an expert Neo4j Cypher query generator specialized in correcting failed queries.

        Your task is to FIX broken or unproductive Cypher queries and produce correct, executable, and SAFE READ-ONLY Cypher queries.

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
        - CREATE, MERGE, DELETE, SET, REMOVE, DROP, CALL, LOAD CSV, APOC procedures, db.*, schema modifications, writes of any kind.

        # GRAPH SCHEMA
        {schema}

        # EXTRACTED ENTITIES
        {entities}

        # PREVIOUS FAILED CYPHER QUERY
        {previous_cypher}

        # ERROR MESSAGE OR EXECUTION FAILURE
        {error}

        # TASK
        Analyze why the previous query failed or returned no results, and generate a corrected Cypher query.

        Common failure causes to check:
        1. REVERSED RELATIONSHIPS: The arrow in the relationship pattern went the wrong way (e.g., `(a)<-[:REL]-(b)` instead of `(a)-[:REL]->(b)`). Check the # GRAPH SCHEMA directions carefully!
        2. OVERCONSTRAINED FILTERS: A property match was too specific or misspelled.
        3. SYNTAX ISSUES: Misplaced brackets, invalid clauses, or undeclared variables.
        4. SCHEMA MISMATCH: Using labels, relationships, or properties not explicitly declared in the schema.

        Correction strategy:
        - Align query pattern directions perfectly with the schema.
        - If relationship direction is causing the failure, use an undirected relationship (e.g., `(a)-[:REL_TYPE]-(b)`) to bypass directional strictness safely.
        - Simplify overly complex conditions.
        - Return ONLY the corrected Cypher query. Do NOT wrap it in markdown code blocks and do NOT explain anything.
        - Do NOT make the query return whole nodes when they include embedding properties like titleEmbedding, labelEmbedding or descriptionEmbedding.
        - RESOLUTION STRATEGY: If you need to see the history, scan the conversation history backward, starting from the IMMEDIATELY PRECEDING MESSAGE (the very last assistant turn) up to the oldest.
        """
    ),
    MessagesPlaceholder("history"),
    (
        "human",
        """
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
     

