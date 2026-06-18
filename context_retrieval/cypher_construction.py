from typing import List
from graph_db import graph 
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

        # EXTRACTED CANDIDATE ENTITIES 
        The following is a dictionary, where:
        - Each key is a segment of the user's query representing an entity that could be a skill, an occupation, a job offer or a location
        - Each value is a list of candidate entities retrieved for that phrase.
        - Each candidate has: label, description, type, id, score

        {entities}

        
        PROCESS:
        1. Treat each entity key independently as a semantic query.
        2. From each list, select the ONE entity that best matches the user's intent, as you can find it in the user's question.
        3. Construct the cypher based on the selected entities and the user's question. Prefer entity.id if available for Cypher matching.

        STRICT REQUIREMENTS:
        - Only one candidate per key should be chosen 
        - The cypher search query should be based only on the information from the retrieved entities that are relevant to the user's query. 
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

        # EXTRACTED CANDIDATE ENTITIES
        The following is a dictionary, where:
        - Each key is a segment of the user's query representing an entity that could be a skill, an occupation, a job or a location
        - Each value is a list of candidate entities retrieved for that phrase.
        - Each candidate has: label, description, type, id, score
        {entities}

        # PREVIOUS FAILED CYPHER QUERY
        {previous_cypher}

        # ERROR MESSAGE OR EXECUTION FAILURE
        {error}

        # MANDATORY CRITICAL FALLBACK STRATEGY (IF PREVIOUS QUERY RETURNED NO RESULTS)
        If the previous query failed because it returned 0 results ("No relevant data was found...")
        - if the user asks about specific jobs, vacancies, or hiring positions, YOU MUST BROADEN THE SCOPE TO OCCUPATIONS.
            Specific job vacancies are highly volatile, whereas generalized ESCO Occupations are stable.
        - if the user's question' contains very specific requirements and no results are found, use `OPTIONAL MATCH` for the requirements to broaden the scope 
        
        - CRITICAL RULE: If matching `(j:Job)` yielded nothing, rewrite the query to match `(o:Occupation)` using the same candidate skills/terms.
        - EXAMPLE SHIFT: 
          Instead of failing on: `MATCH (j:Job)-[:REQUIRES]->(s:Skill {{name: "Python"}})`
          Fallback to matching: `MATCH (o:Occupation)-[:REQUIRES]->(s:Skill {{name: "Python"}})`
        - ALTERNATIVELY: Use an `OPTIONAL MATCH` or check both nodes so that if no specific `Job` matches, general `Occupation` details are still returned.

        # OTHER COMMON FAILURE CAUSES TO CHECK
        1. REVERSED RELATIONSHIPS: The arrow in the relationship pattern went the wrong way (e.g., `(a)<-[:REL]-(b)` instead of `(a)-[:REL]->(b)`). Check the # GRAPH SCHEMA directions carefully!
        2. OVERCONSTRAINED FILTERS: A property match was too specific, misspelled, or combined too many strict `AND` conditions. 
        3. WRONG ENTITY CHOICE FROM CANDIDATES: Check if you picked a hyper-specific candidate string when a broader one exists.
        4. SYNTAX ISSUES: Misplaced brackets, invalid clauses, or undeclared variables.
        5. SCHEMA MISMATCH: Using labels, relationships, or properties not explicitly declared in the schema.

        # OUTPUT INSTRUCTIONS
        - Align query pattern directions perfectly with the schema. If relationship direction is tricky, use an undirected relationship (e.g., `(a)-[:REL_TYPE]-(b)`).
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
     

