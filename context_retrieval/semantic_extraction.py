from llm import llm
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import TypedDict, List




prompt = ChatPromptTemplate.from_messages([
    (
    "system",
    """
    You are an advanced text segmentation engine for an agricultural graph database.
    Your job is to break down a complex user query or conversation history into distinct, standalone semantic segments (fundamental titles or description phrases for skills, occupations, jobs, or locations).

    ---
    CRITICAL RULE FOR SEGMENTATION:
    Extract ONLY the naked core phrase, description, or title. You MUST strip away conversational framing wrappers, intent expressions, or search verbs (e.g., "job openings for", "positions related to", "skills involved in", "I want to look up"). 
    The segment must be reduced to the fundamental entity or action clause itself so it can match clean database entries later. Never include vague phrases like "the last skill", "this occupation", or "the job".
    
    ---
    STRATEGY:
    1. CONTEXTUAL REWRITING & STRIPPING: When extracting a segment, remove the introductory wrapper. 
       - Instead of "job openings for an agronomist", extract ONLY "agronomist".
       - Instead of "skills required for optimizing crop yields", extract ONLY "optimizing crop yields".
    2. KEEP ACTION PHRASES WHOLE: Do not strip down verbal descriptions if they are part of a unified, complex skill. Keep action clauses intact (e.g., "managing large-scale swine breeding arrays" must stay whole; do not chop it down to just "swine").
    3. HISTORIC ANCHOR RESOLUTION: You MUST actively resolve and replace conversational placeholders, references, or pronouns (such as 'it', 'this occupation', 'the last one') by substituting them with the actual stripped concrete entity name found in the "history" log. The final segment must be fully self-contained for standalone database searches. You should NOT include words or phrasing like "this skill", "skill", "occupation" that do not make sense alone. 
    4. DOMAIN GROUNDING: Do NOT invent or guess text segments. Every extracted piece must be derived directly from the text or history facts. Do NOT extract anything that is completely irrelevant to agricultural skills, occupations, jobs, or locations.
    5. MULTI-INTENT SEPARATION: If the user packs multiple intents into one query, separate them cleanly into unique elements.

    ---
    CORRECT BOUNDARY EXAMPLES:

    Input: "I would like to work as an agronomist, in a job where I can handle crop export logistics."
    Output:
    - "text_segment": "agronomist"
    - "text_segment": "handle crop export logistics"

    Input: "Tell me about the occupation of a vineyard manager."
    Output:
    - "text_segment": "vineyard manager"

    ---
    HISTORY REFERENCE RESOLUTION EXAMPLE:

    Chat History:
    Human: "What skills are required for the position of vineyard manager in Östersund, Sweden?"
    AI: "It requires grape harvesting, soil fertilization, and team coordination."
    Current User Query: "What are the occupations related to that position, and what does the last skill entail?"
    
    Output Segments:
    - "text_segment": "vineyard manager" (Resolved 'that position' to 'vineyard manager')
    - "text_segment": "team coordination" (Resolved 'the last skill' to 'team coordination' from AI message history)

    Return ONLY the structured output list of segments matching the requested schema.
    """
    ),
    MessagesPlaceholder("history"),
    ("human", "{user_query}")
])

class Segments(TypedDict):
    """the structure of the LLM output - required to be in this form for the llm"""
    segments: List[str]

# chain to extract job, occuppation and skill labels from the user's query
extraction_chain = prompt | llm.with_structured_output(Segments) 

def extract_segments(user_input,history):
    return extraction_chain.invoke({"user_query":user_input,"history":history})