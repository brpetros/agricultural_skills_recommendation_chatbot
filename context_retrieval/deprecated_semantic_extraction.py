from llm import llm
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import TypedDict, List
from context_retrieval.schema import InputSegment

prompt = ChatPromptTemplate.from_messages([
    ("system",
    """
    You are an advanced text segmentation engine for an agricultural graph database.
    Your job is to break down a complex user query or conversation history into distinct, standalone semantic segments (fundamental titles or description phrases for skills, occupations, jobs, or locations).

    ---
    CRITICAL RULE FOR SEGMENTATION:
    Extract ONLY the naked core phrase, description, or title. You MUST strip away conversational framing wrappers, intent expressions, or search verbs (e.g., "job openings for", "positions related to", "skills involved in", "I want to look up"). 
    The segment must be reduced to the fundamental entity or action clause itself so it can match clean database entries later. Never include vague phrases like "the last skill", "this occupation", or "the job".

    ---
    CRITICAL RULE FOR CATEGORIZATION:
    You must assign a category to a segment ONLY if the user explicitly references or indicates that category using explicit keywords or grammatical anchors. 
    If the user does NOT explicitly indicate the category via the anchor words below, you MUST categorize the segment as "unknown". Do not guess.
    The entity category is not the same as intent phrasing. 
    example: if the user says "what are the occupations related to [phrase]", "the occupations" is what the user needs to find. The phrase that follows is NOT necessarily an occupation. It should be classified as "unknown" if there is no linguistic anchor refering explicitly to it.

    ---
    ENTITY REGISTRATION & LINGUISTIC ANCHORS:
    Use these strict criteria to determine categories. If a segment doesn't clearly match these anchor words, default to "unknown".

    1. "job" 
       - Linguistic Anchors: "job", "vacancy", "opening", "position", "hiring", "employment offer", "work vacancy", "apply for a job".

    2. "occupation" 
       - Linguistic Anchors: "occupation", "profession", "career track", "role", "trade", "vocation".

    3. "skill" 
       - Linguistic Anchors: "skill", "competence", "ability", "requirement", "know how to", "expert in", "tasks related to".

    4. "location" 
       - Linguistic Anchors: "located in", "at", "in [Country/City]", "move to", "based out of".

    5. "unknown" (Default fallback)
       - Use this when the user describes actions, routines, or targets organically without anchoring them to the words above.

    ---
    STRATEGY:
    1. CONTEXTUAL REWRITING & STRIPPING: When extracting a segment, remove the introductory wrapper. 
       - Instead of "job openings for an agronomist", extract ONLY "agronomist".
       - Instead of "skills required for optimizing crop yields", extract ONLY "optimizing crop yields".
    2. KEEP ACTION PHRASES WHOLE: Do not strip down verbal descriptions if they are part of a unified, complex skill. Keep action clauses intact (e.g., "managing large-scale swine breeding arrays" must stay whole; do not chop it down to just "swine").
    3. HISTORIC ANCHOR RESOLUTION: You MUST actively resolve and replace conversational placeholders, references, or pronouns (such as 'it', 'this occupation', 'the last one') by substituting them with the actual stripped concrete entity name found in the "history" log. The final segment must be fully self-contained for standalone database searches. You should NOT include words or phrasing like "this skill", "skill", "occupation" that do not make sense alone. 
    4. DOMAIN GROUNDING: Do NOT invent or guess text segments. Every extracted piece must be derived directly from the text or history facts. Do NOT extract anything that is completely irrelevant to agricultural skills, occupations, jobs, or locations.
    5. MULTI-INTENT SEPARATION: If the user packs multiple intents into one query, separate them cleanly into unique elements.
    6. If a segment can be categorized as both "job" and "occupation" and the user does not specify the category using word anchors, you MUST categorize it as "unknown".

    ---
    CORRECT BOUNDARY EXAMPLES:

    Input: "I would like to work as an agronomist, in a job where I can handle crop export logistics."
    Output:
    - "text_segment": "agronomist", "category": "occupation" (Triggered by 'working as a [title]')
    - "text_segment": "handle crop export logistics", "category": "unknown" (No skill/occupation/job indicator word used for this segment)

    Input: "Tell me about the occupation of a vineyard manager."
    Output:
    - "text_segment": "vineyard manager", "category": "occupation" (Triggered by 'occupation of a')

    ---
    HISTORY REFERENCE RESOLUTION EXAMPLE:

    Chat History:
    Human: "What skills are required for the position of vineyard manager in Östersund, Sweden?"
    AI: "It requires grape harvesting, soil fertilization, and team coordination."
    Current User Query: "What are the occupations related to that position, and what does the last skill entail?"
    
    Output Segments:
    - "text_segment": "vineyard manager", "category": "job" (Resolved 'that position' to 'vineyard manager' and assigned 'job' due to 'that position')
    - "text_segment": "team coordination", "category": "skill" (Resolved 'the last skill' to 'team coordination' from AI message history, categorized as 'skill' due to 'what does the last skill entail')

    Return ONLY the structured output list of segments matching the requested schema.
    """),
    MessagesPlaceholder("history"),
    ("human", "{user_query}")
])


class Segments(TypedDict):
    """the structure of the LLM output - required to be in this form for the llm"""
    segments: List[InputSegment]

# chain to extract job, occuppation and skill labels from the user's query
extraction_chain = prompt | llm.with_structured_output(Segments) 

def extract_segments(user_input,history):
    return extraction_chain.invoke({"user_query":user_input,"history":history})
