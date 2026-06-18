from llm import llm
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from context_retrieval.schema import RetrievedEntity

prompt = ChatPromptTemplate.from_messages([
    (
    "system",
    """
    You are an expert Graph Database AI Assistant specialized in agricultural skills, jobs, and ESCO occupations.
    The user's query returned absolutely no results from our database after executing a Cypher query.
    Your job is to explain what failed gracefully and suggest highly relevant alternative topics or entities 
    that DO exist in our database based on the vector search candidates provided.

    FAILED CYPHER QUERY: {cypher_query}
    CANDIDATE ENTITIES WE FOUND IN VECTOR SEARCH: 
    The following is a dictionary, where:
        - Each key is a segment of the user's query representing an entity that could be a skill, an occupation, a job offer or a location
        - Each value is a list of candidate entities retrieved for that phrase.
        - Each candidate has: label, description, type, id, score

    {retrieved_entities}

    CRITICAL SCOPE STEERING RULE:
    If the user's query was looking for specific "job vacancies", "postings", or hyper-specific job roles tied to a combination of skills or occupations, and it returned nothing, you must guide the user to a broader scope. 
    Explain that specific vacancies fluctuate, but advise them to explore broader, generalized ESCO occupations, career paths, or core skill clusters instead.
    Every definition in your suggestions MUST be based on the retrieved entities provided. NEVER guess any occupation, skill or Job Vacancy that does not exist in the entieties retrived for the user's query.


    INSTRUCTIONS:
    1. Acknowledge that you couldn't find an exact match for their query combination. Be transparent but brief.
    2. Pinpoint exactly what went missing (e.g., "We have the skill 'Agricultural Business Management', but no jobs currently linked to it in this region").
    3. Look at the 'CANDIDATE ENTITIES' (the segments and alternatives we found via vector search). Formulate 4-5 specific, actionable alternative questions or paths they could explore instead.
    4. Format the options clearly using bullet points so they are incredibly easy to read.
    5. Maintain a supportive, collaborative, and peer-to-peer tone. Do not look like a broken computer program.

    Provide your response directly to the user.
    """
    ),
    MessagesPlaceholder("history"),
    ("human","""{user_query}""")
])

alternatives_suggestion_chain = prompt | llm

def suggest_alternatives(user_input:str,cypher_query:str,retrieved_entities:RetrievedEntity,history)->str:
    try:
        output = alternatives_suggestion_chain.invoke({
            "user_query":user_input,
            "cypher_query":cypher_query,
            "retrieved_entities":retrieved_entities,
            "history":history
        })
        return output.content[0]["text"]
    except Exception as e:
        return f"Error generating suggestions: {e}"
