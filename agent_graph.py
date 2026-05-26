from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from llm import llm
from context_retrieval.context_retrieval import get_occupations_by_label, get_jobs_by_label, get_skills_by_label, RetrievedInfoSchema
from context_retrieval.cypher_construction import get_cypher
from pprint import pprint
import json



class InputEntitiesSchema(TypedDict):
    """
        Schema for the entities that the user want information about.
        The LLM is initially instructed to extract those entities.
    """
    jobs: List[str]
    skills: List[str]
    occupations: List[str]


class State(TypedDict):
    """State of the graph."""
    user_query: str
    occupations : List[RetrievedInfoSchema]
    jobs : List[RetrievedInfoSchema]
    skills : List[RetrievedInfoSchema]
    context: List
    output: List 

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an entity extraction system for a graph database. "
     "Extract occupations, skills, and jobs from the user query. "
     "Extract only the labels for each entity. "
     "Return ONLY structured output. "
     "If you do not spot an entity, return an empty list for this entity category. For example, if you do not find a job, return an empty list for jobs." ),
    ("human", "{user_query}")
    ]
)

# chain to extract job, occuppation and skill labels from the user's query
extraction_chain = prompt | llm.with_structured_output(InputEntitiesSchema)

#tools = [get_jobs_by_label, get_occupations_by_label, get_skills_by_label, get_cypher]

def entity_extraction(state: State)->State:
    """
        Node for entity esxtraction.
        Instructs the LLM to extract the entities from the user's query and categorize them to occupations, skills and jobs.
    """
    result = extraction_chain.invoke({"user_query":state["user_query"]})
    return {
        "jobs":[RetrievedInfoSchema(label=l) for l in result.get("jobs",[])],
        "occupations": [RetrievedInfoSchema(label=l) for l in result.get("occupations",[])],
        "skills": [RetrievedInfoSchema(label=l) for l in result.get("skills",[])],
    }

def occupations_context(state: State)->State:
    """
        Node for context retrieval.
        Performs vector search to spot the extracted entities in the graph.
    """
    occupations = []
    skills = []
    jobs = []
    if state["occupations"]:
        occupations = [get_occupations_by_label(occ["label"]) for occ in state["occupations"] ]
    if state["skills"]:
        skills = [get_skills_by_label(skill["label"]) for skill in state["skills"]]
    if state["jobs"]:
        jobs = [get_jobs_by_label(job["label"]) for job in state["jobs"]]
    
    return {
         "occupations":occupations,
         "jobs":jobs,
         "skills":skills
    }


def cypher_generation(state: State) -> State:
     entities = {
         "occupations":state["occupations"],
         "jobs":state["jobs"],
         "skills":state["skills"]
    }
     query = get_cypher(question=state["user_query"],entities=entities)
     print(query)
     return state

graph = StateGraph(State)
graph.add_node("entities",entity_extraction)
graph.add_node("context",occupations_context)
graph.add_node("cypher",cypher_generation)
graph.add_edge(START,"entities")
graph.add_edge("entities","context")
graph.add_edge("context","cypher")
graph.add_edge("cypher", END)
app = graph.compile()
# graph visualization
png_data = app.get_graph().draw_mermaid_png()

with open("langgraph_tests/graph_extraction_1.png", "wb") as f:
    f.write(png_data)

result = app.invoke(State(user_query="I want information about the occupation Pig Breed and the dollowing skills: manage technical aspects, pest controlling and maintain plant growth. Also, tell me information about the agricultural technician job. Also tell me where the agricultural technitian job is"))


with open("vector_search_structured_1.json","w",encoding="utf-8") as f:
        json.dump(result,f,indent=4,ensure_ascii=False)