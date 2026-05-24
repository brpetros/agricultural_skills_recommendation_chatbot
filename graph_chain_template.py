from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from llm import llm
from vectors import get_occupation_label, convert_result, OccupationSchema    

class GroundingSchema(TypedDict):
    label: str
    grounded_label: str

class EntitiesSchema(TypedDict):
    jobs: List[str]
    skills: List[str]
    occupations: List[str]


class State(TypedDict):
    user_query: str
    occupations : List[OccupationSchema]
    jobs : List[str]
    skills : List[str]
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

extraction_chain = prompt | llm.with_structured_output(EntitiesSchema)

def entity_extraction(state: State)->State:
    result = extraction_chain.invoke({"user_query":state["user_query"]})
    
    return {
        "jobs": result.get("jobs",[]),
        "occupations": result.get("occupations",[]),
        "skills": result.get("skills",[])
    }

def occupations_context(state: State)->State:
    print("--this is the retrieved context--\n")
    occupations = []
    if state["occupations"]:
        occupations = [convert_result(get_occupation_label(occ)) for occ in state["occupations"] ]
    
    state["occupations"] = occupations
    print(state)
    return state


graph = StateGraph(State)
graph.add_node("entities",entity_extraction)
graph.add_node("context",occupations_context)
graph.add_edge(START,"entities")
graph.add_edge("entities","context")
graph.add_edge("context", END)
app = graph.compile()
# graph visualization
png_data = app.get_graph().draw_mermaid_png()

"""with open("langgraph_tests/graph_extraction.png", "wb") as f:
    f.write(png_data)"""

result = app.invoke(State(user_query="I want information about the following occupations: Pig breeder, accomodation manager, operator for meat preparations. "))


