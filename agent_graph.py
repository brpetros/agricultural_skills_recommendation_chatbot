from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_neo4j import Neo4jChatMessageHistory
from llm import llm
from context_retrieval.context_retrieval import get_occupations_by_label, get_jobs_by_label, get_skills_by_label
from context_retrieval.cypher_construction import get_cypher
from pprint import pprint
import json
from skills_graph import graph
from context_retrieval.schema import InputEntitiesSchema, RetrievedEntitiesSchema, RetrievedInfoSchema
from final_answer import get_final_answer

def get_memory(session_id):
    """Returns the chat's memory by the graph"""
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

class State(TypedDict):
    """State of the graph."""
    session_id: str
    messages: List[BaseMessage]

    user_query: str

    retrieved_entities: List[List[RetrievedEntitiesSchema]]

    cypher_query: str
    cypher_result: List 
    is_relevant: bool
    cypher_error: str
    cypher_retry_count: int 

    output: str

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

 

def entity_extraction(state: State)->State:
    """
        Node for entity esxtraction.
        Instructs the LLM to extract the entities from the user's query and categorize them to occupations, skills and jobs.
    """
    result = extraction_chain.invoke({"user_query":state["user_query"]})

    return { "retrieved_entities":
        {
        "jobs":[RetrievedInfoSchema(label=l) for l in result.get("jobs",[])],
        "occupations": [RetrievedInfoSchema(label=l) for l in result.get("occupations",[])],
        "skills": [RetrievedInfoSchema(label=l) for l in result.get("skills",[])],
    }}

def entities_assessment(state: State)->State:
    """Assesses if the given query is relevant (if it has entities like jobs occupations or skills)."""
    has_entities = any([
        state["retrieved_entities"]["occupations"],
        state["retrieved_entities"]["jobs"],
        state["retrieved_entities"]["skills"]
        ] 
    )

    if not has_entities:
        return {
            "is_relevant": False,
            "output": "The query does not appear related to the graph domain."
        }

    return {
        "is_relevant": True
    }

def route_after_assessment(state: State)->str:
    """conditinal node after entities assessment"""
    if state["is_relevant"]:
        return "entities_relevant"
    else:
        return "entities_irrelevant"

def retrieve_context(state: State)->State:
    """
        Node for context retrieval.
        Performs vector search to spot the extracted entities in the graph.
    """
    occupations = []
    skills = []
    jobs = []
    if state["retrieved_entities"]["occupations"]:
        occupations = [get_occupations_by_label(occ["label"]) for occ in state["retrieved_entities"]["occupations"] ]
    if state["retrieved_entities"]["skills"]:
        skills = [get_skills_by_label(skill["label"]) for skill in state["retrieved_entities"]["skills"]]
    if state["retrieved_entities"]["jobs"]:
        jobs = [get_jobs_by_label(job["label"]) for job in state["retrieved_entities"]["jobs"]]
    result = {
        "occupations":occupations,
        "jobs":jobs,
        "skills":skills
    }
    with open("final_entities_result.json","w",encoding="utf-8") as f:
        json.dump(result,f,default=str,indent=4,ensure_ascii=False)
    return { "retrieved_entities":{
         "occupations":occupations,
         "jobs":jobs,
         "skills":skills
    }}


def context_assessment(state:State)->State:
    """node to assess the score of the retrieved context (if it is actually relevant)"""
    # to-do: expand this node so that it actually checks the retrieved results for their relevance 
    return {
        "is_relevant":True
    }

def route_after_context_assessment(state:State)->str:
    """routes after context assessment"""
    if state["is_relevant"]:
        return "context_relevant"
    else:
        return "context_irrelevant"

def cypher_generation(state: State) -> State:
    """node for cypher generation based on retrieved entities and query"""
    
    # if we need to retry we have the cypher retry node
    query = get_cypher(question=state["user_query"],entities=state["retrieved_entities"], retry=False)
    print("--returned cypher--")
    print(query.content[0]["text"])
    return {
        "cypher_query": query.content[0]["text"],
        "cypher_retry_count":0
    }

def cypher_safety_validation(state: State)->State:
    """validation if there is any dangerous request in cypher"""
    # list of forbidden cypher commands for sefety check
    FORBIDDEN = [
        "CREATE",
        "DELETE",
        "DETACH",
        "SET",
        "DROP",
        "MERGE",
        "CALL"
    ]

    upper = state["cypher_query"].upper()

    for keyword in FORBIDDEN:
        if keyword in upper:
            return {
                "cypher_error":"forbidden dangerous request in query"
            }
        
    return {
        "cypher_error":""
    }
    
def route_after_safety_validation(state:State)->str:
    """conditional node after cypher safety assessment"""
    if state["cypher_error"] != "":
        return "unsafe_cypher"
    return "safe_cypher"

def cypher_execution(state: State)->State:
    """node to execute the cypher query or return the error if it has failed."""
    
    try:
        result = graph.query(state["cypher_query"])
        if not result:
            return {
                "cypher_result": [],
                "cypher_error": "",
            }
        return {
            "cypher_result": result,
            "cypher_error": "",
        }
    except Exception as e:
        return {
            "cypher_result": result,
            "cypher_error": str(e),
        }

MAX_RETRIES = 3

def cypher_result_validation(state: State)->State:
    """conditional node to check if there is any error by cypher execution"""
    if state["cypher_error"] != "" and state["cypher_retry_count"] <= MAX_RETRIES :
        return "cypher_unsuccessful"
    if state["cypher_retry_count"] >= MAX_RETRIES:
        state["cypher_result"] = "No data found in the graph."
    return "cypher_ok_or_no_results"


def cypher_retry(state:State)->State:
    """special node for cypher retry so that the LLM knows it has to change it"""
    
    print(f"cypher retry {state['cypher_retry_count']}")
    new_cypher = get_cypher(question=state["user_query"],entities=state["retrieved_entities"], retry=True, previous_cypher=state["cypher_query"],error=state["cypher_error"])
    return {
        "cypher_query": new_cypher.content[0]["text"],
        "cypher_retry_count": state["cypher_retry_count"] + 1
    }



def final_answer_generation(state:State)->State:
    """generation of the final answer"""
    print("final result")
    output = get_final_answer(query=state["user_query"],entities=state["retrieved_entities"],context=state["cypher_result"])
    with open("debug.json","w",encoding="utf-8") as f:
        json.dump(state,f,default=str,indent=4,ensure_ascii=False)
    print(output.content[0]["text"])
    return {
        "output":output.content[0]["text"]
    }



agent_graph = StateGraph(State)
agent_graph.add_node("extract_entities",entity_extraction)
agent_graph.add_node("assess_entities",entities_assessment)


agent_graph.add_node("retrieve_context",retrieve_context)
agent_graph.add_node("assess_context",context_assessment)
agent_graph.add_node("generate_cypher",cypher_generation)
agent_graph.add_node("validate_cypher",cypher_safety_validation)

agent_graph.add_node("retry_cypher",cypher_retry)
agent_graph.add_node("execute_cypher",cypher_execution)

agent_graph.add_node("generate_final_answer",final_answer_generation)

agent_graph.add_edge(START,"extract_entities")
agent_graph.add_edge("extract_entities","assess_entities")


agent_graph.add_conditional_edges(
    "assess_entities",
    route_after_assessment,
    {
        "entities_relevant":"retrieve_context",
        "entities_irrelevant":END 
    }
)
agent_graph.add_edge("retrieve_context","assess_context")

agent_graph.add_conditional_edges(
    "assess_context",
    route_after_context_assessment,
    {
        "context_relevant":"generate_cypher",
        "context_irrelevant":END #to-do: replace END by a node that manages irrelevant query
    }
)

agent_graph.add_conditional_edges(
    "validate_cypher",
    route_after_safety_validation,
    {
        "unsafe_cypher":"retry_cypher",
        "safe_cypher":"execute_cypher"
    }
)
agent_graph.add_edge("generate_cypher","validate_cypher")
agent_graph.add_edge("retry_cypher","validate_cypher")

agent_graph.add_conditional_edges(
    "execute_cypher",
    cypher_result_validation,
    {
        "cypher_unsuccessful":"retry_cypher",
        "cypher_ok_or_no_results":"generate_final_answer"
    }
)

app = agent_graph.compile()

# graph visualization
png_data = app.get_graph().draw_mermaid_png()


with open("langgraph_tests/graph_extraction_1.png", "wb") as f:
    f.write(png_data)

def generate_response(query):
    result = app.invoke(State(user_query=query))
    return result["output"]
