from typing import TypedDict, List, Dict, Annotated
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_neo4j import Neo4jChatMessageHistory
from context_retrieval.semantic_extraction import extract_segments
from context_retrieval.context_retrieval import retrieve_entities
from context_retrieval.cypher_construction import get_cypher
from pprint import pprint
from graph_db import graph
from final_answer import get_final_answer
import time
from datetime import datetime, timezone
from log_data import Interaction, save_interaction
from suggest_alternatives import suggest_alternatives

def get_memory(session_id):
    """Returns the chat's memory by the graph"""
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

class State(TypedDict):
    """State of the graph."""
    session_id: str
    start_time: float
    messages: Annotated[List[BaseMessage],add_messages] # list of messages that gets updated easily using the langgraphs ad_message function

    user_query: str
    
    input_entities: List[str]
    retrieved_entities: Dict # the entities found in the graph
    
    cypher_query: str
    cypher_result: List 
    entities_relevant: bool
    cypher_error: str
    cypher_retry_count: int 

    output: str




def history_recovery(state:State)->State:
    session_id = state["session_id"] 
    history = get_memory(session_id)
    return {
        "session_id": session_id,
        "messages": history.messages + [HumanMessage(content=state["user_query"])],
        "start_time": time.perf_counter()
    }
    

def entity_extraction(state: State)->State:
    """
        Node for entity esxtraction.
        Instructs the LLM to extract the entities from the user's query and categorize them to occupations, skills and jobs.
    """
    result = extract_segments(user_input=state["user_query"],history=state["messages"][-5:])
    return { 
        "input_entities":result["segments"]
    }

def entities_assessment(state: State)->State:
    """Assesses if the given query is relevant (if it has entities like jobs occupations or skills)."""

    if len(state["input_entities"])==0:
        return {
            "entities_relevant": False,
            "cypher_error": ""
        }
    return {
        "entities_relevant": True,
        "cypher_error": ""
    }

def route_after_assessment(state: State)->str:
    """conditinal node after entities assessment"""
    if state["entities_relevant"]:
        return "entities_relevant"
    else:
        return "entities_irrelevant"

def retrieve_context(state: State)->State:
    """
        Node for context retrieval.
        Performs vector search to spot the extracted entities in the graph.
    """
    
    return {
        "retrieved_entities": retrieve_entities(state["input_entities"])
    }


def context_assessment(state:State)->State:
    """node to assess the score of the retrieved context (if it is actually relevant)"""
    # to-do: expand this node so that it actually checks the retrieved results for their relevance 
    return {
        "entities_relevant":True
    }

def route_after_context_assessment(state:State)->str:
    """routes after context assessment"""
    if state["entities_relevant"]:
        return "context_relevant"
    else:
        return "context_irrelevant"

def cypher_generation(state: State) -> State:
    """node for cypher generation based on retrieved entities and query"""
    
    # if we need to retry we have the cypher retry node
    query = get_cypher(question=state["user_query"],entities=state["retrieved_entities"], history=state["messages"][-5:], retry=False)

    return {
        "cypher_query": query.content[0]["text"],
        "cypher_retry_count": 0
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

MAX_RETRIES = 2

def route_after_safety_validation(state:State)->str:
    """conditional node after cypher safety assessment"""
    if state["cypher_error"] != "" :
        return "unsafe_cypher"
    return "safe_cypher"

def cypher_execution(state: State)->State:
    """node to execute the cypher query or return the error if it has failed."""
    
    if state["cypher_retry_count"] >= MAX_RETRIES:
        return {
            "cypher_result": [],
            "cypher_error": "No relevant data was found in the database.",
        }

    try:
        result = graph.query(state["cypher_query"])
        if not result:
            return {
                "cypher_result": [],
                "cypher_error": "No relevant data was found in the database.",
            }
        return {
            "cypher_result": result,
            "cypher_error": "",
        }
    except Exception as e:
        return {
            "cypher_result": [],
            "cypher_error": str(e),
        }


def cypher_result_validation(state: State)->str:
    """conditional node to check if there is any error by cypher execution"""
    if state["cypher_error"] != "" and state["cypher_retry_count"] < MAX_RETRIES :
        return "cypher_unsuccessful"
    if state["cypher_retry_count"] >= MAX_RETRIES:
        return "max_retries"
    return "cypher_ok"


def cypher_retry(state:State)->State:
    """special node for cypher retry so that the LLM knows it has to change it"""
    
    if state["cypher_retry_count"] >= MAX_RETRIES:
        return {
            "cypher_query":"maximum retries"
        }
    new_cypher = get_cypher(question=state["user_query"],entities=state["retrieved_entities"],history=state["messages"][-5:], retry=True, previous_cypher=state["cypher_query"],error=state["cypher_error"])
    

    return {
        "cypher_query": new_cypher.content[0]["text"],
        "cypher_retry_count": state["cypher_retry_count"] + 1
    }

def alternatives_suggestion(state:State)->State:
    """suggestion of alternative search paths if no results were found"""
    # in case of no cypher result, the retrieved entities will be saved as cypher result, so they can be saved as context.
    context = []
    for values in state["retrieved_entities"].values():
        context.extend(values)
    return {
        "output": suggest_alternatives(user_input=state["user_query"],cypher_query=state["cypher_query"],retrieved_entities=state["retrieved_entities"],history=state["messages"][-5:]),
        "cypher_result": context
    }

def no_result_answer(state:State)->State:
    
    output = """
    It seems that I am not able to answer :(\n
    This is probably because what you are asking for is not relevant to agricultural skills, jobs, or occupations.\n
    I am only trained to answer this kind of questions, based on ESCO and SKILLAB.\n
    If this is not the case, please try to say what you want differently.\n
    Thank you!
    """
    if state["cypher_error"]!="":
       output = f"""{state["cypher_error"]}
                If you think that this should not be the case, please try the following:
                - Make sure that what you are asking for is relevant to **agricultural skills, occupations and jobs**, as I am only trained for this. My information is based on the ESCO classification and Skillab job data.
                - Try to specify the type of what you are looking for. For example, if you are trying to find information about Agricultural Business Management, specify that it is a skill and not an occupation.
                - Change the terms that you used or the order of your phrasing.
                """
    return {
        "output":output
    }

def final_answer_generation(state:State)->State:
    """generation of the final answer"""
    return {
        "output":get_final_answer(query=state["user_query"],cypher_query=state.get("cypher_query","No information available."),context=state.get("cypher_result","No information available."),history=state["messages"][-5:])
    }
    


def save(state:State)->State:
    """node to save session messages to neo4j and metadata to jsonl"""
    # saving messages permanently in the graph
    history = get_memory(state["session_id"])
    history.add_user_message(state["user_query"])
    history.add_ai_message(state["output"])
    
    print("---interaction logs---")
    print(f"Input: {state["user_query"]}")
    print(f"Retrieved Entities:")
    pprint(state["retrieved_entities"])
    print("\n\n")
    print(f"cypher: {state['cypher_query']}")
    print(f"Cypher retries: {state['cypher_retry_count']}\n\n")
    print(f"Context: {state["cypher_result"]}\n\n")
    print(f"Output: {state['output']}")

    save_interaction(Interaction(
        session_id=state.get("session_id",""),
        timestamp=datetime.now(timezone.utc).isoformat(),
        latency=time.perf_counter()-state["start_time"],
        user_query=state.get("user_query",""),
        input_entities=state.get("input_entities",[]),
        retrieved_entities=state.get("retrieved_entities",[]),
        cypher_query=state.get("cypher_query",""),
        cypher_result=state.get("cypher_result",[]),
        entities_relevant=state.get("entities_relevant",True),
        cypher_error=state.get("cypher_error",""),
        cypher_retry_count=state.get("cypher_retry_count",0),
        output=state.get("output","")
    ))
    return state


agent_graph = StateGraph(State)
agent_graph.add_node("history_recovery",history_recovery)
agent_graph.add_node("extract_entities",entity_extraction)
agent_graph.add_node("assess_entities",entities_assessment)


agent_graph.add_node("retrieve_context",retrieve_context)
agent_graph.add_node("assess_context",context_assessment)
agent_graph.add_node("generate_cypher",cypher_generation)
agent_graph.add_node("validate_cypher",cypher_safety_validation)

agent_graph.add_node("retry_cypher",cypher_retry)
agent_graph.add_node("execute_cypher",cypher_execution)

agent_graph.add_node("generate_final_answer",final_answer_generation)
agent_graph.add_node("suggest_alternatives",alternatives_suggestion)
agent_graph.add_node("no_result_answer",no_result_answer)
agent_graph.add_node("save_data",save)

agent_graph.add_edge(START,"history_recovery")
agent_graph.add_edge("history_recovery","extract_entities")
agent_graph.add_edge("extract_entities","assess_entities")


agent_graph.add_conditional_edges(
    "assess_entities",
    route_after_assessment,
    {
        "entities_relevant":"retrieve_context",
        "entities_irrelevant":"no_result_answer" 
    }
)
agent_graph.add_edge("retrieve_context","assess_context")

agent_graph.add_conditional_edges(
    "assess_context",
    route_after_context_assessment,
    {
        "context_relevant":"generate_cypher",
        "context_irrelevant":"no_result_answer"
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
        "cypher_ok":"generate_final_answer",
        "cypher_unsuccessful":"retry_cypher",
        "max_retries":"suggest_alternatives"
    }
)

agent_graph.add_edge("generate_final_answer","save_data")
agent_graph.add_edge("no_result_answer","save_data")
agent_graph.add_edge("suggest_alternatives","save_data")
agent_graph.add_edge("save_data",END)

app = agent_graph.compile()

# graph visualization
png_data = app.get_graph().draw_mermaid_png()


with open("graph_image.png", "wb") as f:
    f.write(png_data)

def generate_response(session_id,query):
    result = app.invoke(State(session_id=session_id,user_query=query))
    return result["output"]
