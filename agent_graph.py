from typing import TypedDict, List, Dict, Annotated
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_neo4j import Neo4jChatMessageHistory
from llm import llm
from context_retrieval.context_retrieval import retrieve_entity
from context_retrieval.cypher_construction import get_cypher
from pprint import pprint
import json
from skills_graph import graph
from context_retrieval.schema import InputEntity, RetrievedEntity
from final_answer import get_final_answer
from utils import get_session_id
import time
from datetime import datetime, timezone
from log_data import Interaction, save_interaction

def get_memory(session_id):
    """Returns the chat's memory by the graph"""
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

class State(TypedDict):
    """State of the graph."""
    session_id: str
    start_time: float
    messages: Annotated[List[BaseMessage],add_messages] # list of messages that gets updated easily using the langgraphs ad_message function

    user_query: str
    #input_entities: InputEntitiesSchema # the intities that the llm initially identifies at the user's query
    input_entities: List[InputEntity]
    retrieved_entities: List[RetrievedEntity] # the entities found in the graph
    
    cypher_query: str
    cypher_result: List 
    is_relevant: bool
    cypher_error: str
    cypher_retry_count: int 

    output: str

prompt = ChatPromptTemplate.from_messages([
    ("system",
    """
    You are an entity extraction system for a graph database.

    Extract occupations, skills, jobs, or locations from the current user query.
    Save each entity exactly as stated by the user. 

    Assign a category ONLY if the user provides it. 
    Example 1: if the user says that 
    the entity is a skill, you should save "skill" as the entity category. 
    Example 2: if the user says "I need information about the following occupations", 
    the entities that follow this phrase should be categorized as occupations.
    If the user does not specify a category, you MUST save "unknown" as the entity's category.
    A category can have one of the following values: 
    - "job"
    - "skill"
    - "occupation"
    - "location"
    - "unknown"

    If the query contains references such as
    'it', 'they', 'that occupation', 'the previous job',
    use the conversation history to resolve those references.

    Return ONLY structured output.
    If you do not spot a relevant entity, return an empty list.
    """),
    MessagesPlaceholder("history"),
    ("human", "{user_query}")
    ]
)

class Entities(TypedDict):
    """the structure of the LLM output - required to be in this form for the llm"""
    entities: List[InputEntity]

# chain to extract job, occuppation and skill labels from the user's query
extraction_chain = prompt | llm.with_structured_output(Entities)

def history_recovery(state:State)->State:
    session_id = get_session_id()
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
    result = extraction_chain.invoke({"user_query":state["user_query"], "history":state["messages"][-5:]}) # provides the last 5 messages to the llm + the query
    print("--spotted--")
    print(result)
    return { 
        "input_entities":result["entities"]
    }

def entities_assessment(state: State)->State:
    """Assesses if the given query is relevant (if it has entities like jobs occupations or skills)."""

    if len(state["input_entities"])==0:
        return {
            "is_relevant": False
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
    
    return {
        "retrieved_entities":[
            retrieved
            for input_entity in state["input_entities"]
            for retrieved in retrieve_entity(input_entity)
        ]
    }


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
    query = get_cypher(question=state["user_query"],entities=state["retrieved_entities"], history=state["messages"][-5:], retry=False)
    print("--returned cypher--")
    print(query.content[0]["text"])
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
                "cypher_error": "Cypher returned no results",
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

MAX_RETRIES = 2

def cypher_result_validation(state: State)->str:
    """conditional node to check if there is any error by cypher execution"""
    if state["cypher_error"] != "" and state["cypher_retry_count"] < MAX_RETRIES :
        return "cypher_unsuccessful"
    if state["cypher_retry_count"] >= MAX_RETRIES:
        return "max_retries"
    return "cypher_ok"


def cypher_retry(state:State)->State:
    """special node for cypher retry so that the LLM knows it has to change it"""
    
    print(f"cypher retry {state['cypher_retry_count']}")
    new_cypher = get_cypher(question=state["user_query"],entities=state["retrieved_entities"],history=state["messages"][-5:], retry=True, previous_cypher=state["cypher_query"],error=state["cypher_error"])
    print("--debug cypher")
    
    print(state["cypher_query"])
    print(f"errors: {state["cypher_error"]}")
    print(f"retries: {state["cypher_retry_count"]}")
    

    return {
        "cypher_query": new_cypher.content[0]["text"],
        "cypher_retry_count": state["cypher_retry_count"] + 1
    }

def no_result_answer(state:State)->State:
    return {
        "output":"It seems that I am not able to answer your question :(\n"
        "\nPossible solutions:\n"
        "- Make sure that what you are asking for is relevant to **agricultural skills, occupations and jobs**, as I am only trained "
        "for this. My information is based on the ESCO classification and Skillab job data.\n"
        "- Try to specify the type of the entities you are looking for. For example, if you are trying to find about Agricultural Business Management, "
        "specify that this is a skill and not an occupation."
    }

def final_answer_generation(state:State)->State:
    """generation of the final answer"""

    output = get_final_answer(query=state["user_query"],cypher_query=state.get("cypher_query","No information available."),context=state.get("cypher_result","No information available."),history=state["messages"][-5:])
    
    return {
        "output":output.content[0]["text"]
    }

def save(state:State)->State:
    """node to save session messages to neo4j and metadata to jsonl"""
    # saving messages permanently in the graph
    history = get_memory(state["session_id"])
    history.add_user_message(state["user_query"])
    history.add_ai_message(state["output"])

    save_interaction(Interaction(
        session_id=state.get("session_id",""),
        timestamp=datetime.now(timezone.utc).isoformat(),
        latency=time.perf_counter()-state["start_time"],
        user_query=state.get("user_query",""),
        input_entities=state.get("input_entities",[]),
        retrieved_entities=state.get("retrieved_entities",[]),
        cypher_query=state.get("cypher_query",""),
        cypher_result=state.get("cypher_result",[]),
        is_relevant=state.get("is_relevant",True),
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
        "cypher_unsuccessful":"retry_cypher",
        "max_retries":"no_result_answer",
        "cypher_ok":"generate_final_answer"
    }
)

agent_graph.add_edge("generate_final_answer","save_data")
agent_graph.add_edge("no_result_answer","save_data")
agent_graph.add_edge("save_data",END)

app = agent_graph.compile()

# graph visualization
png_data = app.get_graph().draw_mermaid_png()


with open("graph_image.png", "wb") as f:
    f.write(png_data)

def generate_response(query):
    result = app.invoke(State(user_query=query))
    return result["output"]
