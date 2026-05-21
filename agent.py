from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import Tool
from langchain_neo4j import Neo4jChatMessageHistory
from langchain_classic.agents import  create_react_agent, AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory
from utils import get_session_id
from skills_graph import graph
from llm import llm
from langchain_core.prompts import PromptTemplate
from cypher_qa import cypher_qa
import json
from datetime import datetime
from pprint import pprint
import vectors

def serialize_steps(steps):
    clean = []

    for action, observation in steps:

        clean.append({
            "tool": getattr(action, "tool", None),
            "tool_input": getattr(action, "tool_input", None),
            "log": getattr(action, "log", None),

            # IMPORTANT: your tool returns dict → store it safely
            "result": observation.get("result") if isinstance(observation, dict) else observation,

            "query": observation.get("query") if isinstance(observation, dict) else None
        })

    return clean


def create_record(user_input, response, session_id):
    raw_steps = response["intermediate_steps"]

    # flatten one level
    flat_steps = raw_steps[0] if raw_steps and isinstance(raw_steps[0], list) else raw_steps
    
    record = {
        "timestamp":str(datetime.now()),
        "user_input":user_input,
        "session_id":session_id,
        "intermediate_steps":serialize_steps(flat_steps),
        "response":response['output']
    }
    pprint(f"----record----\n{record}")

    with open("execution_logs.jsonl","a",encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    return record



def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are an expert in agriculture. You provide recommendations about skills that are related to agriculture."),
        ("human","{input}")
    ]
)

agricultural_chat = chat_prompt | llm | StrOutputParser()


tools = [
    Tool.from_function(
        name="graph_information",
        description="Use this tool to provide information related to agricultural skills, occupations and jobs and their relations. Input should always be natural languade only.",
        func=cypher_qa
    ),
    Tool.from_function(
        name="occupation_label_match",
        description="Use this tool to find information about similar occupations based on the occupation's label.",
        func=vectors.get_occupation_label
    ),
    Tool.from_function(
        name="skill_label_match",
        description="Use this tool to find information about similar skills based on the skill's label.",
        func=vectors.get_skill_label
    ),
    Tool.from_function(
        name="job_title_match",
        description="Use this tool to find information about similar job offers based on the skill's label.",
        func=vectors.get_job_title
    ),
]

# prompt template that give the agent instructions related to the required train of thoughts.
# gives the agent the available tools options and instructs on how to choose one of the tools.
agent_prompt = PromptTemplate.from_template(
    """
    You are an expert in agriculture, providing informations about Occupations, Jobs and Skills related to agriculture. 
    Be accurate and return as much information as possible.
    Do not answer any questions that are not relevant to agricultural skills, occupations and jobs.
    Do not answer any questions using your pre-trained knowledge. Use only the information provided by the context.

    #TOOLS:

    You have access to the following tools.

    {tools}

    To use a tool, you have to use the following format:
    ```
    Though: Do I need to use a tool? Yes
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ```
    If you have context, you have to give a response. 
    If you have no context provided by the tools, just say that you don't know.
    To give the final answer, you have to use the following format.

    ```
    Thought: Do I need to use a tool? No
    Final Answer: [your response here]
    ```

    Begin.

    Previous conversation history:
    {chat_history}

    New input: {input}
    {agent_scratchpad}
    """
)


agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True
    )

chat_agent = RunnableWithMessageHistory(
    agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)


def generate_response(user_input):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI
    """

    response = chat_agent.invoke(
        {"input": user_input},
        {"configurable": {"session_id": get_session_id()}},)
    
    response_record = create_record(user_input, response, get_session_id()) 

    return response["output"]