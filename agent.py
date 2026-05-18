from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.tools import Tool
from langchain_neo4j import Neo4jChatMessageHistory
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from utils import get_session_id
from skills_graph import graph
from llm import llm
from langchain_core.prompts import PromptTemplate
from cypher_qa import cypher_qa

def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are an expert in agriculture. You provide recommendations about skills that are related to agriculture."),
        ("human","{input}")
    ]
)

agricultural_chat = chat_prompt | llm | StrOutputParser()

def simple_agricultural_chat():
    return agricultural_chat.invoke
tools = [
    Tool.from_function(
        name="general_chat",
        description="For general chat related to agriculture not covered by other tools",
        func=simple_agricultural_chat()
    ),
    Tool.from_function(
        name="graph_information",
        description="Provide information related to agricultural skills, occupations and jobs using Cypher.",
        func=cypher_qa
    )
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
    When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

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
    verbose=True
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

    return response['output']