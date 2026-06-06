from typing import TypedDict, List
from context_retrieval.schema import RetrievedEntity, InputEntity
from datetime import datetime
import json
from pathlib import Path

# interaction to be saved
class Interaction(TypedDict):
    session_id: str
    timestamp: datetime
    latency: float

    user_query: str
    input_entities: List[InputEntity] # the intities that the llm initially identifies at the user's query
    retrieved_entities: List[RetrievedEntity] # the entities found in the graph

    cypher_query: str
    cypher_result: List 
    is_relevant: bool
    cypher_error: str
    cypher_retry_count: int 

    output: str

INTERACTIONS_FILE = Path("chatbot_logs.jsonl")
def save_interaction(interaction: Interaction):
    with INTERACTIONS_FILE.open("a", encoding="utf-8") as f:
        json.dump(
            interaction,
            f,
            ensure_ascii=False,
            default=str
        )
        f.write("\n")