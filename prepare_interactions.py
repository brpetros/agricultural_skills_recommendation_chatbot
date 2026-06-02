from typing import TypedDict, List
from log_data import Interaction
import json
from pprint import pprint

class InteractionToEvaluate(TypedDict):
    """dict for user-agent interactions for ragas evaluation"""
    user_input: str
    response: str
    retrieved_contexts: List[str]


def prepare_interaction(interaction:Interaction)->InteractionToEvaluate:
    cypher_result = interaction['cypher_result']

    retrieved_contexts = [
        "\n".join(f"{key}: {value}" for key, value in row.items())
        for row in cypher_result
    ]

    return {
        "user_input":interaction["user_query"],
        "response":interaction["output"],
        "retrieved_contexts":retrieved_contexts
    }


interactions = []

with open("interactions.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        interactions.append(json.loads(line))

interactions_to_evaluate = [prepare_interaction(interaction) for interaction in interactions]

with open("interactions_to_evaluate.json", "w", encoding="utf-8") as f:
    json.dump(interactions_to_evaluate,f,default=str)

pprint(interactions_to_evaluate)

