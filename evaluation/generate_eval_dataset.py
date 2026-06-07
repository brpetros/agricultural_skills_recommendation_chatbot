import json
from typing import TypedDict, List
from log_data import Interaction

class FinalInteraction(TypedDict):
    session_id: str
    input: str
    actual_output: str
    retrieval_context: List[str]



def prepare_interaction(interaction:Interaction)->FinalInteraction:
    """saves interaction in a way to be appropriate for evaluation with deepeval"""
    cypher_result = interaction['cypher_result']

    retrieved_contexts = [
        "\n".join(f"{key}: {value}" for key, value in row.items())
        for row in cypher_result
    ]
    
    return {
        "session_id":interaction["session_id"],
        "input":interaction["user_query"],
        "actual_output":interaction["output"],
        "retrieval_context":retrieved_contexts
    }

def save_interactions(sessions_to_save:str):
    """
    generation of the test interactions and creation of evaluation dataset
    
    Params: 
    
    sessions_to_save: string with how the session id of the interactions that we wish to evalaute stars.
    For example: "occupation testing for" -> the interactions which have a session id that starts with this string will be saved to be evaluated.
    """
    #generate_responses_test_1(skills_to_evaluate=skills_to_evaluate)
    interactions = []
    with open("chatbot_logs.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            interactions.append(json.loads(line))

    interactions_to_evaluate = [prepare_interaction(interaction) for interaction in interactions if interaction["session_id"].startswith(sessions_to_save)]

    with open("evaluation/interactions_to_evaluate.json","w",encoding="utf-8") as f:
        json.dump(
            interactions_to_evaluate,
            f,
            indent=4,
            ensure_ascii=False
        )
    print("interactions saved")

save_interactions("single questions testing")