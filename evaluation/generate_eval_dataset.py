import json
from typing import TypedDict, List
from log_data import Interaction
from flashrank import Ranker, RerankRequest

_FLASH_RANKER = Ranker(model_name="ms-marco-MiniLM-L-12-v2")


class FinalInteraction(TypedDict):
    session_id: str
    input: str
    actual_output: str
    retrieval_context: List[str]

def compress_graph_context(query: str, raw_strings: list[str], top_n: int = 5) -> list[str]:
    """
    Takes a query and a flat list of text strings retrieved from Neo4j,
    uses FlashRank to score them, and returns only the top N strings.
    """
    if not raw_strings:
        return []

    # 1. FlashRank requires an ID and a text key for each input chunk
    passages = [
        {"id": idx, "text": text_string}
        for idx, text_string in enumerate(raw_strings)
    ]

    # 2. Package the request
    request = RerankRequest(query=query, passages=passages)

    # 3. Process the cross-encoder sorting execution
    ranked_result = _FLASH_RANKER.rerank(request)

    # 4. Strip the FlashRank score metadata and return just the raw text strings
    return [item["text"] for item in ranked_result[:top_n]]


def prepare_interaction(interaction:Interaction)->FinalInteraction:
    """saves interaction in a way to be appropriate for evaluation with deepeval"""
    cypher_result = interaction['cypher_result']

    retrieved_contexts = [
        "\n".join(f"{key}: {value}" for key, value in row.items())
        for row in cypher_result
    ]
    
    total_character_count = sum(len(c) for c in retrieved_contexts)
    
    # THRESHOLD LIMIT: 20,000 characters
    CHARACTER_THRESHOLD = 20000 
    
    if total_character_count > CHARACTER_THRESHOLD:
        print(f"[Eval Warning] Context size ({total_character_count} chars) exceeds threshold. Applying FlashRank...")
        eval_context = compress_graph_context(
            query=interaction["user_query"], 
            raw_strings=retrieved_contexts, 
            top_n=15
        )
    else:
        # If context size is small and safe, use it in its entirety
        eval_context = retrieved_contexts
    
    return {
        "session_id":interaction["session_id"],
        "input":interaction["user_query"],
        "actual_output":interaction["output"],
        "retrieval_context":eval_context
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

