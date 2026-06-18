from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.test_case import Turn, ConversationalTestCase, MultiTurnParams
from deepeval.metrics import TurnFaithfulnessMetric, KnowledgeRetentionMetric, TurnRelevancyMetric, ConversationalGEval, ConversationCompletenessMetric
from deepeval.models import GPTModel
import csv
import json
from pprint import pprint 
from dotenv import load_dotenv
import os
from collections import defaultdict
load_dotenv()

model = GPTModel("gpt-4.1-mini",api_key=os.getenv("OPENAI_API_KEY"),temperature=0)

def generate_turns(interactions_to_evaluate):
    """generates the turns that will be used in the test cases"""
    turns = defaultdict(list)
    for interaction in interactions_to_evaluate:
        session_id = interaction["session_id"]
        turns[session_id].append(
            Turn(role="user", 
                content=interaction["input"])
        )
        turns[session_id].append(
            Turn(
                role="assistant",
                content=interaction["actual_output"],
                retrieval_context=interaction["retrieval_context"]
            )
        )
    return turns 

def generate_test_cases(turn_collections):
    """generates the test cases for the evaluation"""
    test_cases = [
        ConversationalTestCase(
            turns=turn_collection, 
            window_size=5,
        )
        for turn_collection in turn_collections.values()
    ]
    return test_cases

    



def conversational_evaluation():
    os.environ["DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE"] = "900"
    os.environ["DEEPEVAL_RESULTS_FOLDER"] = "./evaluation_logs_conversational"

    with open("evaluation/interactions_to_evaluate.json", "r", encoding="utf-8") as f:
        interactions_to_evaluate = json.load(f)
    turn_collections = generate_turns(interactions_to_evaluate)
    test_cases = generate_test_cases(turn_collections)

    faithfulness = TurnFaithfulnessMetric(model=model) 
    knowledge_retention =  KnowledgeRetentionMetric(model=model) 
    relevancy = TurnRelevancyMetric(model=model) 
    completeness = ConversationCompletenessMetric(model=model)

    evaluate(test_cases=test_cases,metrics=[faithfulness, knowledge_retention, relevancy, completeness],async_config=AsyncConfig(run_async=True, max_concurrent=2))
    


"""
def conversational_evaluation():
    os.environ["DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE"] = "900"
    
    with open("evaluation/interactions_to_evaluate.json", "r", encoding="utf-8") as f:
        interactions_to_evaluate = json.load(f)
        
    turn_collections = generate_turns(interactions_to_evaluate)
    test_cases = generate_test_cases(turn_collections)

    # Initialize metrics
    faithfulness = TurnFaithfulnessMetric(model=model, verbose_mode=True) 
    knowledge_retention = KnowledgeRetentionMetric(model=model, verbose_mode=True) 
    relevancy = TurnRelevancyMetric(model=model, verbose_mode=True) 
    completeness = ConversationCompletenessMetric(model=model, verbose_mode=True)

    # List to store data for saving later
    evaluation_results = []

    for idx, test_case in enumerate(test_cases):
        print(f"\n=========================================")
        print(f"Evaluating Test Case {idx + 1}/{len(test_cases)}")
        print(f"=========================================")

        # 1. Faithfulness
        print("\n--- Running Faithfulness ---")
        faithfulness.measure(test_case)
        faithfulness_score = faithfulness.score
        faithfulness_reason = faithfulness.reason 

        # 2. Knowledge Retention
        print("\n--- Running Knowledge Retention ---")
        knowledge_retention.measure(test_case)
        kr_score = knowledge_retention.score
        kr_reason = knowledge_retention.reason 

        # 3. Relevancy
        print("\n--- Running Relevancy ---")
        relevancy.measure(test_case)
        relevancy_score = relevancy.score
        relevancy_reason = relevancy.reason

        # 4. Completeness
        print("\n--- Running Completeness ---")
        completeness.measure(test_case)
        completeness_score = completeness.score
        completeness_reason = completeness.reason

        # Gather metrics into a single test case record
        record = {
            "test_case_index": idx + 1,
            "faithfulness_score": faithfulness_score,
            "faithfulness_reason": faithfulness_reason,
            "knowledge_retention_score": kr_score,
            "knowledge_retention_reason": kr_reason,
            "relevancy_score": relevancy_score,
            "relevancy_reason": relevancy_reason,
            "completeness_score": completeness_score,
            "completeness_reason": completeness_reason
        }
        
        evaluation_results.append(record)
        
        # [SAFEGUARD]: Save incrementally so you don't lose data if the script stops mid-way
        save_results_backup(evaluation_results)

    # Final explicit save when all cases finish completely
    save_results_final(evaluation_results)

# --- SAFE SAVING FUNCTIONS ---

def save_results_backup(results, filename="evaluation/results_checkpoint.json"):
    
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Warning: Failed to save checkpoint: {e}")

def save_results_final(results, csv_filename="evaluation/final_report.csv"):
    
    if not results:
        return
    
    keys = results[0].keys()
    with open(csv_filename, "w", newline="", encoding="utf-8") as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)
    print(f"\n🎉 Success! Final spreadsheet saved to: {csv_filename}")
    """