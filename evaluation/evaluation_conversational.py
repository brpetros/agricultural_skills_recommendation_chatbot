from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.test_case import Turn, ConversationalTestCase, MultiTurnParams
from deepeval.metrics import TurnFaithfulnessMetric, KnowledgeRetentionMetric, TurnRelevancyMetric, ConversationalGEval, ConversationCompletenessMetric
from deepeval.models import GPTModel
import json
from pprint import pprint 
from dotenv import load_dotenv
import os
from collections import defaultdict
load_dotenv()

model = GPTModel("gpt-4.1-mini",api_key=os.getenv("OPENAI_API_KEY"),temperature=0)

with open("interactions_to_evaluate.json", "r", encoding="utf-8") as f:
    interactions_to_evaluate = json.load(f)



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
            expected_outcome="The chatbot has to answer any questions regarding agricultural skills, occupations, or jobs and any relation between them, based on the database's context."
        )
        for turn_collection in turn_collections.values()
    ]
    return test_cases

    



faithfulness = TurnFaithfulnessMetric(model=model) 
knowledge_retention =  KnowledgeRetentionMetric(model=model) 
relevancy = TurnRelevancyMetric(model=model) 
clarity = ConversationalGEval(
    name="Clarity",
    criteria="Evaluate whether the assistant's responses stay clear, easy to understand, well-structured, and free from unnecessary complexity or ambiguity through the conversation.",
    evaluation_params=[MultiTurnParams.CONTENT],
    model=model
) 

turn_collections = generate_turns(interactions_to_evaluate)
test_cases = generate_test_cases(turn_collections)

evaluate(test_cases=test_cases,metrics=[faithfulness, knowledge_retention, relevancy, clarity],async_config=AsyncConfig(max_concurrent=2))
