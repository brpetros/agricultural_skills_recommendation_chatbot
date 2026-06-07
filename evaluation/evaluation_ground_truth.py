from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.models import GPTModel
import json
from dotenv import load_dotenv
import os
from pprint import pprint
load_dotenv()

model = GPTModel("gpt-4.1-mini",api_key=os.getenv("OPENAI_API_KEY"),temperature=0)

answer_relevancy = AnswerRelevancyMetric(model=model) # measures input - output relevance / how is the answer relevant to the user's question ?
faithfulness = FaithfulnessMetric(model=model)
contextual_recall = ContextualRecallMetric(model=model)
contextual_precision = ContextualPrecisionMetric(model=model)

with open("interactions_to_evaluate.json", "r", encoding="utf-8") as f:
    interactions_to_evaluate = json.load(f)

test_cases = [LLMTestCase(input=interaction["input"],
                          actual_output=interaction["actual_output"],
                          retrieval_context=interaction["retrieval_context"],
                          expected_output=interaction["expected_output"]
                          ) 
              for interaction in interactions_to_evaluate]

evaluate(test_cases=test_cases,metrics=[answer_relevancy,faithfulness,contextual_recall,contextual_precision],async_config=AsyncConfig(max_concurrent=2))