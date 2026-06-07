from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.metrics import ContextualRelevancyMetric, AnswerRelevancyMetric, FaithfulnessMetric, GEval, ContextualRecallMetric
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.models import GPTModel
import json
from dotenv import load_dotenv
import os
from pprint import pprint
load_dotenv()

model = GPTModel("gpt-4.1-mini",api_key=os.getenv("OPENAI_API_KEY"),temperature=0)


answer_relevancy = AnswerRelevancyMetric(model=model) # measures input - output relevance / how is the answer relevant to the user's question ?
faithfulness = FaithfulnessMetric(model=model) # measures output - context relevance / how is the answer relevant to the retrieved context

clarity = GEval(
      model=model,
      name="Clarity",
      criteria="Determine whether the output is easy to understand, logically structured, and presents ideas coherently.",
      evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT]
) # custom metric to evaluate clarity - how understandable is the answer

consistency = GEval(
      model=model,
      name="Consistency",
      criteria="Determine whether the output remains internally non-contradictory and maintains a stable tone.",
      evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT]
)

with open("interactions_to_evaluate.json", "r", encoding="utf-8") as f:
    interactions_to_evaluate = json.load(f)

test_cases = [LLMTestCase(input=interaction["input"],
                          actual_output=interaction["actual_output"],
                          retrieval_context=interaction["retrieval_context"]
                          ) 
              for interaction in interactions_to_evaluate]



evaluate(test_cases=test_cases,metrics=[answer_relevancy, faithfulness, clarity, consistency],async_config=AsyncConfig(max_concurrent=2))

