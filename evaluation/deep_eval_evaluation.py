from deepeval import evaluate
from deepeval.metrics import ContextualRelevancyMetric, AnswerRelevancyMetric, FaithfulnessMetric, GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.models import GPTModel
import json
from dotenv import load_dotenv
import os
from pprint import pprint
load_dotenv()

model = GPTModel("gpt-4.1-mini",api_key=os.getenv("OPENAI_API_KEY"))

contextual_relevancy = ContextualRelevancyMetric(model=model) # measures context - input relevance / how relevant to the input is the retrieved context ?
answer_relevancy = AnswerRelevancyMetric(model=model) # measures input - output relevance / how is the answer relevant to the user's question ?
faithfulness = FaithfulnessMetric(model=model) # measures output - context relevance / how is the answer relevant to the retrieved context

clarity = GEval(
      model=model,
      name="Clarity",
      criteria="Determine whether the output is easy to understand, logically structured, and presents ideas coherently.",
      evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT]
) # custom metric to evaluate clarity - how understandable is the answer

fluency = GEval(
      model=model,
      name="Fluency",
      criteria="Determine whether the output is well formed, grammatically and syntactically correct.",
      evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT]
) 

consistency = GEval(
      model=model,
      name="Consistency",
      criteria="Determine whether the output remains internally non-contradictory and maintains a stable tone.",
      evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT]
)

with open("interactions_to_evaluate.json", "r", encoding="utf-8") as f:
    interactions_to_evaluate = json.load(f)

test_cases = [LLMTestCase(input=interaction["user_input"],
                          actual_output=interaction["response"],
                          retrieval_context=interaction["retrieved_contexts"]) 
              for interaction in interactions_to_evaluate]


eval_result = evaluate(test_cases=test_cases,metrics=[contextual_relevancy, answer_relevancy, faithfulness, clarity])
#pprint(eval_result)

with open("evaluation/evaluation_results.json","w",encoding="utf-8") as f:
        json.dump(
            eval_result.test_results,
            f,
            ensure_ascii=False,
            default=str
        )
        f.write("\n")