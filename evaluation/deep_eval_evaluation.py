from deepeval import evaluate
from deepeval.metrics import ContextualRelevancyMetric, AnswerRelevancyMetric, FaithfulnessMetric, GEval, ContextualRecallMetric
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.models import GPTModel
import json
from dotenv import load_dotenv
import os
from pprint import pprint
load_dotenv()

model = GPTModel("gpt-4.1-mini",api_key=os.getenv("OPENAI_API_KEY"),temperature=0)

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

"""with open("interactions_to_evaluate.json", "r", encoding="utf-8") as f:
    interactions_to_evaluate = json.load(f)"""
"""
test_cases = [LLMTestCase(input=interaction["user_input"],
                          actual_output=interaction["response"],
                          retrieval_context=interaction["retrieved_contexts"],
                          expected_output=interaction.get("expected_output","no expected output")) 
              for interaction in interactions_to_evaluate]"""


test_interaction_1 = {
      "user_input":"I need information about the skill \"agronomical production principles\"",
      "actual_output":"The skill \"agronomical production principles\" is a knowledge-based competency covering the techniques, methods, and principles used in conventional agricultural production, including crop management, soil cultivation, irrigation, fertilization, and pest control.", # invoke the llm chain
      "retrieval_context":[
            "label: \"agronomical production principles\"",
            "skillType: \"knowledge\"",
            "description: \"The techniques, methods and principles of conventional agronomical production.\"",
      ],
      "expected_output":"The skill \"agronomical production principles\" is a knowledge-based competency covering the techniques, methods, and principles used in conventional agricultural production" # query the graph to see what should be retrieved -> ask the llm to turn it into a string
}

test_eval = LLMTestCase(
      input=test_interaction_1["user_input"],
      actual_output=test_interaction_1["actual_output"],
      retrieval_context=test_interaction_1["retrieval_context"],
      expected_output=test_interaction_1["expected_output"]
)


evaluate(test_cases=[test_eval],metrics=[answer_relevancy, faithfulness, contextual_recall, clarity, consistency])

