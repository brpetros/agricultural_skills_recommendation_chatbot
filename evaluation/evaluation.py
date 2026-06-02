from langchain_openai import ChatOpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from openai import AsyncOpenAI
import numpy as np
import os
from dotenv import load_dotenv
from ragas import EvaluationDataset, experiment
from ragas.metrics.collections import ContextRecall, Faithfulness, FactualCorrectness
import json
import asyncio
load_dotenv()


with open("interactions_to_evaluate.json", "r", encoding="utf-8") as f:
    interactions_to_evaluate = json.load(f)
print([type(interaction["retrieved_contexts"]) for interaction in interactions_to_evaluate])
for interaction in interactions_to_evaluate:
    for context in interaction["retrieved_contexts"]:
        print(type(context))
#llm = ChatOpenAI(model="gpt-4.1-mini",api_key=os.getenv("OPENAI_API_KEY"))
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
evaluator_llm = llm_factory('gpt-4.1-mini', client=client)


# initializing the metrics
factual_correctness = FactualCorrectness(llm=evaluator_llm)
faithfulness = Faithfulness(llm=evaluator_llm)
context_recall = ContextRecall(llm=evaluator_llm)

async def evaluate(interaction):
    user_input = interaction["user_input"]
    response = interaction["response"]
    retrieved_contexts = interaction["retrieved_contexts"]


    faithfulness_score = await faithfulness.ascore(
        user_input=user_input,
        response=response,
        retrieved_contexts=retrieved_contexts
    )

    return {
        "query":user_input,
        "response":response,
        "faithfulness":faithfulness_score
    }



#evaluation_result = [evaluate(interaction) for interaction in interactions_to_evaluate]

async def evaluate_set(interactions_to_evaluate):
    evaluation_result = await asyncio.gather(
        *[evaluate(interaction) for interaction in interactions_to_evaluate]
    )
    
    with open("evaluation/evaluation_results.json","w",encoding="utf-8") as f:
        json.dump(
            evaluation_result,
            f,
            indent=4,
            ensure_ascii=False,
            default=str
        )
        f.write("\n")

#asyncio.run(evaluate_set(interactions_to_evaluate))