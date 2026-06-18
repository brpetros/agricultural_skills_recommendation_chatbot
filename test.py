from evaluation.generate_evaluation_responses import queries_occupations_related, queries_job_related, queries_skills_related, test_conversations_jobs,test_conversations_occupations, test_conversations_skills, generate_single_interactions, generate_conversational_interactions
from evaluation.generate_eval_dataset import save_interactions
from context_retrieval.semantic_extraction import extract_segments
from context_retrieval.context_retrieval import retrieve_entities
from evaluation.evaluation_single import evaluation
from evaluation.evaluation_conversational import conversational_evaluation
from pprint import pprint
import json

#outputs = generate_single_interactions("17-6 single-turn tests skills",queries=queries_skills_related)
#res = generate_conversational_interactions("17-6 multi-turn tests jobs",test_conversations_jobs)
#print(res)
#save_interactions("17-6 multi-turn tests occupations")
#evaluation()
conversational_evaluation()

#entities = retrieve_entities(["bilingual or multilingual","judging the genetic potential of breeding cattle","complete operational control over a commercial orchard","operating mechanical harvesters","Lindesberg, Sweden","Thessaloniki, Greece","Lantbrukare med huvudansvar för jordbruk och djurhållning"])

#pprint(entities)

