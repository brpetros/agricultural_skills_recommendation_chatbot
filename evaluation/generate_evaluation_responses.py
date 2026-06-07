from agent_graph import generate_response
from typing import TypedDict, List
from log_data import Interaction
import json



def generate_test_responses(session_id,skill):
    output_1 = generate_response(session_id,f"I need information about the following skill: {skill}") # response and metadata are saved in the jsonl new_interactions
    print(f"output for skill {skill}: {output_1}")
    output_2 = generate_response(session_id,"What occupations require this skill?")
    print(output_2)
    output_3 = generate_response(session_id,"I am interested in the first occupation you mentioned. Are there any jobs that contain it? ")
    print(output_3)
    output_4 = generate_response(session_id,"Are there any other skills that this occupation requires?")
    print(output_4)
    
def generate_multiple_responses(skills_to_evaluate):
    for skill in skills_to_evaluate:
        generate_test_responses(f"eval_test for {skill}",skill)

def generate_test_responses_1(session_id,occupation):
    output_1 = generate_response(session_id,f"I am a {occupation}. I wish to change my career. Tell me what paths I could follow with the skills that are essential for what I do.") # response and metadata are saved in the jsonl new_interactions
    print(f"output for occupation {occupation}: {output_1}")
    output_2 = generate_response(session_id,"In order to pursue the last path that you proposed, what are the important skills that I should focus on?")
    print(output_2)
    output_3 = generate_response(session_id,"I think I can manage this. Is there any job relevant to this occupation?")
    print(output_3)
    
    
occupations_to_evaluate = [
    "vineyard supervisor",
    "crop production manager",
    "pig breeder",
    "fruit production team leader"
]

def generate_multiple_response_occs(occupations_to_evaluate):
    for occupation in occupations_to_evaluate:
        generate_test_responses(f"occupation testing for {occupation}",occupation)