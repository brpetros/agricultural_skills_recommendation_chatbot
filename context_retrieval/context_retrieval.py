from typing import List, Dict, TypedDict, Tuple
from langchain_core.documents import Document
from context_retrieval.vectors import search_vector
from context_retrieval.schema import RetrievedEntity, InputSegment
from pprint import pprint

def document_to_schema(doc: Document, score: float) -> RetrievedEntity:
    """converts a langchain document into a structured RetrievedEntity schema"""

    metadata = doc.metadata or {} 

    return {
        "label": doc.page_content,
        "description": metadata.get("description","")[:300],
        "id": metadata.get("id",""),
        "type": metadata.get("type",""),
        "score": score
    }


@DeprecationWarning
def get_jobs(input:InputSegment, k:int = 1) -> List[RetrievedEntity]:
    '''retrurns list of retrieved jobs based on title + description'''
    results = job_vector.similarity_search_with_score(query=input["original_input"],k=k)
    print("--retrieved jobs")
    print(results)
    return [document_to_schema(input,doc,score) for doc,score in results]

@DeprecationWarning
def get_occupations(input:InputSegment, k:int = 1) -> List[RetrievedEntity]:
    '''retrurns list of retrieved occupations based on label + description'''
    results = occupation_vector.similarity_search_with_score(query=input["original_input"],k=k)
    print("--retrieved occupations")
    print(results)
    return [document_to_schema(input,doc,score) for doc,score in results]

@DeprecationWarning
def get_skills(input:InputSegment, k:int = 1) -> List[RetrievedEntity]:
    '''retrurns list of retrieved skills based on label + description'''
    results = skill_vector.similarity_search_with_score(query=input["original_input"],k=k)
    print("--retrieved jobs")
    print(results)
    return [document_to_schema(input,doc,score) for doc,score in results]

@DeprecationWarning
def get_location(input:InputSegment)->List[RetrievedEntity]:
    """returns the location exactly as it is in the input. Need to update by adding vector search."""
    return [{
        "original_input": input["original_input"],
        "category":input["category"],
        "label": "",
        "description": "",
        "id": "",
        "type": "",
        "score": 0
    }]

# dict of the search functions used depending on each entity's category
SEARCH_FUNCTIONS = {
    "occupation":get_occupations,
    "job":get_jobs,
    "skill":get_skills,
    "location":get_location
}
@DeprecationWarning 
def retrieve_entity(entity:InputSegment,k:int = 1)->List[RetrievedEntity]:
    """
    Runs the correct search function depending on the entity's category.
    If it is job or occupation, gets the one with the best score.
    If the category is unknown, runs all 3 search functions and returns the result with the best score.

    replaced by retrieve_entities
    """
    category = entity["category"]
    candidates = []
    if category in ["job","occupation"]:
        candidates.extend(get_occupations(entity))
        candidates.extend(get_jobs(entity))  
    elif category == "unknown" or category not in SEARCH_FUNCTIONS.keys():
        for search_function in SEARCH_FUNCTIONS.values():
            candidates.extend(search_function(entity))   
    elif category in SEARCH_FUNCTIONS.keys():
        candidates.extend( SEARCH_FUNCTIONS[category](entity))
    
    if not candidates:
        return []
    
    candidates.sort(key=lambda x: x["score"],reverse=True)
    print(f"--retrieved candidates for entity{entity['original_input']}")
    pprint(candidates)

    return [max(candidates,key=lambda x: x["score"])]



def get_candidates(input:str, k:int = 3)->List[RetrievedEntity]:
    """returns the top k possible candidates for a specific input accross all nodes"""
    candidates = search_vector.similarity_search_with_score(query=input,k=k)
    return [document_to_schema(doc,score) for doc,score in candidates]


def retrieve_entities(input_segments:List[str]):
    retrieved_entities = {}
    for seg in input_segments:
        retrieved_entities[seg] = get_candidates(seg)
    return retrieved_entities




