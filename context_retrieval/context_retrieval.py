from typing import List, Dict, TypedDict, Tuple
from langchain_core.documents import Document
from context_retrieval.vectors import occupation_label_vector, skill_label_vector, job_title_vector
from langchain_core.tools import tool
from context_retrieval.schema import InputEntity, RetrievedEntity
from pprint import pprint

def document_to_schema(input:InputEntity, doc: Document, score: float) -> RetrievedEntity:
    """converts a langchain document into a structured RetrievedEntity schema"""

    metadata = doc.metadata or {} 

    return {
        "original_input": input["original_input"],
        "category":input["category"],
        "label": doc.page_content,
        "description": metadata.get("description",""),
        "id": metadata.get("id",""),
        "type": metadata.get("type","Job offer"),
        "score": score
    }

def get_occupations_by_label(input:InputEntity, k:int = 1) -> List[RetrievedEntity]:
    """search for occupations in the graph based on the label"""
    results = occupation_label_vector.similarity_search_with_score(query=input["original_input"],k=k)
    print("--retrieved occupations")
    print(results)
    return [document_to_schema(input,doc,score) for doc,score in results]

def get_skills_by_label(input:InputEntity, k:int = 1) -> List[RetrievedEntity]:
    """search for skills in the graph based on the label"""
    results = skill_label_vector.similarity_search_with_score(query=input["original_input"],k=k)
    print("--retrieved skills")
    print(results)
    return [document_to_schema(input,doc,score) for doc,score in results]

def get_jobs_by_label(input:InputEntity, k:int = 1) -> List[RetrievedEntity]:
    """search for jobs in the graph based on the label"""
    results = job_title_vector.similarity_search_with_score(query=input["original_input"],k=k)
    print("--retrieved jobs")
    print(results)
    return [document_to_schema(input,doc,score) for doc,score in results]

# dict of the search functions used depending on each entity's category
SEARCH_FUNCTIONS = {
    "occupation":get_occupations_by_label,
    "job":get_jobs_by_label,
    "skill":get_skills_by_label
}

def retrieve_entity(entity:InputEntity,k:int = 1)->List[RetrievedEntity]:
    """
    Runs the correct search function depending on the entity's category.
    If the category is unknown, runs all 3 search functions and returns the result with the best score.
    """
    category = entity["category"]

    if category !="unknown":
        return SEARCH_FUNCTIONS[category](entity,k)
    
    # category unknown -> searches all of the indexes
    candidates = []
    for search_function in SEARCH_FUNCTIONS.values():
        candidates.extend(search_function(entity,k))

    candidates.sort(key=lambda x: x["score"],reverse=True)
    print(f"--retrieved candidates for entity{entity['original_input']}")
    pprint(candidates)

    return [max(candidates,key=lambda x: x["score"])]