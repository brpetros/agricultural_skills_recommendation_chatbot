from typing import List, Dict, TypedDict, Tuple
from langchain_core.documents import Document
from context_retrieval.vectors import occupation_label_vector, skill_label_vector, job_title_vector
from langchain_core.tools import tool
from context_retrieval.schema import RetrievedInfoSchema


def document_to_schema(doc: Document, score: float) -> RetrievedInfoSchema:
    """converts a langchain document into a structured schema"""

    metadata = doc.metadata or {} 

    return {
        "label": doc.page_content,
        "description": metadata.get("description",""),
        "id": metadata.get("id",""),
        "type": metadata.get("type","Job offer"),
        "score": score
    }

def get_occupations_by_label(input:str, k:int = 2) -> List[RetrievedInfoSchema]:
    """search for occupations in the graph based on the label"""
    results = occupation_label_vector.similarity_search_with_score(query=input,k=k)
    return [document_to_schema(doc,score) for doc,score in results]

def get_skills_by_label(input:str, k:int = 2) -> List[RetrievedInfoSchema]:
    """search for skills in the graph based on the label"""
    results = skill_label_vector.similarity_search_with_score(query=input,k=k)
    return [document_to_schema(doc,score) for doc,score in results]

def get_jobs_by_label(input:str, k:int = 2) -> List[RetrievedInfoSchema]:
    """search for jobs in the graph based on the label"""
    results = job_title_vector.similarity_search_with_score(query=input,k=k)
    return [document_to_schema(doc,score) for doc,score in results]
