from vectors import get_occupations_by_label
from typing import List, Dict, TypedDict
from langchain_core.documents import Document
from pprint import pprint

class ConnectedNodeSchema(TypedDict):
    """
    Dictionary for the data of the connected nodes retrieved. 
    Every for every node that is retrieved (Skill, Occupation or Job), we search for the connected Jobs, Occupations and Skills.
    From those, we only take the label (or title) and the description.
    """
    label: str
    description: str

class OccupationSchema(TypedDict):
    """Dictionary for each occupation"""
    label: str
    score: float
    description: str
    connected_skills: List[ConnectedNodeSchema]
    connected_jobs: List[ConnectedNodeSchema]
    broader_occupations: List[ConnectedNodeSchema]
    smaller_occupations: List[ConnectedNodeSchema]


def get_retrieved_occupation(doc: Document,score:float) -> OccupationSchema:
    """
    Converts a langchain document to a structured occupation schema
    """
    metadata = doc.metadata or {} 
    print("--getting occupation--")
    return {
        
        "label": doc.page_content,
        "score": score,
        "occupation_description":metadata.get("occupation_description",""),
        "connected_skills":  [
            ConnectedNodeSchema(
                label=skill[0],
                description=skill[1]
            )
            for skill in metadata.get("skills", []) 
        ],
        "connected_jobs": [
             ConnectedNodeSchema(
                label=job[0],
                description=job[1]
            )
            for job in metadata.get("connected_jobs", []) 
        ],
        "broader_occupations": [
            ConnectedNodeSchema(
                label=occ[0],
                description=occ[1]
            )
            for occ in metadata.get("broader_occupations", []) 
        ],
        "smaller_occupations": [
            ConnectedNodeSchema(
                label=occ[0],
                description=occ[1]
            )
            for occ in metadata.get("broader_occupations", []) 
        ]
    }

def convert_result(docs: Dict)->List[OccupationSchema]:
     print("--converting document--")
     return [get_retrieved_occupation(doc,score) for doc,score in docs]


result = get_occupations_by_label("Animal Behaviorist")
pprint(result)
res = convert_result(result)
with open("vector_search_execution.txt","w",encoding="utf-8") as f:
        f.write(str(result)) 

with open("vector_search_structured.txt","w",encoding="utf-8") as f:
        f.write(str(res)) 