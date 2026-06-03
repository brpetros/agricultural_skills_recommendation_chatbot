from typing import TypedDict, List, Literal

class InputEntity(TypedDict):
    """
        Schema for the entities that the user want information about.
        The LLM is initially instructed to extract those entities.
    """
    original_input: str
    category: Literal["occupation","job","skill","location","unknown"]

class RetrievedEntity(TypedDict):
    """Schema for retrieved occupations, skills, jobs, locations"""
    original_input: str  #what the user wants lo look for - before retrieval 
    category: Literal["occupation","job","skill","location","unknown"]
    id: str
    label: str
    description: str
    type: str
    score: float

"""
class RetrievedEntitiesSchema(TypedDict):
    
    occupations : List[EntitySchema]
    jobs : List[EntitySchema]
    skills : List[EntitySchema]
"""