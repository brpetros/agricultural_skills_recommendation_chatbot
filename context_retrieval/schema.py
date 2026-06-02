from typing import TypedDict, List

class InputEntitiesSchema(TypedDict):
    """
        Schema for the entities that the user want information about.
        The LLM is initially instructed to extract those entities.
    """
    jobs: List[str]
    skills: List[str]
    occupations: List[str]

class EntitySchema(TypedDict):
    """Schema for retrieved occupations, skills or jobs"""
    original_input: str  #what the user wants lo look for - before retrieval 
    id: str
    label: str
    description: str
    type: str
    score: float


class RetrievedEntitiesSchema(TypedDict):
    """Schema for the entities after vector retrieval"""
    occupations : List[EntitySchema]
    jobs : List[EntitySchema]
    skills : List[EntitySchema]
