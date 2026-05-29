from typing import TypedDict, List

class InputEntitiesSchema(TypedDict):
    """
        Schema for the entities that the user want information about.
        The LLM is initially instructed to extract those entities.
    """
    jobs: List[str]
    skills: List[str]
    occupations: List[str]

class RetrievedInfoSchema(TypedDict):
    """Schema for retrieved occupations, skills or jobs"""
    id: str
    label: str
    description: str
    type: str
    score: float


class RetrievedEntitiesSchema(TypedDict):
    """Schema for the entities after vector retrieval"""
    occupations : List[RetrievedInfoSchema]
    jobs : List[RetrievedInfoSchema]
    skills : List[RetrievedInfoSchema]
