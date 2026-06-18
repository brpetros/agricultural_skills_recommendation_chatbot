from typing import TypedDict, List, Literal


# deprecated schema for input segments categosized by the llm 
@DeprecationWarning
class InputSegment(TypedDict):
    """
    Schema for each separate segment of the query that the user need information about.
    The LLM is instructed to extract those segments.

    replaced by a simple list of strings - no categorization
    """
    original_input: str
    category: Literal["occupation","job","skill","location","unknown"]

class RetrievedEntity(TypedDict):
    """Schema for retrieved occupations, skills, jobs, locations"""
    id: str
    label: str
    description: str
    type: str
    score: float

