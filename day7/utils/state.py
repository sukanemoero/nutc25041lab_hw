from typing_extensions import TypedDict
from operator import add as operator_add
from typing import List, Annotated, Literal


class State(TypedDict):
    query: str
    plan: dict[Literal["title", "description"], str]
    knowledge_base: Annotated[List[str], operator_add]
    steps: int
    answer: str
