from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langchain_core.runnables import RunnableConfig
from utils.configuration import Metadata, EmbeddingConfig
from models.builder import embeddings
from utils.state import State
from utils.vector import LocalVector
from typing import Literal


async def cache_checking_node(
    state: State, config: RunnableConfig
) -> Command[Literal["coordinator"]]:
    metadata = Metadata.get_from_config(config)
    embed_config: EmbeddingConfig = metadata.llms_config.get("EMBED", None)
    if not embed_config:
        raise ValueError("Embedding model not exist.")
    update = {"knowledge_base": []}
    query = state.get("query", None)
    if not query:
        raise ValueError("Query is empty.")

    model = embeddings(**embed_config.model_dump())
    vector = LocalVector(model)
    result = await vector.search(query[-1].content)
    for r in result:
        # if r[1] >= 0.5:
        #     break
        update["knowledge_base"] += [r[0]]
    return Command(goto="coordinator", update=update)
