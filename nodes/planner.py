from datetime import datetime
from os import getenv
from typing import Literal
from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from prompt.prompt import load_prompt
from models.builder import builder, embeddings
from tools.searxng import get_searxng_tool
from utils.configuration import ChatModelConfig, EmbeddingConfig, Metadata
from utils.state import State
from utils.vector import LocalVector


async def planner_node(
    state: State, config: RunnableConfig
) -> Command[Literal["coordinator"]]:

    metadata = Metadata.get_from_config(config)
    llm_config: ChatModelConfig = metadata.llms_config.get("BASIC", None)
    embed_config: EmbeddingConfig = metadata.llms_config.get("EMBED", None)

    plan = state.get("plan", {})
    update = {"plan": {}}
    if not llm_config:
        raise ValueError("Basic llm is not exist.")
    llm = builder(**(llm_config.model_dump()))
    embed = embeddings(**(embed_config.model_dump()))
    vector = LocalVector(embed)
    syspromt = SystemMessage(
        await load_prompt(
            "PLANNER",
            locale=getenv("locale", "zh-tw"),
            current_time=datetime.now().isoformat(),
        )
    )
    userquery = HumanMessage(
        await load_prompt(
            "PLANNER_INPUT",
            title=plan.get("title", "<NULL/>"),
            description=plan.get("description", "<NULL/>"),
        )
    )

    agent = create_agent(
        llm, [get_searxng_tool(metadata.searxng_config)], system_prompt=syspromt
    )
    result = await agent.ainvoke({"messages": [userquery]})
    message: BaseMessage = result["messages"][-1]
    documents = await vector.add_text(message.content)
    update["knowledge_base"] = documents
    return Command(goto="coordinator", update=update)
