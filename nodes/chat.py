from datetime import datetime
from os import getenv
from typing import Literal
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from models.builder import builder
from prompt.prompt import load_prompt
from utils.configuration import ChatModelConfig, Metadata
from utils.state import State


async def chat_node(
    state: State, config: RunnableConfig
) -> Command[Literal["chat", "planner"]]:
    metadata = Metadata.get_from_config(config)
    llm_config: ChatModelConfig = metadata.llms_config.get("BASIC", None)

    if not llm_config:
        raise ValueError("Basic llm is not exist.")
    documents = state.get("knowledge_base", [])
    query = state.get("query", "")
    llm = builder(**(llm_config.model_dump()))

    syspromt = SystemMessage(
        await load_prompt(
            "CHAT",
            locale=getenv("locale", "zh-tw"),
            current_time=datetime.now().isoformat(),
        )
    )
    userquery = HumanMessage(
        await load_prompt("CHAT_INPUT", documents=documents, query=query)
    )
    result = await llm.ainvoke([syspromt, userquery])
    return Command(goto="__end__", update={"answer": result.content})
