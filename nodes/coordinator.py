from datetime import datetime
from os import getenv
from typing import Literal
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from models.builder import builder
from models.structured_invoke import structured_invoke
from prompt.prompt import load_prompt
from utils.configuration import ChatModelConfig, Metadata
from utils.state import State


COORDINATOR_TYPE = {"accept": bool, "plan": {"title": str, "description": str}}


async def coordinator_node(
    state: State, config: RunnableConfig
) -> Command[Literal["chat", "planner"]]:
    metadata = Metadata.get_from_config(config)
    llm_config: ChatModelConfig = metadata.llms_config.get("BASIC", None)

    step = state.get("steps", 0)
    update = {"steps": step + 1}
    if not llm_config:
        raise ValueError("Basic llm is not exist.")
    documents = state.get("knowledge_base", [])
    query = state.get("query", "")
    llm = builder(**llm_config.model_dump())
    syspromt = SystemMessage(
        await load_prompt(
            "COORDINATOR",
            locale=getenv("locale", "zh-tw"),
            current_time=datetime.now().isoformat(),
        )
    )
    userquery = HumanMessage(
        await load_prompt("COORDINATOR_USER_QUERY", documents=documents, query=query)
    )
    result = await structured_invoke(llm, COORDINATOR_TYPE, [syspromt, userquery])
    goto = "chat"

    if (
        step <= 3
        and not result.get("accept", False)
        and (plan := result.get("plan", {}))
        and isinstance(plan, dict)
        and plan.get("title", "")
    ):
        goto = "planner"
        update["plan"] = plan

    return Command(goto=goto, update=update)
