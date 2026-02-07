from utils.logger import logger
from langchain_openai import ChatOpenAI
from typing import AnyStr, Dict, Literal, Optional, List, Union
from dotenv import load_dotenv
from os import environ
from langgraph.types import Command
from datetime import datetime
from langchain_core.messages import HumanMessage
from openai import InternalServerError
from graph.builder import graph_build
from utils.configuration import Configurable, Metadata
from langchain_core.runnables import RunnableConfig
import asyncio

from utils.state import State

load_dotenv()


model = ChatOpenAI(
    model_name="google/gemma-3-27b-it",
    base_url="https://ws-02.wade0426.me/v1",
    api_key="</NULL>",
    max_completion_tokens=100,
    temperature=0,
)

print_temp = ""


async def timing_shower(stop_event):
    start_time = asyncio.get_event_loop().time()

    while not stop_event.is_set():
        current_now = asyncio.get_event_loop().time()
        elapsed = current_now - start_time

        print(
            f"\rRunning for: {elapsed:.2f}s | {print_temp + ' --> '}",
            end="",
            flush=True,
        )

        await asyncio.sleep(0.05)

    print(f"\rRunning for: {elapsed:.2f}s | {print_temp}", end="", flush=True)
    print()


async def with_timer(event, **kwargs):
    stop_event = asyncio.Event()

    timer_task = asyncio.create_task(timing_shower(stop_event))

    async for chunk in event(**kwargs):
        yield chunk

    stop_event.set()

    await timer_task
    # return result


def build_async_workflow(
    configurable: Optional[Union[Configurable, Dict]] = None,
):
    async def workflow(
        input: Union[
            List[Union[AnyStr, HumanMessage, Dict[Literal["content"], AnyStr]]],
            AnyStr,
            HumanMessage,
            Dict[Literal["content"], AnyStr],
        ],
        metadata: Union[Metadata, Dict],
        configurable: Optional[Union[Configurable, Dict]] = configurable,
    ):
        configurable = (
            Configurable(**configurable)
            if isinstance(configurable, Dict)
            else configurable
        )

        thread_id = configurable.thread_id
        metadata = Metadata(**metadata) if isinstance(metadata, Dict) else metadata

        graph = graph_build()
        config = RunnableConfig(
            metadata=metadata.model_dump(),
            configurable=configurable.model_dump() if configurable else {},
        )

        now = datetime.now().microsecond
        formatted_queries = []
        if not isinstance(input, list):
            input = [input]
        for q in input:
            if isinstance(q, HumanMessage):
                q.name = "user_query"
                q.created = now
                formatted_queries += [q]
            elif isinstance(q, dict):
                formatted_queries += [
                    HumanMessage(
                        content=q.get("content", "<NULL/>"),
                        name="user_query",
                        created=now,
                    )
                ]
            else:
                formatted_queries += [
                    HumanMessage(content=q, name="user_query", created=now)
                ]
        state = {
            "query": formatted_queries,
        }

        if graph.get_state(config).interrupts:
            state_input = Command(resume=State(**state))
        else:
            state_input = Command(update=State(**state))

        stream_input = {"input": state_input, "config": config}
        async for s in graph.astream(**stream_input):
            yield s
        checkpointer = graph.checkpointer
        checkpointer.delete_thread(thread_id)

    return workflow


def conf_from_env(
    conf_prefix: str, subs: Optional[Union[tuple[str], list[str], set[str]]] = None
):
    def _get_env_conf(sub_name: Optional[str] = None) -> dict[str, any]:
        prefix = (
            f"{conf_prefix.upper()}_{sub_name.upper()}__"
            if sub_name
            else f"{conf_prefix.upper()}__"
        )
        conf = {}

        def append_to_conf(term: str, key: str, value: any) -> None:
            if key.startswith(term):
                conf_key = key[len(term) :].lower()
                conf[conf_key] = value

        for key, value in environ.items():
            append_to_conf(prefix, key, value)

        return conf

    conf: dict[str, dict[str, any]] = {}
    if subs:
        for t in subs:
            conf[t] = _get_env_conf(t)
    else:
        conf = _get_env_conf()
    return conf


def main():
    load_dotenv()
    try:
        workflow = build_async_workflow(Configurable())
        conf = conf_from_env("MODEL", ["BASIC", "EMBED"])
        conf["EMBED"]["embed"] = True

        async def _main():
            result = ""
            path = []
            async for c in with_timer(
                workflow,
                input=input("input: "),
                metadata=Metadata(
                    llms_config=conf, searxng_config=conf_from_env("SEARXNG")
                ),
            ):
                for k, v in c.items():
                    path += [k]
                    global print_temp
                    print_temp = " --> ".join(path)
                    if k == "chat":
                        result = v.get("answer")
            return result

        try:
            result = asyncio.run(_main())
            print(result)
        except InternalServerError as e:
            print(f"\nModel Server Error: HTTP/{e.status_code}")
    except Exception:
        logger.exception("haha")


if __name__ == "__main__":
    # print(asyncio.run(load_audio(Path(__file__).resolve().parent/'wav'/'Podcast_EP14.wav')))
    main()
