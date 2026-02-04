from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.prompts import PromptTemplate
from pathlib import Path, PosixPath
from typing import Optional, List, Union
from dotenv import load_dotenv
from os import getenv
from pydantic import BaseModel, Field
from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage, AIMessageChunk
from openai import InternalServerError
from langchain.tools import tool
from utils.maybe_this_this_speech_to_text import get_srt
import base64
import asyncio

load_dotenv()


model = ChatOpenAI(
    model_name="google/gemma-3-27b-it",
    base_url="https://ws-02.wade0426.me/v1",
    api_key="</NULL>",
    max_completion_tokens=100,
    temperature=0,
)


async def timing_shower(stop_event):
    start_time = asyncio.get_event_loop().time()

    while not stop_event.is_set():
        current_now = asyncio.get_event_loop().time()
        elapsed = current_now - start_time

        print(f"\rRunning for: {elapsed:.2f}s", end="", flush=True)

        await asyncio.sleep(0.05)


async def with_timer(event, **kwargs):
    stop_event = asyncio.Event()

    timer_task = asyncio.create_task(timing_shower(stop_event))

    async for chunk in event(**kwargs):
        yield chunk

    stop_event.set()

    await timer_task
    # return result


class SourceSummerizing(BaseModel):
    source: str = Field(..., description="The url of source.")
    description: str = Field(
        ..., description="The description description of the source."
    )
    original_content: Optional[str] = Field(
        None, description="Original content of this source."
    )


class WebSource(BaseModel):
    source: str = Field(..., description="The url of source.")
    title: Optional[str] = Field("", description="The title of source.")
    content: Optional[str] = Field("", description="The content of source.")
    favicon: Optional[str] = Field("", description="The url of favicon in source.")


class RawSource(WebSource):
    raw: str = Field(..., description="The raw of source.")


class FoundWebSource(WebSource):
    score: float = Field(..., description="The score of this search.")


class ImageResult(BaseModel):
    result: List[SourceSummerizing] = Field(..., description="The image result.")


def load_prompt(name: str, **kwargs):
    template = ""
    with open(
        Path(__file__).resolve().parent / "prompt" / f"{name.upper()}.md",
        mode="r",
        encoding="utf-8",
    ) as f:
        template = f.read()
    prompt = PromptTemplate.from_template(template)
    return prompt.format(**kwargs)


@tool
async def load_audio(path: Union[str, PosixPath] = "") -> str:
    """
    Transcribes an audio file into text using a multimodal AI model.

    This function reads a WAV file from the local path, encodes it into base64,
    and streams it to the model for transcription. If the primary model
    service encounters an InternalServerError, it automatically falls back
    to an alternative SRT generation method.

    Args:
        path (Union[str, PosixPath]): The file system path to the .wav audio file.

    Returns:
        str: The transcribed text content of the audio.
    """

    audio_b64 = None
    try:
        with open(path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")
        message = HumanMessage(
            content=[
                {
                    "type": "input_audio",
                    "input_audio": {"data": audio_b64, "format": "wav"},
                }
            ]
        )
        sysprompt = SystemMessage(load_prompt("AUDIO"))
        result = AIMessageChunk("")
        async for chunk in model.astream([sysprompt, message]):
            result += chunk
        return result.content
    except InternalServerError:
        result = await get_srt(path)
        return result


tools = [load_audio]
agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=load_prompt(
        "AGENT_AUDIO",
        current_time=datetime.now().isoformat(),
        locale=getenv("LOCALE", "zh-tw"),
    ),
)


def main():
    inp = {"messages": [HumanMessage(input())]}
    result = None

    async def _main():
        async for c in with_timer(agent.astream, input=inp):
            if (
                (model := c.get("model", None))
                and (messages := model.get("messages", []))
                and (message := messages[-1])
            ):
                if message.tool_calls:
                    print("\nTool calling...")
                else:
                    print("\n" + message.content)
            elif (
                (tools := c.get("tools", None))
                and (messages := tools.get("messages", []))
                and (message := messages[-1])
            ):
                print("\n" + message.content)

    try:
        result = asyncio.run(_main())
    except InternalServerError as e:
        print(f"\nModel Server Error: HTTP/{e.status_code}")
    if result:
        for i, r in enumerate(result):
            gr = r.get("messages", [None])[-1]
            if gr:
                print(gr.content)
            else:
                print("NULL")


if __name__ == "__main__":
    # print(asyncio.run(load_audio(Path(__file__).resolve().parent/'wav'/'Podcast_EP14.wav')))
    main()
