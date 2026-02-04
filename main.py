from langchain_openai import ChatOpenAI
from langchain.agents import create_agent 
from langchain_core.prompts import PromptTemplate, load_prompt
from pathlib import Path
from typing import Optional, List
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
from os import getenv
from pydantic import BaseModel, Field
from datetime import datetime
from langchain_core.messages import HumanMessage
from uuid import uuid4
from openai import InternalServerError
from langchain.tools import tool
import asyncio
load_dotenv()
PLATFORM=["Facebook", "Instagram", "Linkin"]

model=ChatOpenAI(
    model_name="google/gemma-3-27b-it",
    base_url="https://ws-02.wade0426.me/v1",
    api_key="</NULL>",
    max_completion_tokens=100,
    temperature=0
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

    result = await event(**kwargs)

    stop_event.set()
    
    await timer_task
    return result 

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
    with open(Path(__file__).resolve().parent/'prompt'/f'{name.upper()}.md', mode='r', encoding="utf-8") as f:
        template = f.read()
    prompt = PromptTemplate.from_template(template)
    return prompt.format(**kwargs)


@tool
async def fake_tool() -> str:
    """
    if you want to do anything, just call it
    """
    return f"[!IMPOSSIBLE] Please reply directly."
    
class TavilySearchFormatted(TavilySearch):
    def _run(self, *args, **kwargs):
        return asyncio.create_task(self._arun(*args, **kwargs))

    async def _arun(self, *args, **kwargs):
        _call_id = str(uuid4())
        query = kwargs.get("query", "")
        tool_type = "search"

        try:
            result = await super()._arun(*args, **kwargs)
            # tidied = await self._result_tidy(result)
            # for k, ws in tidied.items():
            #     for w in ws:
            #         wc = w.copy()
            #         wc.pop("score", None)
        except Exception as e:
            raise e
        # print(tidied)
        # return tidied 
        print()
        print(result.get('answer'))
        return result.get('answer')

    async def _result_tidy(self, result):
        images = []

        for image in result.pop("images", []):
            url, description = self._content_filter(image)
            images += [{"url": url, "description": description}]

        image_response = []
        for image in images:
            url = image.get("url", "")
            description = image.get("description", "")
            image_response += [
                SourceSummerizing(
                    source=url if url else "",
                    description=description if description else "",
                ).model_dump()
            ]

        founds = result.get("results", [])
        founds.sort(key=lambda x: x.get("score", 1) * -1)
        found_response = []
        raw_response = []
        for found in founds:
            source = found.get("url", "")
            title = found.get("title", "")
            content = found.get("content", "")
            favicon = found.get("favicon", "")

            source = WebSource(
                source=source, title=title, content=content, favicon=favicon
            ).model_dump()
            score = found.get("score", -1)

            found_response += [FoundWebSource(**source, score=score).model_dump()]

            if raw := found.get("raw_content", ""):
                raw_response += [RawSource(**source, raw=raw).model_dump()]

        return {
            "images": image_response,
            "founds": found_response,
            "raws": raw_response,
        }
    def _content_filter(self, source):
        url = None
        description = None
        print(source)
        if isinstance(source, str):
            if source.startswith("http"):
                url = source
            else:
                description = source
        elif isinstance(source, dict):
            url = source.get("url", None)
            description = source.get("description", None)
        return url, description
tools = []
if (k:= getenv('TAVILY_API_KEY', None)):
    tavily = TavilySearchFormatted(
        tavily_api_key=k,
        extract_depth="advanced",
        include_favicon=True,
        max_results=3,
        include_answer=True,
    )
    tools += [tavily]
else:
    tools += [fake_tool]
    print('Run without tavily.')
agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=load_prompt('AGENT', current_time=datetime.now().isoformat(), locale=getenv("LOCALE",'zh-tw')),
)


def main():
    t = input('Now your topic: ')
    inp = [{"messages":[HumanMessage(load_prompt('INPUT', platform=p, topic=t))]} for p in PLATFORM]
    result = None
    try:
        result = asyncio.run(with_timer(agent.abatch, inputs=inp))
    except InternalServerError as e:
        print(f'\nModel Server Error: HTTP/{e.status_code}')
    if result:
        for i, r in enumerate(result):
            gr = r.get('messages',[None])[-1]
            print(f'{PLATFORM[i]}: ')
            if gr:
                print(gr.content)
            else:
                print('NULL')


if __name__ == "__main__":
    main()
