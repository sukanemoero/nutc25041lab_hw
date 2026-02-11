from pathlib import Path
from langchain_core.prompts import PromptTemplate
import aiofiles


async def load_prompt(name: str, **kwargs):
    prompt = await get_prompt(name)
    return prompt.format(**kwargs)


async def get_prompt(name: str):
    template = ""
    async with aiofiles.open(
        Path(__file__).resolve().parent / f"{name.upper()}.md",
        mode="r",
        encoding="utf-8",
    ) as f:
        template = await f.read()
    prompt = PromptTemplate.from_template(template)
    return prompt
