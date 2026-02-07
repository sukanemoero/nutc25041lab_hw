from os import environ
from typing import Optional, Union

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from utils.spliter import Splitter
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers.json import JsonOutputParser
from models.builder import builder, embeddings
from utils.logger import logger
from utils.qdrant import LocalQdrant
from asyncio import run, gather
from aiofiles import open
from pathlib import Path
from prompt.prompt import load_prompt

ROOT = Path(__file__).resolve().parent
FILE = ROOT / "text.txt"
HTML_PATH = ROOT / "table_html.html"
MD_PATH = ROOT / "table_txt.md"


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


def format_docs_with_metadata(docs):
    formatted = []
    for doc in docs:
        header = doc.metadata.get("Header 1", "Normal section")
        content = f"[{header}]: {doc.page_content}"
        formatted.append(content)
    return "\n\n".join(formatted)


async def amain(que):
    conf = conf_from_env("MODEL", ["EMBED", "BASIC"])
    conf["EMBED"]["embed"] = True

    embed: Embeddings = embeddings(**conf["EMBED"])
    logger.info("Embedding model has been built")
    llm: BaseChatModel = builder(**conf["BASIC"])
    logger.info("Base model has been built")
    logger.info("Dimention testing...")

    dims = len((await embed.aembed_query("Apple")))
    logger.info(f"Got dimention: {dims}")

    lq = LocalQdrant(embeddings=embed, dims=dims, **conf_from_env("QDRANT"))
    text = ""
    html_text = ""
    md_text = ""
    async with open(FILE, mode="r", encoding="utf-8") as f:
        text = await f.read()
    async with open(HTML_PATH, mode="r", encoding="utf-8") as f:
        html_text = await f.read()
    async with open(MD_PATH, mode="r", encoding="utf-8") as f:
        md_text = await f.read()

    char_split = Splitter.split_characters([text])
    text_split = Splitter.split_texts([text])
    semantic_text_split = Splitter.split_semantic_texts([text])

    md_text_split = Splitter.split_markdown(md_text)
    html_split = Splitter.split_html(html_text)

    merged_html = format_docs_with_metadata(html_split)
    merged_md = format_docs_with_metadata(md_text_split)

    v1prompt = SystemMessage(await load_prompt("CW2-1"))
    v2prompt = SystemMessage(await load_prompt("CW2-2"))
    v1prompts = ChatPromptTemplate.from_messages(
        [v1prompt, HumanMessage("# Input Data: \n\t{input}")]
    )
    v2prompts = ChatPromptTemplate.from_messages(
        [v2prompt, HumanMessage("# Input Data: \n\t{input}")]
    )

    class V2Output(BaseModel):
        question: str
        answer: str

    v1llm = v1prompts | llm
    v2llm = v2prompts | llm | JsonOutputParser(pydantic_object=V2Output)

    logger.info("Processing cw2...")
    prompts = [
        {"input": merged_html},
        {"input": merged_md},
    ]

    v1, v2 = await gather(v1llm.abatch(prompts), v2llm.abatch(prompts))

    logger.success("V1 process finished.")
    logger.info(f"HTTP > {v1[0].content}")
    logger.info(f"Markdwon > {v1[1].content}")

    logger.success("V2 process finished.")
    logger.info(f"HTTP \n - {'\n - '.join([str(v) for v in v2[0]])}")
    logger.info(f"Markdwon \n - {'\n - '.join([str(v) for v in v2[1]])}")

    # def _prompts(inps):
    #     return [
    #         {"input": m } for m in inps
    #     ]

    # htmlprompts = _prompts(html_split)
    # mdprompts = _prompts(md_text_split)

    # logger.info("Processing HTML v1...")
    # r = await v1llm.abatch(htmlprompts)

    queue = (
        Splitter.split_texts([v.content for v in v1])
        + char_split
        + text_split
        + semantic_text_split
    )
    logger.info("Insertint texts...")
    await lq.qdrant().aadd_texts(queue)
    logger.info("Searching texts...")
    print(await lq.qdrant().asimilarity_search_with_score(query=que, k=5))


if __name__ == "__main__":
    load_dotenv()
    try:
        que = input("Query: ")

        run(amain(que))
    except Exception:
        logger.exception("haha")
