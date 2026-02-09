from utils.deepevel import LocalDeepeval
from asyncio import gather, run, to_thread
from datetime import datetime
from io import StringIO
from pathlib import Path, PosixPath
from dotenv import load_dotenv
from os import environ as env, getenv
from typing import Any, Coroutine, Optional, Union
from aiofiles import open
from langchain_core.messages import merge_message_runs
from httpx import AsyncClient
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_qdrant import FastEmbedSparse
from utils.logger import logger
from pandas import DataFrame, read_csv, read_excel
from models.builder import builder, embeddings
from prompt.prompt import load_prompt
from utils.qdrant import LocalQdrant
from utils.reranker import Reranker
from utils.spliter import Splitter

ROOT = Path(__file__).resolve().parent
HWPATH = ROOT / "day6" / "HW"

DATA = [HWPATH / "qa_data.docx"]

CLIENT = AsyncClient(timeout=1000)
HWAPI_URL = "https://hw-01.wade0426.me/submit_answer"
REWRITE_QUESTIONS = HWPATH / "questions.csv.xlsx"
REWRITE_ANSWER = HWPATH / "questions_answer.csv.xlsx"
MODEL_PATH = ROOT / "llms"


async def csv_reader(path: Union[PosixPath, str]) -> Coroutine[Any, Any, DataFrame]:
    async with open(path) as f:
        with StringIO(await f.read()) as str_io:
            return await to_thread(read_csv, str_io)


async def xlsx_reader(path: Union[PosixPath, str]) -> Coroutine[Any, Any, DataFrame]:
    return await to_thread(read_excel, path)


async def get_docx_text(path):
    from docx import Document

    doc = await to_thread(Document, path)
    full_text = [para.text for para in doc.paragraphs]
    return "\n".join(full_text)


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

        for key, value in env.items():
            append_to_conf(prefix, key, value)

        return conf

    conf: dict[str, dict[str, any]] = {}
    if subs:
        for t in subs:
            conf[t] = _get_env_conf(t)
    else:
        conf = _get_env_conf()
    return conf


async def query_rewrite(config: dict[str, any], querys: list):
    llm = builder(**config)
    prompt = await load_prompt("REWRITE")
    r = await llm.ainvoke(
        merge_message_runs(
            [SystemMessage(prompt)]
            + [
                HumanMessage(query) if isinstance(query, str) else query
                for query in querys
            ]
        )
    )
    return r


async def amain():
    logger.info("Starting main process...")
    load_dotenv()

    try:
        r: DataFrame = await xlsx_reader(REWRITE_QUESTIONS)
        logger.info(f"Loaded {len(r)} rows from {REWRITE_QUESTIONS}")
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        return

    v = [tuple(temp.tolist()[:2]) for temp in r.values]

    conf = conf_from_env("MODEL", ["EMBED", "BASIC"])
    conf["EMBED"]["embed"] = True
    embed = embeddings(**conf["EMBED"])

    llm = builder(**conf["BASIC"])
    eval_tool = LocalDeepeval(llm)
    reranker = Reranker(
        model_path=str((MODEL_PATH / "Qwen3-Reranker-0.6B").resolve()),
        top_n=3,
        model_kwargs={
            "device_map": "auto",
            "torch_dtype": "auto",
        },
    )
    try:
        dims = len((await embed.aembed_query("Apple")))
        logger.info(f"Embeddings initialized. Vector dimensions: {dims}")
    except Exception:
        logger.exception("Failed to initialize embeddings")
        return
    lq = LocalQdrant(
        embeddings=embed,
        dims=dims,
        **conf_from_env("QDRANT"),
        sparse_embeddings=FastEmbedSparse(model_name="Qdrant/bm25"),
        force_recreate=True,
    )

    async def _read(path):
        try:
            t = await get_docx_text(path)
            return (path.name, t)
        except Exception as e:
            logger.error(f"Error reading source file {path}: {e}")
            return (path.name, "")

    logger.info(f"Reading {len(DATA)} source documents for context.")
    ts = await gather(*[_read(p) for p in DATA])
    queue = []

    def _process(t):
        if not t[1]:
            return []

        metadata = {"name": t[0]}

        logger.debug(f"Splitting content for document: {t[0]}")

        stext_split = Splitter.split_semantic_texts([t[1]])

        stext_metadata = metadata.copy() | {"type": "stext"}

        return [
            Document(page_content=splited, metadata=stext_metadata)
            for splited in stext_split
        ]

    logger.info("Processing document chunks...")
    processed_results = await gather(*[to_thread(_process, t) for t in ts])
    for temp in processed_results:
        queue += temp

    logger.info(f"Index prepared with {len(queue)} total document chunks.")

    try:
        await lq.qdrant().aadd_documents(queue)
        logger.info("Vector database population complete.")
    except Exception as e:
        logger.error(f"Vector DB ingestion failed: {e}")

    querys = {}
    csv_data = []

    for qi, q in v:
        ci = 1
        querys[ci] = querys.get(ci, {}) | {qi: q}

    logger.info(f"Grouped input into {len(querys)} unique conversations.")

    async def _invoke(ci, qiq):
        re = []
        qe = []
        try:
            items = sorted(list(qiq.items()))
            logger.info(f"Processing conversation {ci} with {len(items)} questions.")

            for qi, q in items:
                logger.debug(f"Rewriting query {qi} for conversation {ci}")
                qs = [
                    HumanMessage(
                        await load_prompt("CW3_INPUT", query=q),
                        additional_kwargs={"question_id": qi},
                    )
                ]

                r = await query_rewrite(conf["BASIC"], qs)

                sr = await lq.qdrant().asimilarity_search_with_relevance_scores(
                    r.content, k=5
                )
                logger.debug(f"Found {len(sr)} context chunks for query {qi}")

                sr = await reranker.acompress_documents([s[0] for s in sr], q)

                source = []
                for s in sr:
                    source += [s.metadata.get("name", "null")]

                prompt = SystemMessage(
                    await load_prompt(
                        "CW3",
                        locale=getenv("locale", "zh-tw"),
                        current_time=datetime.now().isoformat(),
                        rag_database_content="\n".join([s.page_content for s in sr]),
                    )
                )

                r = await llm.ainvoke([prompt] + qs)
                qe += [
                    await eval_tool.get_test_case(
                        q, r.content, [s.page_content for s in sr]
                    )
                ]
                re += [{"q_id": qi, "questions": q, "answer": r.content}]
            temp = eval_tool.evaluate_response(qe)
            for i in range(len(re)):
                re[i] |= temp[i].model_dump()
            logger.info(f"Completed conversation {ci}.")
            return re

        except Exception as e:
            logger.exception(f"Error processing conversation {ci}: {e}")
            return []

    logger.info("Starting concurrent LLM invocation...")
    for temp in await gather(*[_invoke(ci, qiq) for ci, qiq in querys.items()]):
        csv_data += temp

    logger.info(f"All processing finished. Collected {len(csv_data)} result rows.")

    try:
        DataFrame(csv_data).to_csv(REWRITE_ANSWER, index=False)
        logger.info(f"Successfully saved results to {REWRITE_ANSWER}")
    except Exception as e:
        logger.error(f"Failed to save CSV output: {e}")


if __name__ == "__main__":
    run(amain())
