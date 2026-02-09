from asyncio import gather, run, to_thread
from datetime import datetime
from io import StringIO
from pathlib import Path, PosixPath
from dotenv import load_dotenv
from os import environ as env, getenv
from typing import Any, Optional, Union
from aiofiles import open
from langchain_core.messages import merge_message_runs
from httpx import AsyncClient
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_qdrant import FastEmbedSparse
from utils.logger import logger
from pandas import DataFrame, read_csv
from models.builder import builder, embeddings
from prompt.prompt import load_prompt
from utils.qdrant import LocalQdrant
from utils.reranker import Reranker
from utils.spliter import Splitter

ROOT = Path(__file__).resolve().parent
HWPATH = ROOT / "day6"
DATA: list[PosixPath] = [
    HWPATH / "data_01.txt",
    HWPATH / "data_02.txt",
    HWPATH / "data_03.txt",
    HWPATH / "data_04.txt",
    HWPATH / "data_05.txt",
]
CLIENT = AsyncClient(timeout=1000)
HWAPI_URL = "https://hw-01.wade0426.me/submit_answer"
REWRITE_QUESTIONS = HWPATH / "questions.csv"
REWRITE_ANSWER = HWPATH / "questions_answer.csv"
MODEL_PATH = ROOT / "llms"


async def csv_reader(path: Union[PosixPath, str]) -> DataFrame:
    logger.debug(f"Reading CSV file from path: {path}")
    async with open(path) as f:
        content = await f.read()
        with StringIO(content) as str_io:
            return await to_thread(read_csv, str_io)


def conf_from_env(
    conf_prefix: str, subs: Optional[Union[tuple[str], list[str], set[str]]] = None
):
    def _get_env_conf(sub_name: Optional[str] = None) -> dict[str, Any]:
        prefix = (
            f"{conf_prefix.upper()}_{sub_name.upper()}__"
            if sub_name
            else f"{conf_prefix.upper()}__"
        )
        conf = {}

        def append_to_conf(term: str, key: str, value: Any) -> None:
            if key.startswith(term):
                conf_key = key[len(term) :].lower()
                conf[conf_key] = value

        for key, value in env.items():
            append_to_conf(prefix, key, value)
        return conf

    conf: dict[str, dict[str, Any]] = {}
    if subs:
        for t in subs:
            conf[t] = _get_env_conf(t)
    else:
        conf = _get_env_conf()
    return conf


async def query_rewrite(config: dict[str, Any], queries: list):
    logger.info("Invoking LLM for query rewriting...")
    llm = builder(**config)
    prompt = await load_prompt("REWRITE")
    r = await llm.ainvoke(
        merge_message_runs(
            [SystemMessage(prompt)]
            + [
                HumanMessage(query) if isinstance(query, str) else query
                for query in queries
            ]
        )
    )
    return r


async def amain():
    load_dotenv()

    try:
        r: DataFrame = await csv_reader(REWRITE_QUESTIONS)
        logger.info(f"Successfully loaded {len(r)} questions from {REWRITE_QUESTIONS}")
    except Exception as e:
        logger.error(f"Critical failure loading questions CSV: {e}")
        return

    v = [tuple(temp.tolist()[:2]) for temp in r.values]

    conf = conf_from_env("MODEL", ["EMBED", "BASIC"])
    conf["EMBED"]["embed"] = True

    logger.info("Initializing Embedding model and Reranker...")
    embed = embeddings(**conf["EMBED"])

    try:
        reranker = Reranker(
            model_path=str((MODEL_PATH / "Qwen3-Reranker-0.6B").resolve()),
            top_n=3,
            model_kwargs={
                "device_map": "auto",
                "torch_dtype": "auto",
            },
        )
        dims = len((await embed.aembed_query("Apple")))
        logger.info(f"Embeddings ready. Dimensions: {dims}")
    except Exception as e:
        logger.exception(f"Failed to initialize models (Embedding/Reranker): {e}")
        return

    logger.info("Connecting to Local Qdrant instance...")
    lq = LocalQdrant(
        embeddings=embed,
        dims=dims,
        **conf_from_env("QDRANT"),
        sparse_embeddings=FastEmbedSparse(model_name="Qdrant/bm25"),
        force_recreate=True,
    )

    async def _read(path):
        try:
            async with open(path) as f:
                t = await f.read()
            return (path.name, t)
        except Exception as e:
            logger.error(f"Error reading source file {path}: {e}")
            return (path.name, "")

    logger.info(f"Loading {len(DATA)} context files...")
    ts = await gather(*[_read(p) for p in DATA])

    def _process(t):
        if not t[1]:
            return []
        metadata = {"name": t[0]}
        logger.debug(f"Splitting document: {t[0]}")
        text_split = Splitter.split_semantic_texts([t[1]])
        stext_metadata = metadata.copy() | {"type": "stext"}

        return [
            Document(page_content=chunk, metadata=stext_metadata)
            for chunk in text_split
        ]

    logger.info("Chunking documents via semantic splitter...")
    processed_results = await gather(*[to_thread(_process, t) for t in ts])
    queue = [doc for sublist in processed_results for doc in sublist]

    logger.info(f"Total chunks generated: {len(queue)}")

    try:
        await lq.qdrant().aadd_documents(queue)
        logger.info("Vector database successfully populated.")
    except Exception as e:
        logger.error(f"Vector DB ingestion failed: {e}")
        return

    queries_map = {}
    csv_data = []

    for qi, q in v:
        ci = 1
        queries_map[ci] = queries_map.get(ci, {}) | {qi: q}

    async def _invoke(ci, qiq):
        qs = []
        re = []
        try:
            items = sorted(list(qiq.items()))
            logger.info(f"Processing conversation {ci} ({len(items)} steps)")

            for qi, q in items:
                logger.info(f"  > Processing Query ID: {qi}")

                qs.append(
                    HumanMessage(
                        await load_prompt("CW3_INPUT", query=q),
                        additional_kwargs={"question_id": qi},
                    )
                )
                rewritten_query = await query_rewrite(conf["BASIC"], qs)
                logger.debug(f"  > Rewritten query: {rewritten_query.content[:50]}...")

                sr = await lq.qdrant().asimilarity_search_with_relevance_scores(
                    rewritten_query.content, k=5
                )
                logger.debug(f"  > Retrieved {len(sr)} initial candidates.")

                sr = await reranker.acompress_documents([s[0] for s in sr], q)
                logger.debug(f"  > Reranker narrowed down to {len(sr)} documents.")

                sources = [s.metadata.get("name", "unknown") for s in sr]

                system_prompt = SystemMessage(
                    await load_prompt(
                        "CW3",
                        locale=getenv("locale", "zh-tw"),
                        current_time=datetime.now().isoformat(),
                        rag_database_content="\n".join([s.page_content for s in sr]),
                    )
                )

                llm = builder(**conf["BASIC"])
                response = await llm.ainvoke([system_prompt] + qs)

                qs.append(response)
                re.append(
                    {
                        "題目_ID": qi,
                        "題目": q,
                        "標準答案": response.content,
                        "來源文件": sources,
                    }
                )

            logger.info(f"Finished conversation {ci}.")
            return re

        except Exception as e:
            logger.exception(f"Error in conversation {ci} loop: {e}")
            return []

    logger.info("Starting concurrent inference workers...")
    all_results = await gather(*[_invoke(ci, qiq) for ci, qiq in queries_map.items()])

    for result_group in all_results:
        csv_data.extend(result_group)

    logger.info(f"Task complete. Total answers generated: {len(csv_data)}")

    try:
        DataFrame(csv_data).to_csv(REWRITE_ANSWER, index=False)
        logger.info(f"Results saved to: {REWRITE_ANSWER}")
    except Exception as e:
        logger.error(f"Failed to write results to CSV: {e}")


if __name__ == "__main__":
    run(amain())
