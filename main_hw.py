from os import environ
from uuid import uuid4
from utils.logger import logger
from httpx import AsyncClient
from dotenv import load_dotenv
from langchain_core.documents import Document
from pandas import read_csv, DataFrame
from aiofiles import open
from io import StringIO
from pathlib import Path, PosixPath
from typing import Any, Coroutine, Optional, Union
from asyncio import gather, to_thread, run
from qdrant_client.models import FieldCondition, MatchValue, Filter


from models.builder import embeddings
from utils.qdrant import LocalQdrant
from utils.spliter import Splitter


HWPATH = Path(__file__).resolve().parent / "day5"
DATA: list[PosixPath] = [
    HWPATH / "data_01.txt",
    HWPATH / "data_02.txt",
    HWPATH / "data_03.txt",
    HWPATH / "data_04.txt",
    HWPATH / "data_05.txt",
]
CLIENT = AsyncClient(timeout=1000)
HWAPI_URL = "https://hw-01.wade0426.me/submit_answer"
CSV = HWPATH / "1111032045_RAG_HW_01.csv"


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


async def csv_reader(path: Union[PosixPath, str]) -> Coroutine[Any, Any, DataFrame]:
    async with open(path) as f:
        with StringIO(await f.read()) as str_io:
            return await to_thread(read_csv, str_io)


async def search(lq, query, must, k=1):
    return lq.qdrant().similarity_search_with_score(
        query=query, k=k, filter=Filter(must=must)
    )


async def check(q_id, answer):
    return (
        await CLIENT.post(HWAPI_URL, json={"q_id": q_id, "student_answer": answer})
    ).json()


async def amain():
    logger.info("Starting execution and loading environment variables.")
    load_dotenv()

    try:
        r: DataFrame = await csv_reader(HWPATH / "questions.csv")
        logger.info(f"Successfully loaded {len(r)} questions for scoring.")
    except Exception as e:
        logger.error(f"Failed to load questions CSV: {e}")
        return

    v = [tuple(temp.tolist()[:2]) for temp in r.values]
    conf = conf_from_env("MODEL", ["EMBED", "BASIC"])
    conf["EMBED"]["embed"] = True

    embed = embeddings(**conf["EMBED"])
    try:
        dims = len((await embed.aembed_query("Apple")))
        logger.info(f"Embeddings initialized. Vector dimensions: {dims}")
    except Exception:
        logger.exception("Failed to initialize embeddings")
        return

    lq = LocalQdrant(embeddings=embed, dims=dims, **conf_from_env("QDRANT"))

    async def _read(path):
        try:
            async with open(path) as f:
                t = await f.read()
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

        char_split = Splitter.split_characters([t[1]])
        text_split = Splitter.split_texts([t[1]])
        stext_split = Splitter.split_semantic_texts([t[1]])

        char_metadata = metadata.copy() | {"type": "char"}
        text_metadata = metadata.copy() | {"type": "text"}
        stext_metadata = metadata.copy() | {"type": "stext"}

        return (
            [
                Document(page_content=splited, metadata=char_metadata)
                for splited in char_split
            ]
            + [
                Document(page_content=splited, metadata=text_metadata)
                for splited in text_split
            ]
            + [
                Document(page_content=splited, metadata=stext_metadata)
                for splited in stext_split
            ]
        )

    processed_results = await gather(*[to_thread(_process, t) for t in ts])
    for temp in processed_results:
        queue += temp

    logger.info(f"Index prepared with {len(queue)} total document chunks.")

    try:
        await lq.qdrant().aadd_documents(queue)
        logger.info("Vector database population complete.")
    except Exception as e:
        logger.error(f"Vector DB ingestion failed: {e}")

    async def _search(qid, q):
        types = ["char", "text", "stext"]
        logger.info(f"Retrieving contexts for qid: {qid}")
        result = {}
        try:
            res = await gather(
                *[
                    search(
                        lq,
                        q,
                        [
                            FieldCondition(
                                key="metadata.type", match=MatchValue(value=t)
                            )
                        ],
                        k=1,
                    )
                    for t in types
                ]
            )
            for i, r in enumerate(res):
                result[types[i]] = [rr[0] for rr in r]
        except Exception as e:
            logger.error(f"Context retrieval failed for qid {qid}: {e}")
        return result

    search_results = await gather(*[_search(qid, q) for qid, q in v])

    async def _check(qid, ans_docs):
        logger.info(f"Running automated score evaluation for qid: {qid}")

        def unpack_doc(docs):
            return "\n".join([d.page_content for d in docs])

        def get_doc_references(docs):
            return [d.metadata.get("name", "null") for d in docs]

        async def temp(t, n, qid, d):
            try:
                res = await check(qid, d)
                current_score = res.get("score", 0)
                logger.debug(f"QID {qid} ({t}) - Score received: {current_score}")
                return (t, n, res)
            except Exception as e:
                logger.error(f"Scoring error for qid {qid} method {t}: {e}")
                return (t, n, {})

        docs = tuple(ans_docs.items())
        re = await gather(
            *[temp(k, get_doc_references(v), qid, unpack_doc(v)) for k, v in docs]
        )
        for r in re:
            ans_docs[r[0]] = (r[1], r[2])
        return ans_docs

    logger.info("Initiating parallel score calculation for all retrieved results.")
    check_result = await gather(
        *[_check(v[i][0], ans_docs) for i, ans_docs in enumerate(search_results)]
    )

    csv_data = []
    for c in check_result:
        for method, val in c.items():
            id = uuid4()
            source, result = val

            match method:
                case "char":
                    chinese_method = "固定大小"
                case "text":
                    chinese_method = "滑動視窗"
                case "stext":
                    chinese_method = "語意切塊"
                case _:
                    chinese_method = "未知"

            csv_data += [
                {
                    "id": id,
                    "source": source,
                    "score": result.get("score"),
                    "retrieve_text": result.get("student_answer"),
                    "q_id": result.get("q_id"),
                    "method": chinese_method,
                }
            ]

    try:
        DataFrame(csv_data).to_csv(CSV, index=False)
        logger.info(f"Evaluation report successfully exported to {CSV}")
    except Exception as e:
        logger.error(f"CSV report generation failed: {e}")

    logger.info("Finished.")


print(__name__)
if __name__ == "__main__":
    run(amain())
