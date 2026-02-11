from datetime import datetime
import aiofiles
from io import StringIO
from httpx import AsyncClient
from asyncio import gather 
from utils.reranker import Reranker
from docling.datamodel.base_models import DocumentStream
from langchain_core.messages import HumanMessage, SystemMessage, merge_message_runs
from pandas import DataFrame, read_csv, read_excel
from os import environ, getenv
from pathlib import Path, PosixPath
from sys import exception
from typing import Any, Coroutine, Optional, Union
from prompt.prompt import load_prompt
from utils.deepevel import LocalDeepeval
# from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.datamodel.pipeline_options import VlmPipelineOptions
from langchain_qdrant import FastEmbedSparse
from src.docling import docling_with_olm, get_converter, get_docling_pdf
from src.pdf_convert import convert_image_to_pdf_buffer, convert_docx_to_pdf_buffer
from utils.qdrant import LocalQdrant
from models.builder import embeddings, builder
# from src.vlm import olmocr2_vlm_options
from utils.logger import logger
# from src.scanner import converter
from utils.spliter import Splitter
from src.vlm import olmocr2_vlm_options

from asyncio import to_thread

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
async def csv_reader(path: Union[PosixPath, str]) -> Coroutine[Any, Any, DataFrame]:
    async with aiofiles.open(path) as f:
        with StringIO(await f.read()) as str_io:
            return await to_thread(read_csv, str_io)


async def xlsx_reader(path: Union[PosixPath, str]) -> Coroutine[Any, Any, DataFrame]:
    return await to_thread(read_excel, path)


async def get_docx_text(path):
    from docx import Document

    doc = await to_thread(Document, path)
    full_text = [para.text for para in doc.paragraphs]
    return "\n".join(full_text)
TOPQ=5
TOPC=5
MAX_RETRY = 5
HWPATH = Path(__file__).resolve().parent/'HW'
DATA = [
    HWPATH / '5.docx',
    HWPATH / 'idk.png',
    HWPATH / '4.png',
    HWPATH / '1.pdf',
    HWPATH / '2.pdf',
    HWPATH / '3.pdf',
]
ROOT = Path(__file__).resolve().parent
CLIENT = AsyncClient(timeout=1000)
QUESTIONS = HWPATH / "test_dataset.csv"
ANSWER = HWPATH / "test_dataset_answer.csv"
DEEPEVAL_INPUT = HWPATH / "questions_answer.csv"
DEEPEVEL_OUTPUT = HWPATH / "deepevel.csv"
MODEL_PATH = HWPATH / "llms"

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
def file_process(lq):
    opt = olmocr2_vlm_options(
            base_url="http://127.0.0.1:5433/v1",
            model="llava-phi3:latest",
        api_key="38datbest",
        prompt="Convert this page to clean, readable markdown format.",
        temperature=0.0,
    )

    pipeline_options = VlmPipelineOptions(
        enable_remote_services=True  
    )
    pipeline_options.vlm_options = opt


    from src.scanner import scanner
    
    for d in DATA:
        # if d.name.endswith('.png'):
        #     d = DocumentStream(name=d.name+'.pdf',stream=convert_image_to_pdf_buffer(d))
        # elif d.name.endswith('.docx'):
        #     d = DocumentStream(name=d.name+'.pdf',stream=convert_docx_to_pdf_buffer(d))
        md = docling_with_olm(d,opt) 
        # md = get_docling_pdf(d)

        mds = Splitter.split_semantic_texts([md])
        print(d, 'aadd')

        l = []
        for m in mds:
            _, b, _ = scanner.scan(m)
            if b:
                l += [m]

        if not l:
            continue
        lq.qdrant().add_texts(l, metadatas=[{'path': d.name} for _ in range(len(l))])
        print(d, 'done')

async def llm_process(llm, lq):
    querys = {}
    test_querys = {}
    csv_data = []
    test_csv_data = []
    reranker = Reranker(
        model_path=str((MODEL_PATH / "Qwen3-Reranker-0.6B").resolve()),
        top_n=5,
        model_kwargs={
            "device_map": "auto",
            "torch_dtype": "auto",
        },
    )
    try:
        r: DataFrame = await csv_reader(DEEPEVAL_INPUT)
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        return

    try:
        qv: DataFrame = await csv_reader(QUESTIONS)
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        return

    v = [tuple(temp.tolist()[:2]) for temp in r.values]
    qv = [tuple(temp.tolist()[:2]) for temp in qv.values]
    da = [tuple(temp.tolist()[:3]) for temp in r.values]
    dda = {}
    eval_tool = LocalDeepeval(llm)
    for qi, q, a in da:
        dda[qi] = (q, a)
    for qi, q in v:
        ci = 1
        querys[ci] = querys.get(ci, {}) | {qi: q}

    for qi, q in qv:
        ci = 1
        test_querys[ci] = test_querys.get(ci, {}) | {qi: q}
    logger.info(f"Grouped input into {len(querys)} unique conversations.")

    async def _invoke(ci, qiq, eval=True):
        re = []
        qe = []
        try:
            items = sorted(list(qiq.items()))
            items = items[:min(len(items), TOPQ)]
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

                sr = await lq.qdrant().asimilarity_search_with_score(
                    '\n'.join(r.content) if isinstance(r.content, list) else r.content, k=5
                )
                logger.debug(f"Found {len(sr)} context chunks for query {qi}")
                
                # sr = [s[0] for s in sr]
                sr = await reranker.acompress_documents([s[0] for s in sr], q)
                source = [temp.metadata.get('path', '</>') for temp in sr]
                source = list(tuple(source))
                prompt = SystemMessage(
                    await load_prompt(
                        "CW3",
                        locale=getenv("locale", "zh-tw"),
                        current_time=datetime.now().isoformat(),
                        rag_database_content="\n".join([s.page_content for s in sr]),
                    )
                )

                r = await llm.ainvoke([prompt] + qs)
                if eval:
                    qe += [
                        await eval_tool.get_test_case(
                            dda[qi][0], '\n'.join(r.content) if isinstance(r.content, list) else r.content, [s.page_content for s in sr], dda[qi][1]
                        )
                    ]
                re += [{"q_id": qi, "questions": q, "answer": r.content, 'source': source}]
            if eval:
                temp = eval_tool.evaluate_response(qe)
                for i in range(len(re)):
                    re[i] |= temp[i].model_dump()
            logger.info(f"Completed conversation {ci}.")
            return re

        except Exception as e:
            logger.exception(f"Error processing conversation {ci}: {e}")
            return []

    logger.info("Starting concurrent LLM invocation...")
    for temp in (await gather(*[_invoke(ci, qiq) for ci, qiq in list(querys.items())[:TOPC]])):
        csv_data += temp
    for temp in (await gather(*[_invoke(ci, qiq, False) for ci, qiq in list(test_querys.items())[:TOPC]])):
        test_csv_data += temp
    logger.info(f"All processing finished. Collected {len(csv_data)} result rows.")

    try:
        DataFrame(csv_data).to_csv(DEEPEVEL_OUTPUT, index=False)
        DataFrame(test_csv_data).to_csv(ANSWER, index=False)
    except Exception as e:
        logger.error(f"Failed to save CSV output: {e}")


if __name__ == '__main__':

    from dotenv import load_dotenv
    from asyncio import run 

    conf = conf_from_env("MODEL", ["EMBED", "BASIC"])
    conf["EMBED"]["embed"] = True
    embed = embeddings(**conf["EMBED"])
    llm = builder(**conf["BASIC"])

    try:
        dims = len(embed.embed_query("Apple"))
    except Exception:
        exit()
    lq = LocalQdrant(
        embeddings=embed,
        dims=dims,
        **conf_from_env("QDRANT"),
        sparse_embeddings=FastEmbedSparse(model_name="Qdrant/bm25"),
        force_recreate=True,
    )
    try:
        load_dotenv()
        file_process( lq)
        run(llm_process(llm,lq))
    except Exception:
        logger.exception('123')


