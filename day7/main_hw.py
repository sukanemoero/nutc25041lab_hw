
from os import environ, getenv
from pathlib import Path
from typing import Optional, Union

from docling.datamodel.pipeline_options import VlmPipelineOptions
from langchain_qdrant import FastEmbedSparse
from src.docling import docling_with_olm, get_converter
from utils.qdrant import LocalQdrant
from models.builder import embeddings, builder
from src.vlm import olmocr2_vlm_options
# from src.scanner import converter
from utils.spliter import Splitter

HWPATH = Path(__file__).resolve().parent/'HW'
DATA = [
    HWPATH / '1.pdf',
    HWPATH / '2.pdf',
    HWPATH / '3.pdf',
    HWPATH / '4.png',
    HWPATH / '5.docx',
]


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
def file_process():
    conf = conf_from_env("MODEL", ["EMBED", "BASIC"])
    conf["EMBED"]["embed"] = True
    embed = embeddings(**conf["EMBED"])
    llm = builder(**conf["BASIC"])

    try:
        dims = len(embed.embed_query("Apple"))
    except Exception:
        return
    lq = LocalQdrant(
        embeddings=embed,
        dims=dims,
        **conf_from_env("QDRANT"),
        sparse_embeddings=FastEmbedSparse(model_name="Qdrant/bm25"),
        force_recreate=True,
    )
    opt = olmocr2_vlm_options(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        model="gemini-2.5-flash",
        api_key=getenv("API_KEY", ""),
        prompt="Convert this page to clean, readable markdown format.",
        temperature=0.0,
    )
    
    pipeline_options = VlmPipelineOptions(
        enable_remote_services=True  
    )
    pipeline_options.vlm_options = opt
    
    
    for d in DATA:
        md = docling_with_olm(d,opt) 
        mds = Splitter.split_semantic_texts([md])
        lq.qdrant().add_documents(mds)
    
        
if __name__ == '__main__':

    from dotenv import load_dotenv
    load_dotenv()
    file_process()


