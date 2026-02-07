from os import environ
from typing import Optional, Union

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from models.builder import embeddings
from utils import logger
from utils.qdrant import LocalQdrant
from asyncio import run


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


async def amain(inps, que):
    conf = conf_from_env("MODEL", ["EMBED"])["EMBED"]
    conf["embed"] = True

    embed: Embeddings = embeddings(**conf)
    dims = len((await embed.aembed_query("Apple")))

    lq = LocalQdrant(embeddings=embed, dims=dims, **conf_from_env("QDRANT"))

    await lq.qdrant().aadd_texts(inps)

    print(await lq.qdrant().asimilarity_search_with_score(query=que, k=5))


if __name__ == "__main__":
    load_dotenv()
    try:
        inps = [input(f"Enter query ({i + 1}): ") for i in range(5)]
        que = input("Query: ")

        run(amain(inps, que))
    except Exception:
        logger.logger.exception("haha")
