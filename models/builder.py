from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

PLATFORM = Literal["NVIDIA", "OPENAI"]
PLATFORM_EMBEDDINGS = {"NVIDIA": NVIDIAEmbeddings, "OPENAI": OpenAIEmbeddings}
PLATFORM_CHATMODELS = {
    "NVIDIA": None,
    "OPENAI": ChatOpenAI,
}


def builder(platform: PLATFORM, *, api_key="<NULL/>", **kwargs):
    if platform := PLATFORM_CHATMODELS.get(platform):
        return platform(api_key=api_key, **kwargs)
    raise ValueError(f"Invalid platform: {platform}")


def embeddings(platform: PLATFORM, *, api_key="<NULL/>", dimensions=None, **kwargs):
    if platform := PLATFORM_EMBEDDINGS.get(platform):
        embed = platform(api_key=api_key, dimensions=dimensions, **kwargs)
        return embed
    raise ValueError(f"Invalid platform: {platform}")
