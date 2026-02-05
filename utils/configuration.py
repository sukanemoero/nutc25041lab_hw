from uuid import uuid4
from langchain_core.runnables import RunnableConfig
from typing import Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator
from models.builder import PLATFORM


class ChatModelConfig(BaseModel):
    platform: PLATFORM = "OPENAI"
    model_name: str = Field(...)
    temperature: float = Field(default=0.7, ge=0, le=2.0)
    max_tokens: Optional[int] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    streaming: bool = False


class EmbeddingConfig(BaseModel):
    platform: PLATFORM = "OPENAI"
    model_name: str = Field(...)
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    dimensions: Optional[int] = None
    chunk_size: int = 1000


class ConfBase(BaseModel):
    @classmethod
    def _type_dump(cls, t):
        if isinstance(t, dict):
            r = t
        elif isinstance(t, cls):
            r = t.model_dump(exclude_unset=True)
        else:
            raise TypeError("Config type error.")
        return r

    def __init__(self, **kwargs):
        inp = {}
        for k, w in kwargs.items():
            if w:
                inp[k] = w
        super().__init__(**inp)

    def __or__(self, o):
        merged = self.model_dump(exclude_unset=True) | self.__class__._type_dump(o)
        return self.__class__(**merged)


class _CustomConf(ConfBase):
    @classmethod
    def get_from_config(cls, config: RunnableConfig):
        name = cls.__name__.lower()
        if (data := config.get(name, None)) and isinstance(data, dict):
            return cls(**data)
        raise ValueError("Invalid config.")


class Metadata(_CustomConf):
    llms_config: dict[str, Union[ChatModelConfig, EmbeddingConfig]] = {}
    searxng_config: dict[str, Any] = {}

    @field_validator("llms_config", mode="before")
    @classmethod
    def transform(cls, v):
        if v is None:
            return v
        transformed = {}
        for key, config_data in v.items():
            upper_key = key.upper()

            if isinstance(config_data, dict):
                is_embedding = config_data.pop("embed", False)

                if is_embedding:
                    transformed[upper_key] = EmbeddingConfig(**config_data)
                else:
                    transformed[upper_key] = ChatModelConfig(**config_data)
            else:
                transformed[upper_key] = config_data

        return transformed


class Configurable(_CustomConf):
    thread_id: Optional[str] = Field(str(uuid4()), description="Thread's uuid.")
