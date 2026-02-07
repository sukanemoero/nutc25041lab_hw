from langchain_text_splitters import (
    RecursiveCharacterTextSplitter as TextSplitter,
    MarkdownHeaderTextSplitter as MarkdownSplitter,
    CharacterTextSplitter,
    HTMLHeaderTextSplitter,
)
from semantic_text_splitter import TextSplitter as SemanticTextSplitter
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
from typing import List, Dict, Any

from utils.logger import logger

_ENCODING_BASE = "cl100k_base"
_HTML_TO_SPLIT_ON = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
    ("h4", "Header 4"),
]
_HEADERS_TO_SPLIT_ON = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]


class Splitter:
    @staticmethod
    def _get_text_spliter(chunk_size: int, chunk_overlap: int):
        return TextSplitter.from_tiktoken_encoder(
            encoding_name=_ENCODING_BASE,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    @staticmethod
    def _get_semantic_text_spliter(chunk_size: int, chunk_overlap: int):
        return SemanticTextSplitter.from_tiktoken_model(
            "gpt-3.5-turbo", capacity=chunk_size, overlap=chunk_overlap
        )

    @staticmethod
    def _get_markdown_spliter(split_on: list[tuple[str, str]]):
        return MarkdownSplitter(split_on)

    @staticmethod
    def _get_html_spliter(split_on: list[tuple[str, str]]):
        return HTMLHeaderTextSplitter(split_on)

    @staticmethod
    def _get_character_spliter(chunk_size: int):
        return CharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=_ENCODING_BASE,
            separator="",
            chunk_size=chunk_size,
            chunk_overlap=0,
            is_separator_regex=False,
        )

    @classmethod
    def create_document_by_messages(
        cls,
        messages: List[BaseMessage],
        metadatas: List[Dict[Any, Any]] = [],
    ):
        messages = [message.model_dump() for message in messages]
        texts = [message.pop("content") for message in messages]
        metadatas = [
            (messages[i] | metadatas[i]) if i < len(metadatas) else messages[i]
            for i in range(len(messages))
        ]

        documents = [
            Document(page_content=texts[i], metadata=metadatas[i])
            for i in range(len(messages))
        ]

        return documents

    @classmethod
    def split_messages(
        cls,
        messages: List[BaseMessage],
        chunk_size: int = 150,
        chunk_overlap: int = 20,
        metadatas: List[Dict[Any, Any]] = [],
    ):
        return cls.split_documents(
            cls.create_document_by_messages(messages, metadatas),
            chunk_size,
            chunk_overlap,
        )

    @classmethod
    def split_documents(
        cls, documents: List[Document], chunk_size: int = 400, chunk_overlap: int = 80
    ):
        splitter = cls._get_text_spliter(chunk_size, chunk_overlap)
        return cls._split_documents(splitter, documents)

    @staticmethod
    def _split_documents(splitter, documents):
        spletted = splitter.split_documents(documents)
        for i in range(len(spletted)):
            spletted[i].metadata |= {"index": i}
        return spletted

    @staticmethod
    def _split_texts(splitter, texts):
        chunks = []
        for text in texts:
            try:
                chunks += splitter.split_text(text)

            except Exception as e:
                logger.warning(f"Messages splitter has an error: {e}")
        return chunks

    @classmethod
    def split_texts(
        cls, texts: List[str], chunk_size: int = 150, chunk_overlap: int = 20
    ):
        if not texts:
            return []
        splitter = cls._get_text_spliter(chunk_size, chunk_overlap)
        return cls._split_texts(splitter, texts)

    @classmethod
    def split_html(cls, text: str):
        if not text:
            return []
        splitter = cls._get_html_spliter(_HTML_TO_SPLIT_ON)
        return cls._split_texts(splitter, [text])

    @classmethod
    def split_semantic_texts(
        cls, texts: List[str], chunk_size: int = 150, chunk_overlap: int = 20
    ):
        if not texts:
            return []
        splitter = cls._get_semantic_text_spliter(chunk_size, chunk_overlap)
        result = splitter.chunk_all(texts)
        temp = []
        for r in result:
            temp += r
        return temp

    @classmethod
    def split_characters(cls, texts: List[str], chunk_size: int = 150):
        if not texts:
            return []
        splitter = cls._get_character_spliter(chunk_size)
        return cls._split_texts(splitter, texts)

    @classmethod
    def split_markdown(cls, text: str):
        if not text:
            return []
        splitter = cls._get_markdown_spliter(_HEADERS_TO_SPLIT_ON)
        return cls._split_texts(splitter, [text])
