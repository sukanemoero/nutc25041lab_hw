from asyncio import get_running_loop
from concurrent.futures import ThreadPoolExecutor
from torch import no_grad, stack, tensor
from torch.nn import functional
from typing import Sequence, Dict, Any, List
from pydantic import Field, PrivateAttr

from langchain_core.documents import BaseDocumentCompressor, Document
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.logger import logger


class Reranker(BaseDocumentCompressor):
    model_path: str = Field(description="Local path or HuggingFace ID of the model")
    model_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Arguments for AutoModel"
    )
    tokenizer_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Arguments for AutoTokenizer"
    )
    top_n: int = Field(default=3, description="Number of documents to return")
    max_length: int = Field(default=8192, description="Maximum context length")

    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    _token_true_id: int = PrivateAttr()
    _token_false_id: int = PrivateAttr()
    _prefix_tokens: List[int] = PrivateAttr()
    _suffix_tokens: List[int] = PrivateAttr()
    _executor: ThreadPoolExecutor = PrivateAttr(
        default_factory=lambda: ThreadPoolExecutor(max_workers=1)
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_model()
        self._prepare_tokens()

    def _load_model(self):
        logger.info(f"Loading reranker model from: {self.model_path}")

        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
        } | self.model_kwargs

        tokenizer_kwargs = {
            "trust_remote_code": True,
            "padding_side": "left",
        } | self.tokenizer_kwargs

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, **tokenizer_kwargs
            )

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path, **model_kwargs
            ).eval()

            logger.success("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _prepare_tokens(self):
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._token_false_id = self._tokenizer.convert_tokens_to_ids("no")
        self._token_true_id = self._tokenizer.convert_tokens_to_ids("yes")

        prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

        self._prefix_tokens = self._tokenizer.encode(prefix, add_special_tokens=False)
        self._suffix_tokens = self._tokenizer.encode(suffix, add_special_tokens=False)

    def _format_instruction(self, query: str, doc_content: str) -> str:
        instruction = (
            "Given a web search query, retrieve relevant passages that answer the query"
        )
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc_content}"

    @no_grad()
    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
    ) -> Sequence[Document]:

        if not documents:
            logger.warning("No documents provided for reranking")
            return []

        logger.debug(f"Reranking {len(documents)} documents for query: {query[:50]}...")

        pairs = [self._format_instruction(query, doc.page_content) for doc in documents]

        inputs = self._tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self.max_length
            - len(self._prefix_tokens)
            - len(self._suffix_tokens),
        )

        input_ids = []
        for ids in inputs["input_ids"]:
            input_ids += [self._prefix_tokens + ids + self._suffix_tokens]

        max_len = max(len(ids) for ids in input_ids)
        padded_input_ids = []
        for ids in input_ids:
            padded_input_ids += [
                [self._tokenizer.pad_token_id] * (max_len - len(ids)) + ids
            ]

        input_tensor = tensor(padded_input_ids).to(self._model.device)

        logits = self._model(input_ids=input_tensor).logits[:, -1, :]

        true_vector = logits[:, self._token_true_id]
        false_vector = logits[:, self._token_false_id]

        batch_scores = stack([false_vector, true_vector], dim=1)
        batch_scores = functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()

        for doc, score in zip(documents, scores):
            doc.metadata["relevance_score"] = score

        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        top_docs = [doc for doc, _ in scored_docs[: self.top_n]]
        logger.info(f"Reranking complete. Returning top {len(top_docs)} documents.")

        return top_docs

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
    ) -> Sequence[Document]:
        loop = get_running_loop()
        return await loop.run_in_executor(
            self._executor, self.compress_documents, documents, query
        )
