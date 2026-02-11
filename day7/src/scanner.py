from llm_guard.input_scanners import PromptInjection
from llm_guard.input_scanners.prompt_injection import MatchType
from docling.document_converter import DocumentConverter
from llm_guard.model import Model


converter = DocumentConverter()
scanner = PromptInjection(threshold=0.92, match_type=MatchType.SENTENCE)

