from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline


def olmocr2_vlm_options(
    model: str = "allenai/olmOCR-2-7B-1025-FP8",
    base_url: str = "http://ws-01.wade0426.me/v1",
    prompt: str = "Convert this page to markdown.",
    max_tokens: int = 4096,
    temperature: float = 0.0,
    api_key: str = "",) -> ApiVlmOptions:


    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
   
    options = ApiVlmOptions(
        url=f"{base_url}/chat/completions",
        params=dict(
            model=model,
            max_tokens=max_tokens,
        ),
        headers=headers,
        prompt=prompt,
        timeout=1000,  # olmocr2 可能需要較長處理時間
        scale=.5,  # 圖片縮放比例
        temperature=temperature,
        response_format=ResponseFormat.MARKDOWN,
        
    )
    return options
