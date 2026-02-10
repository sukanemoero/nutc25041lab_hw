from pathlib import PosixPath
from typing import Union
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, VlmPipelineOptions
from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions
from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption, WordFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline


def get_docling_pdf(path: Union[PosixPath, str]):
    pdf_options = PdfPipelineOptions(
        # do_ocr=False,
    )
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)
        }
    )
    result = doc_converter.convert(path).document.export_to_markdown()
    return result
def get_converter(pipeline_options):
    opt = {
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options,
            pipeline_cls=VlmPipeline,
        ),
        InputFormat.IMAGE: ImageFormatOption(
            pipeline_options=pipeline_options,
            pipeline_cls=VlmPipeline,
        ),
        InputFormat.DOCX: WordFormatOption(
            pipeline_options=pipeline_options 
        )
    }
    return DocumentConverter(
        format_options=opt,
    )


def docling_with_olm(path: Union[PosixPath, str], olm_config: ApiVlmOptions):
   
    pipeline_options = VlmPipelineOptions(
        enable_remote_services=True  
    )
    pipeline_options.vlm_options = olm_config
   
    mds = get_converter(pipeline_options).convert(path).document.export_to_markdown()
    return mds
