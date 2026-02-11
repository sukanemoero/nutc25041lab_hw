from pathlib import PosixPath
from typing import Union
from markitdown import MarkItDown

def get_markitdown_pdf(path: Union[PosixPath, str]):
    md = MarkItDown()
    result = md.convert(path).markdown
    return result
