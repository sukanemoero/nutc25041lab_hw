from pdfplumber import open
from pathlib import Path
from typing import Union

def get_pdf(path: Union[Path, str]) -> str:
    md_content = []
    with open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            md_content.append(f"## Page {i + 1}\n")
            text = page.extract_text()
            if text:
                clean_text = text.replace('\n', '\n\n')
                md_content += [clean_text]
                
    return "\n---\n".join(md_content)
