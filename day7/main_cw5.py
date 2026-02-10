
from pathlib import Path

from src.markitdown import get_markitdown_pdf
from src.pdfplumber import get_pdf
from src.docling import get_docling_pdf

PDFPATH = Path(__file__).resolve().parent / 'doc' / 'example.pdf'
RESULTPATH= Path(__file__).resolve().parent / 'results' 
def main():
    mkd = get_markitdown_pdf(PDFPATH)
    pp = get_pdf(PDFPATH)

    dp = get_docling_pdf(PDFPATH)
    
    with open(RESULTPATH / 'markidown.md', mode='w') as f:
        f.write(mkd)
        
    with open(RESULTPATH / 'pdfplumber.md', mode='w') as f:
        f.write(pp)
    with open(RESULTPATH / 'docling.md', mode='w') as f:
        f.write(dp)


if __name__ == "__main__":
    main()
