
from pathlib import Path

from src.docling import docling_with_olm
from src.vlm import olmocr2_vlm_options

PDFPATH = Path(__file__).resolve().parent / 'doc' / 'sample_table.pdf'
RESULTPATH= Path(__file__).resolve().parent / 'results' 
def main():
    opt = olmocr2_vlm_options(
        prompt="Convert this page to clean, readable markdown format.",
        temperature=0.0,
    )

    dp = docling_with_olm(PDFPATH, opt)
    
    with open(RESULTPATH / 'docling_olm.md', mode='w') as f:
        f.write(dp)


if __name__ == "__main__":
    main()
