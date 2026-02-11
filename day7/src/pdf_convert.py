import io
import mammoth
from weasyprint import HTML
from weasyprint.urls import BytesIO
from PIL import Image



def convert_image_to_pdf_buffer(path):
    pdf_buffer = io.BytesIO()
    image1 = Image.open(path).convert('RGB')
    image1.save(
        pdf_buffer, 
        format='PDF', 
        save_all=True, 
    )
    pdf_buffer.seek(0)
    return pdf_buffer


def convert_docx_to_pdf_buffer(path)->BytesIO:
    with open(path, mode='rb') as f:
        html_result = mammoth.convert_to_html(f)
        pdf_buffer = io.BytesIO()
        HTML(string=html_result.value).write_pdf(pdf_buffer)
        
        pdf_buffer.seek(0)
    return pdf_buffer

