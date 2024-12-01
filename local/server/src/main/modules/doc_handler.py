import PyPDF2
from io import BytesIO


def extract_text_from_pdf(pdf_bytes):
    text = ""
    pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text
