import re
import pdfplumber
import docx
from io import BytesIO


def clean_extracted_text(text: str) -> str:
    text = re.sub(r'[^\x00-\x7F]+', ' ', text) # remove non-ASCII
    text = re.sub(r'\*{2,}', '', text) # remove markdown bold
    text = re.sub(r'\n{3,}', '\n\n', text) # collapse excessive newlines
    text = re.sub(r'[ \t]{2,}', ' ', text) # collapse spaces/tabs
    return text.strip().lower()


def extract_from_pdf(file_bytes: bytes) -> str:
    text_parts = []
    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    raw = '\n'.join(text_parts)
    return clean_extracted_text(raw)


def extract_from_docx(file_bytes: bytes) -> str:
    doc = docx.Document(BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    raw = '\n'.join(paragraphs)
    return clean_extracted_text(raw)


def extract_text(uploaded_file) -> tuple[str, str]:
    file_bytes = uploaded_file.read()
    filename = uploaded_file.name.lower()

    try:
        if filename.endswith('.pdf'):
            text = extract_from_pdf(file_bytes)
        elif filename.endswith('.docx'):
            text = extract_from_docx(file_bytes)
        else:
            return '', 'Unsupported file type. Please upload a PDF or DOCX file.'

        if len(text) < 50:
            return '', 'Extracted text is too short. The file may be scanned or image-based.'

        return text, ''

    except Exception as e:
        return '', f'Failed to extract text: {str(e)}'