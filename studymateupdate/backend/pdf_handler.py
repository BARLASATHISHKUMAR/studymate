import PyPDF2
from PIL import Image
import pytesseract
import fitz
from typing import Union, IO

def extract_text_from_pdf(pdf_source: Union[str, IO]) -> str:
    """
    Extract text from PDF using PyPDF2, fallback to OCR if needed.
    """
    text_parts = []

    # Try PyPDF2
    try:
        reader = PyPDF2.PdfReader(pdf_source)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_parts.append(page_text.strip())
    except Exception:
        pass

    # OCR fallback
    if not text_parts:
        try:
            doc = fitz.open(pdf_source)
            for page in doc:
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_text = pytesseract.image_to_string(img)
                if ocr_text.strip():
                    text_parts.append(ocr_text.strip())
        except Exception as e:
            raise RuntimeError(f"OCR PDF extraction failed: {e}")

    return "\n\n".join(text_parts).strip()
