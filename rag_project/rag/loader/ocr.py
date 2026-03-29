import re
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image


def ocr_extract(content: bytes) -> str:
    """
    Converts PDF pages to images and runs Tesseract OCR.
    Used as a fallback when text-based extraction fails.
    """
    try:
        images = convert_from_bytes(content, dpi=200)
        print(f"OCR: Converting {len(images)} pages to images...")

        all_text = []
        for i, img in enumerate(images):
            page_text = pytesseract.image_to_string(img)
            all_text.append(page_text)
            # Free memory immediately
            img.close()

        raw = "\n".join(all_text)
        cleaned = clean_ocr_text(raw)
        print(f"OCR: Extracted {len(cleaned)} chars from {len(images)} pages")
        return cleaned

    except Exception as e:
        print(f"ERROR [OCR]: {e}")
        return ""


def clean_ocr_text(text: str) -> str:
    """
    Cleans up noisy OCR output:
    - Collapse multiple whitespace
    - Normalizes line breaks
    """
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?m)^\s*\S\s*$", "", text)
    return text.strip()
