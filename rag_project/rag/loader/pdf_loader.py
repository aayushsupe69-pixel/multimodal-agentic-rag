import io, re
import fitz  # PyMuPDF
from rag.loader.ocr import ocr_extract

# Minimum chars to consider text extraction "successful"
MIN_TEXT_THRESHOLD = 20


def extract_text(content: bytes) -> str:
    """
    Robust PDF text extraction:
      1. Try PyMuPDF (fitz) - very fast and reliable
      2. If text < 20 chars → fallback to OCR (scanned PDF)
      3. Clean text (remove extra whitespace)
      4. Raise error only if both fail
    """
    # ── Step 1: Try PyMuPDF ─────────────────────────────
    text = _try_fitz(content)

    if len(text) >= MIN_TEXT_THRESHOLD:
        cleaned = _clean_text(text)
        print(f"PDF_LOADER: PyMuPDF extracted {len(cleaned)} cleaned chars ✅")
        return cleaned

    print(f"PDF_LOADER: PyMuPDF got only {len(text)} chars → falling back to OCR...")

    # ── Step 2: OCR Fallback ────────────────────────────
    ocr_text = ocr_extract(content)

    if len(ocr_text) >= MIN_TEXT_THRESHOLD:
        cleaned = _clean_text(ocr_text)
        print(f"PDF_LOADER: OCR extracted {len(cleaned)} cleaned chars ✅")
        return cleaned

    # ── Step 3: Both failed ─────────────────────────────
    combined = text or ocr_text
    if combined:
        final_clean = _clean_text(combined)
        print(f"PDF_LOADER: ⚠️ Only got {len(final_clean)} chars (below threshold)")
        return final_clean

    print("PDF_LOADER: ❌ Both PyMuPDF and OCR returned empty text")
    return ""


def _try_fitz(content: bytes) -> str:
    """Extracts text using PyMuPDF (fitz). Returns empty string on failure."""
    try:
        doc = fitz.open(stream=content, filetype="pdf")
        text = " ".join(page.get_text() for page in doc)
        doc.close()
        return text.strip()
    except Exception as e:
        print(f"PDF_LOADER: PyMuPDF error: {e}")
        return ""


def _clean_text(text: str) -> str:
    """Removes extra white spaces, tab characters, and multiple newlines."""
    # Replace multiple spaces/tabs with a single space
    text = re.sub(r'[ \t]+', ' ', text)
    # Replace 3 or more newlines with just 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()
