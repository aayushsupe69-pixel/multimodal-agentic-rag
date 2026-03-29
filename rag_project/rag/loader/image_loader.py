import io
import time
import requests
import traceback
from PIL import Image
from config import HUGGINGFACE_API_KEY as HF_KEY

def caption_image(image_bytes: bytes, retries=3, delay=2) -> str:
    """
    Robust image captioning:
    1. Preprocess: Resize (max 512px), Convert to RGB, Compress to JPEG.
    2. API Call: Raw requests with binary data for stability.
    3. Resilience: 3-attempt retry loop with fallback caption.
    """
    if not HF_KEY:
        raise ValueError("HUGGINGFACE_API_KEY not found. Please check your .env file.")

    # ── Step 1: Preprocess ──────────────────────────────
    try:
        processed_bytes = _preprocess_image(image_bytes)
        print(f"IMAGE_LOADER: Preprocessed {len(image_bytes)} bytes → {len(processed_bytes)} bytes")
    except Exception as e:
        print(f"IMAGE_LOADER: Preprocessing failed: {e}")
        processed_bytes = image_bytes # Fallback to original

    # ── Step 2: Retry Loop ──────────────────────────────
    url = "https://router.huggingface.co/hf-inference/models/Salesforce/blip-image-captioning-base"
    headers = {"Authorization": f"Bearer {HF_KEY}"}
    
    last_err = None
    for attempt in range(retries):
        try:
            print(f"IMAGE_LOADER: Sending API request (Attempt {attempt+1}/{retries})...")
            r = requests.post(url, headers=headers, data=processed_bytes, timeout=30)
            
            # Handle rate limiting or model loading
            if r.status_code in [429, 503]:
                wait = delay * (2 ** attempt)
                print(f"IMAGE_LOADER: API Busy ({r.status_code}). Retrying in {wait}s...")
                time.sleep(wait)
                continue
                
            r.raise_for_status()
            resp = r.json()
            
            # BLIP usually returns a list of dicts: [{"generated_text": "..."}]
            if isinstance(resp, list) and len(resp) > 0 and "generated_text" in resp[0]:
                caption = resp[0]["generated_text"]
                print(f"IMAGE_LOADER: Success! Caption: {caption}")
                return caption
            
            raise ValueError(f"Unexpected response format: {resp}")

        except Exception as e:
            last_err = e
            print(f"IMAGE_LOADER: Attempt {attempt+1} failed ({e})")
            if attempt < retries - 1:
                time.sleep(delay * (2 ** attempt))
            else:
                break

    # ── Step 3: Global Fallback ─────────────────────────
    print(f"IMAGE_LOADER: All {retries} attempts failed. Triggering fallback...")
    traceback.print_exc()
    return "Image processed but detailed caption unavailable"


def _preprocess_image(content: bytes, max_size=512) -> bytes:
    """Resizes, converts to RGB, and compresses image to JPEG."""
    img = Image.open(io.BytesIO(content))
    
    # Convert to RGB (handles PNG transparency / CMYK)
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    # Resize keeping aspect ratio
    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    # Save to buffer
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85, optimize=True)
    return buf.getvalue()
