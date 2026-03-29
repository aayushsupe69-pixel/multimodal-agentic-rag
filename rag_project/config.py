import os
from dotenv import load_dotenv

# Load from .env in the project root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT_DIR, ".env"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")

# Validation
if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY not found in environment.")
if not HUGGINGFACE_API_KEY:
    print("WARNING: HUGGINGFACE_API_KEY not found in environment.")
